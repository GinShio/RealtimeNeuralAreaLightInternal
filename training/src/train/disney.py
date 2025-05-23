import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
from data_gen.disney import DisneyBRDFDataset


# class FourierFeatureEmbedding(nn.Module):
#     def __init__(self, input_dim: int, num_frequencies: int, scale: float = 1.0):
#         """
#         Fourier Feature Embedding Layer

#         Parameters:
#             input_dim (int): 入力ベクトルの次元数
#             num_frequencies (int): 周波数の数（各次元に対するsin/cosのペア数）
#             scale (float): 周波数のスケーリングファクター（通常は1またはπなど）
#         """
#         super().__init__()
#         self.input_dim = input_dim
#         self.num_frequencies = num_frequencies
#         self.scale = scale

#         # 各次元に対して異なる周波数を割り当てる
#         frequencies = (
#             torch.linspace(1.0, 2.0 ** (num_frequencies - 1), num_frequencies) * scale
#         )
#         self.register_buffer("frequencies", frequencies)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Parameters:
#             x (Tensor): [batch_size, input_dim]のテンソル

#         Returns:
#             Tensor: [batch_size, input_dim * num_frequencies * 2]のFourier埋め込みテンソル
#         """
#         x = x.unsqueeze(-1)  # [B, input_dim, 1]
#         freqs = self.frequencies.to(x.device)  # [num_frequencies]
#         x_proj = x * freqs  # [B, input_dim, num_frequencies]

#         sin = torch.sin(2 * np.pi * x_proj)
#         cos = torch.cos(2 * np.pi * x_proj)
#         fourier_features = torch.cat(
#             [sin, cos], dim=-1
#         )  # [B, input_dim, num_frequencies * 2]

#         return fourier_features.view(x.shape[0], -1)  # フラット化


# training MLP for DisneyBRDF
def train(epochs):
    # # MLP definition: 23 -> 128 -> 128 -> 128 -> 3
    # MLP definition: 15 -> 128 -> 128 -> 128 -> 3
    class DisneyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            # self.fc1 = nn.Linear(23, 128)
            self.fc1 = nn.Linear(15, 128)
            # self.ffe = FourierFeatureEmbedding(15, 6, scale=1.0)
            # self.fc1 = nn.Linear(15 * 6 * 2, 128)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 128)
            self.act2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 128)
            self.act3 = nn.ReLU()
            self.fc4 = nn.Linear(128, 128)
            self.act4 = nn.ReLU()
            self.fc5 = nn.Linear(128, 3)
            self.act5 = nn.ReLU()

        def forward(self, x):
            # x = self.ffe(x)
            x = self.act1(self.fc1(x))
            x = self.act2(self.fc2(x))
            x = self.act3(self.fc3(x))
            x = self.act4(self.fc4(x))
            x = self.act5(self.fc5(x))
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisneyMLP().to(device)

    batch_size = 1024

    # train dataset
    dataset = DisneyBRDFDataset(test=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # test dataset
    test_dataset = DisneyBRDFDataset(test=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    criterion_importance = lambda y_pred, y, weights: (
        nn.MSELoss(reduction="none")(y_pred, y) * weights.unsqueeze(1)
    ).mean()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # train
        epoch_loss = 0.0
        model.train()
        for i, (batch_in, batch_out) in enumerate(dataloader):
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)
            optimizer.zero_grad()
            preds = model(batch_in)

            weights = (batch_out.max(dim=1).values).clamp(max=2.0, min=0.1)

            loss_importance = criterion_importance(preds, batch_out, weights)
            loss_importance.backward()
            optimizer.step()

            with torch.no_grad():
                loss = criterion(preds, batch_out)
                epoch_loss += loss.item()

                losses = (preds - batch_out).sum(dim=1).pow(2)
                max_index = losses.argmax()
                max_data = batch_out[max_index]
                if losses.max() > 0.05:
                    print(
                        f"  Iteration {i}/{len(dataloader)} Loss: {loss.item() / batch_size:.6f} MaxLoss: {losses.max():.6f} y: ({max_data[0]:.6f} {max_data[1]:.6f} {max_data[2]:.6f}) y_pred: ({preds[max_index][0]:.6f} {preds[max_index][1]:.6f} {preds[max_index][2]:.6f}) roughness: {batch_in[max_index][4]:.6f}"
                    )
        train_loss = epoch_loss / len(dataset)

        # test
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_in, batch_out in test_dataloader:
                batch_in = batch_in.to(device)
                batch_out = batch_out.to(device)
                preds = model(batch_in)
                loss = criterion(preds, batch_out)
                test_loss += loss.item()
        test_loss = test_loss / len(test_dataset)

        print(
            f"Epoch {epoch+1}/{epochs} Loss: {train_loss:.6f}  Test Loss: {test_loss:.6f}"
        )

    # Save model parameters to JSON
    def layer_to_json(layer):
        return {
            "type": "linear",
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "weights": layer.weight.detach().cpu().numpy().flatten().tolist(),
            "bias": layer.bias.detach().cpu().numpy().flatten().tolist(),
        }

    model_json = {
        "model": {
            # "input": 23,
            "input": 15,
            "output": 3,
            "layers": [
                layer_to_json(model.fc1),
                {"type": "relu"},
                layer_to_json(model.fc2),
                {"type": "relu"},
                layer_to_json(model.fc3),
                {"type": "relu"},
                layer_to_json(model.fc4),
                {"type": "relu"},
                layer_to_json(model.fc5),
                {"type": "relu"},
            ],
        }
    }

    os.makedirs("output", exist_ok=True)
    with open("output/disney.json", "w") as f:
        json.dump(model_json, f, indent=2)
