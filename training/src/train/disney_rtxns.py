import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
from data_gen.disney_rtxns import DisneyBRDFDataset


# training MLP for DisneyBRDF
def train(epochs):
    # MLP definition: 5 -> 128 -> 128 -> 128 -> 4
    class DisneyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(5, 64)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(64, 64)
            self.act2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 64)
            self.act3 = nn.ReLU()
            self.fc4 = nn.Linear(64, 4)
            self.act4 = lambda x: x.exp()

        def forward(self, x):
            x = self.act1(self.fc1(x))
            x = self.act2(self.fc2(x))
            x = self.act3(self.fc3(x))
            x = self.act4(self.fc4(x))
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisneyMLP().to(device)

    batch_size = 512

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

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

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

            loss = criterion(preds, batch_out)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_loss += loss.item()

                losses = (preds - batch_out).sum(dim=1).pow(2)
                max_index = losses.argmax()
                max_data = batch_out[max_index]
                if losses.max() > 0.05:
                    print(
                        f"  Iteration {i}/{len(dataloader)} Loss: {loss.item() / batch_size:.6f} MaxLoss: {losses.max():.6f} y: ({max_data[0]:.6f} {max_data[1]:.6f} {max_data[2]:.6f} {max_data[3]:.6f}) y_pred: ({preds[max_index][0]:.6f} {preds[max_index][1]:.6f} {preds[max_index][2]:.6f} {preds[max_index][3]:.6f}) roughness: {batch_in[max_index][4]:.6f} NdotH: {batch_in[max_index][2]:.6f}"
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
            "input": 5,
            "output": 4,
            "layers": [
                layer_to_json(model.fc1),
                {"type": "relu"},
                layer_to_json(model.fc2),
                {"type": "relu"},
                layer_to_json(model.fc3),
                {"type": "relu"},
                layer_to_json(model.fc4),
                {"type": "relu"},
            ],
        }
    }

    os.makedirs("output", exist_ok=True)
    with open("output/disney-rtxns.json", "w") as f:
        json.dump(model_json, f, indent=2)
