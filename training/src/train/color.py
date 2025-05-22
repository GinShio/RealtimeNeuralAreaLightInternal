import torch
import torch.nn as nn
import torch.optim as optim
import json
import os


# training MLP
def train(epochs):
    # definition of the MLP model
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 32)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(32, 32)
            self.act2 = nn.ReLU()
            self.fc3 = nn.Linear(32, 3)
            self.act3 = nn.Sigmoid()

        def forward(self, x):
            x = self.act1(self.fc1(x))
            x = self.act2(self.fc2(x))
            x = self.act3(self.fc3(x))
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)

    # generate random data
    num_samples = 10000
    batch_size = 256

    inputs = torch.rand(num_samples, 3, device=device)
    targets = inputs.clone()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            idx = perm[i : i + batch_size]
            batch_in = inputs[idx]
            batch_out = targets[idx]
            optimizer.zero_grad()
            preds = model(batch_in)
            loss = criterion(preds, batch_out)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_in.size(0)
        print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss / num_samples:.6f}")

    # パラメータをjson形式で保存
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
            "input": 3,
            "output": 3,
            "layers": [
                layer_to_json(model.fc1),
                {"type": "sigmoid"},
                layer_to_json(model.fc2),
                {"type": "sigmoid"},
                layer_to_json(model.fc3),
                {"type": "sigmoid"},
            ],
        }
    }

    os.makedirs("output", exist_ok=True)
    with open("output/color.json", "w") as f:
        json.dump(model_json, f, indent=2)
