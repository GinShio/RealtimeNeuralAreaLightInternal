import argparse

# Check if CUDA is available and print GPU information
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a model.")
parser.add_argument("model", help="model name,. one of: 'color', 'disneyBRDF")

args = parser.parse_args()

print("Model name:", args.model)

with open("output/model_name.txt", "w") as f:
    f.write(args.model)
