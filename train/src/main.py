from dotenv import load_dotenv

load_dotenv()

import torch
import argparse

# Check if CUDA is available and print GPU information
print("CUDA available:", torch.cuda.is_available())
print(
    "GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train or generate data for a model.")
parser.add_argument("model")
parser.add_argument("--steps", type=int, default=100000, help="Number of steps")

args = parser.parse_args()

print("Train model: ", args.model)
print("Number of steps: ", args.steps)
print()

if args.model == "disney-rtnam":
    from train.disney_rtnam import train

    train(args.steps)
else:
    print(f"Error: Invalid command '{args.command}'.")
    exit(1)
