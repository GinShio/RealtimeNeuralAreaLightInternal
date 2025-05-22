import torch
import argparse

# Check if CUDA is available and print GPU information
print("CUDA available:", torch.cuda.is_available())
print(
    "GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a model.")
subparsers = parser.add_subparsers(
    dest="model", required=True, help="model name. one of: ['color', 'disney']"
)

# color sub command
parser_color = subparsers.add_parser("color", help="Train a color model.")
parser_color.add_argument("--epochs", type=int, default=500, help="Number of epochs")

# disney sub command
parser_disney = subparsers.add_parser("disney", help="Train disney BRDF model")
# parser_disney.add_argument(...)


args = parser.parse_args()

print("Model name:", args.model)

if args.model == "color":
    import model.color

    model.color.train(epochs=args.epochs)
elif args.model == "disney":
    # import model.disney
    # model.disney.train()
    pass
else:
    print(f"Error: Invalid model argument '{args.model}'.")
    exit(1)
