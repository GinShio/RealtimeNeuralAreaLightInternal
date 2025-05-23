import torch
import argparse

# Check if CUDA is available and print GPU information
print("CUDA available:", torch.cuda.is_available())
print(
    "GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train or generate data for a model.")
subparsers = parser.add_subparsers(
    dest="command", required=True, help="Sub-command: train or data_gen"
)

# train sub command
parser_train = subparsers.add_parser("train", help="Train a model.")
train_subparsers = parser_train.add_subparsers(
    dest="model", required=True, help="Model name for training."
)

# train color
parser_train_color = train_subparsers.add_parser("color", help="Train a color model.")
parser_train_color.add_argument(
    "--epochs", type=int, default=500, help="Number of epochs"
)

# train disney
parser_train_disney = train_subparsers.add_parser(
    "disney", help="Train disney BRDF model"
)
parser_train_disney.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs"
)

# train disney
parser_train_disney = train_subparsers.add_parser(
    "disney-rtxns", help="Train disney BRDF model"
)
parser_train_disney.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs"
)

# data_gen sub command
parser_data_gen = subparsers.add_parser("data_gen", help="Generate data for a model.")
data_gen_subparsers = parser_data_gen.add_subparsers(
    dest="model", required=True, help="Model name for data generation."
)

# data_gen disney
parser_data_gen_disney = data_gen_subparsers.add_parser(
    "disney", help="Generate data for disney BRDF model"
)
# parser_data_gen_disney.add_argument(...)

# data_gen disney
parser_data_gen_disney = data_gen_subparsers.add_parser(
    "disney-rtxns", help="Generate data for disney BRDF model"
)
# parser_data_gen_disney.add_argument(...)

args = parser.parse_args()

print("Command:", args.command)
print("Model name:", args.model)

if args.command == "train":
    if args.model == "color":
        import train.color as color

        color.train(epochs=args.epochs)
    elif args.model == "disney":
        import train.disney as disney

        disney.train(epochs=args.epochs)
    elif args.model == "disney-rtxns":
        import train.disney_rtxns as disney

        disney.train(epochs=args.epochs)
    else:
        print(f"Error: Invalid model argument '{args.model}'.")
        exit(1)
elif args.command == "data_gen":
    if args.model == "disney":
        import data_gen.disney as disney

        disney.data_gen()
    if args.model == "disney-rtxns":
        import data_gen.disney_rtxns as disney

        disney.data_gen()
    else:
        print(f"Error: Invalid model argument '{args.model}'.")
        exit(1)
else:
    print(f"Error: Invalid command '{args.command}'.")
    exit(1)
