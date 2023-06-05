import torch
from utils.utils import read_config
from utils.model_utils import load_model_for_generation, generate
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, default="configs/generate.yaml")
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    config = read_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_for_generation(**config["model"])
    model.eval(); model.to(device)

    response = generate(
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        **config["generation"]
    )

    print(response)