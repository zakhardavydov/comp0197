import torch

from utils import parse_args
from train import train, get_dataloader
from model import get_model


if __name__ == "__main__":

    # For reproducibility
    torch.manual_seed(42)

    batch_size = 64
    epochs = 100
    lr = 0.001

    vit_16_params = {
        "image_size": 32,
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_classes": 10
    }

    args = parse_args()

    # Get train, val and holdout dataloaders
    train_dataloader, test_dataloader, holdout_dataloader = get_dataloader(batch_size)
    
    # Train original net with sampling method == 1
    train(
        get_model(params={}, original=True),
        "models/original_mix_sm_1.ckpt",
        train_dataloader,
        test_dataloader,
        holdout_dataloader,
        epochs,
        lr,
        sm=1
    )

    # Train original net with sampling method == 2
    train(
        get_model(params={}, original=True),
        "models/original_mix_sm_2.ckpt",
        train_dataloader,
        test_dataloader,
        holdout_dataloader,
        epochs,
        lr,
        sm=2
    )

    # Train ViT with sampling method == 1
    train(
        get_model(params=vit_16_params),
        "models/vit_mix_sm_1.ckpt",
        train_dataloader,
        test_dataloader,
        holdout_dataloader,
        epochs,
        lr,
        sm=1
    )

    # Train ViT with sampling method == 2
    train(
        get_model(params=vit_16_params),
        "models/vit_mix_sm_2.ckpt",
        train_dataloader,
        test_dataloader,
        holdout_dataloader,
        epochs,
        lr,
        sm=2
    )
