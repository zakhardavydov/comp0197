import torch

from utils import parse_args
from train import train_vit, get_dataloader


if __name__ == "__main__":

    # Parse args
    args = parse_args()

    # For reproducibility
    torch.manual_seed(42)

    # Basic hyperparameters used for training
    # Batch size here is 16 as it produces the nice grid
    # In reality I trained with batch_size 64, both sampling method produce around 85% accuracy
    batch_size = 16

    # As per requirements, train for 20 epochs
    # In reality, I trained for 100 epochs
    epochs = 20
    lr = 0.001

    # ViT params taken out, so it is easier to experiment
    vit_16_params = {
        "image_size": 32,
        "patch_size": 16,
        "num_layers": 8,
        "num_heads": 8,
        "hidden_dim": 256,
        "mlp_dim": 512,
        "num_classes": 10
    }
    
    # Path to save the final weights
    weights_path = f"models/sm_{args.sampling_method}.ckpt"

    # Path to save inference on test images
    results_sm_path = f"results_sm_{args.sampling_method}.png"

    # Get dataloaders with entire dataset
    train_dataloader, test_dataloader = get_dataloader(batch_size)
    
    # Train!!!
    train_vit(
        weights_path,
        results_sm_path,
        vit_16_params,
        train_dataloader,
        test_dataloader,
        epochs,
        lr,
        sm=args.sampling_method
    )
