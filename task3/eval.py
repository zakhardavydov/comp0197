import torch

from model import get_model

from train import model_eval, get_dataloader


if __name__ == "__main__":
    batch_size = 64

    vit_16_params = {
        "image_size": 32,
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_classes": 10
    }
    
    model_path = "models/sm_1.ckpt"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    criterion = torch.nn.BCEWithLogitsLoss()

    train_dataloader, test_dataloader, holdout_dataloader = get_dataloader(batch_size)

    state_dict = torch.load(model_path)

    model = get_model(vit_16_params, original=False)
    model.load_state_dict(state_dict)

    model.to(device)

    results = model_eval(model, holdout_dataloader, device, criterion, vit_16_params.get("num_classes"))

    print(f"Results - {results}")
