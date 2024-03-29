from utils import parse_args
from train import train_vit, get_dataloader


if __name__ == "__main__":

    batch_size = 64
    epochs = 20
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
    
    sm_1_path = "models/sm_1.ckpt"
    sm_2_path = "models/sm_2.ckpt"

    results_sm1_path = "results_sm_1.png"
    results_sm2_path = "results_sm_2.png"
    
    args = parse_args()

    train_dataloader, test_dataloader = get_dataloader(batch_size)
    
    train_vit(sm_1_path, results_sm1_path, vit_16_params, train_dataloader, test_dataloader, batch_size, epochs, lr, sm=1)
    train_vit(sm_2_path, results_sm2_path, vit_16_params, train_dataloader, test_dataloader, batch_size, epochs, lr, sm=2)
