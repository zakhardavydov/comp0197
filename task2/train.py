from typing import Any

import torch
import torchvision

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from vit import get_vit
from mixup import BatchMixUp


class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_dataloader(batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader


def save_test_images(dataloader, model, device, class_names, file_name="result.png"):
    model.eval() 

    collected_images = []
    collected_labels = []
    collected_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            if len(collected_images) >= 36:
                break
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            collected_images.append(images.cpu())
            collected_labels.extend(labels.cpu())
            collected_preds.extend(predicted.cpu())

            collected_images = collected_images[:36]
            collected_labels = collected_labels[:36]
            collected_preds = collected_preds[:36]

    images_tensor = torch.cat(collected_images, dim=0)
    
    img_grid = make_grid(images_tensor, nrow=6)

    save_image(img_grid, file_name)

    for i in range(len(collected_images)):
        print(f"Image {i+1}, label: {class_names[collected_labels[i]]}, predicted: {class_names[collected_preds[i]]}")


def train_vit(
        model_path: str,
        vit_params: dict[str, Any],
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        batch_size: int = 20,
        epochs: int = 2,
        lr: float = 0.001,
        sm: int = 1,
        alpha: float = 0.2
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    mix = BatchMixUp(sampling_method=sm, alpha=alpha)

    dataiter = iter(train_dataloader)
    batch = next(dataiter)

    mix.visualize(mix.mix_up(batch))

    model = get_vit(vit_params)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_dataloader, 0):
            images = images.to(device)

            labels = labels.type(torch.long)
            hot_labels = torch.nn.functional.one_hot(labels, num_classes=10).to(torch.float32)
            hot_labels = hot_labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, hot_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        with torch.no_grad():
            model.eval()

            correct = 0
            total = 0
            for i, (fv, labels) in enumerate(test_dataloader):
                fv = fv.to(device)
                labels = labels.to(device)
                outputs = model(fv)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print(f"Epoch: {epoch} - Accuracy: {100 * correct / total} %")

    print('Training done.')

    save_test_images(test_dataloader, model, device, class_names)

    torch.save(model.state_dict(), model_path)

    print('Model saved.')
