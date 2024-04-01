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
    # Normalize images as per tutorial
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    # Get relevant parts of the dataset
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    
    # Get dataloaders, shuffle every batch
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader


def save_images(dataloader, model, device, class_names, path="result.png", max_images=36):
    model.eval()

    # We only want a grid that is 6x6, so we are going to iterate over images until we have it
    images_collected = 0

    # Accumulate images in these tensors
    collected_images = torch.empty((max_images, 3, 32, 32))
    collected_labels = torch.empty(max_images, dtype=torch.long)
    collected_preds = torch.empty(max_images, dtype=torch.long)

    with torch.no_grad():
        # We iterate through given dataloader
        for images, labels in dataloader:
            # Until we have enough images
            if images_collected >= max_images:
                break

            # Move data to where the model is
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions
            outputs = model(images)
            # Get predicted labels
            _, predicted = torch.max(outputs, 1)

            # If the batch is over the limit of max_images, trim to only those we want to collect
            images_to_collect = min(max_images - images_collected, images.size(0))

            # Put images in the bucket. For viz, move to CPU
            collected_images[images_collected:images_collected+images_to_collect] = images[:images_to_collect].cpu()
            collected_labels[images_collected:images_collected+images_to_collect] = labels[:images_to_collect].cpu()
            collected_preds[images_collected:images_collected+images_to_collect] = predicted[:images_to_collect].cpu()

            # Increase the counter so we don't go over the limit
            images_collected += images_to_collect

    # Make a grid, knowing that we have exactly max_images in collected_images
    grid = make_grid(collected_images, nrow=6)  # Creates a 6x6 grid

    save_image(grid, path)
    print(f"Saved grid to {path}")

    # Print labels for images that we just saved
    for i in range(max_images):
        true_label = class_names[collected_labels[i].item()]
        pred_label = class_names[collected_preds[i].item()]
        print(f"Image {i+1}: True Label = {true_label}, Predicted Label = {pred_label}")


def train_vit(
        model_path: str,
        results_path: str,
        vit_params: dict[str, Any],
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        epochs: int = 2,
        lr: float = 0.001,
        sm: int = 1,
        alpha: float = 0.3,
        log_step: int = 500
):
    # If we have GPU, use it
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Init the mixup
    mix = BatchMixUp(sampling_method=sm, alpha=alpha)

    # Get first batch for mixup viz
    dataiter = iter(train_dataloader)
    batch = next(dataiter)

    mix.visualize(batch)

    # Get ViT and move to device
    model = get_vit(vit_params)
    model.to(device)

    # We are using the loss optimal for multi-class classification
    criterion = torch.nn.BCEWithLogitsLoss()
    # We use same optimizer as per orginal tutorial
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_dataloader, 0):
            images = images.to(device)

            labels = labels.type(torch.long)

            # We want hot encoding, not CiFAR labels
            hot_labels = torch.nn.functional.one_hot(labels, num_classes=10).to(torch.float32)
            hot_labels = hot_labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, hot_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Print a bit more often than original tutorial
            if i % log_step == (log_step - 1):
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / log_step))
                running_loss = 0.0

        # Run eval for this batch
        with torch.no_grad():
            model.eval()

            # As per task2 requirements, collect only accuracy
            correct = 0
            total = 0
            for i, batch in enumerate(test_dataloader):

                # Do NOT apply mixup for eval
                images, labels = batch
                images = images.to(device)

                labels = labels.type(torch.long)
                labels = labels.to(device)

                hot_labels = torch.nn.functional.one_hot(labels, num_classes=10).to(torch.float32)

                outputs = model(images)
                loss = criterion(outputs, hot_labels)

                _, predicted = torch.max(outputs.data, 1)

                total += hot_labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print(f"Epoch: {epoch} - Accuracy: {correct / total}")

    print('Training done.')

    save_images(test_dataloader, model, device, class_names, results_path)

    torch.save(model.state_dict(), model_path)

    print('Model saved.')
