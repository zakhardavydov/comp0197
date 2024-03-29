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
    model.eval()  # Ensure the model is in evaluation mode

    images_collected = 0
    max_images = 36  # Total number of images we want to collect

    # Prepare a tensor to hold collected images, labels, and predictions
    collected_images = torch.empty((max_images, 3, 32, 32))  # Assuming CIFAR10 image dimensions (3, 32, 32)
    collected_labels = torch.empty(max_images, dtype=torch.long)
    collected_preds = torch.empty(max_images, dtype=torch.long)

    with torch.no_grad():  # No need to track gradients
        for images, labels in dataloader:
            if images_collected >= max_images:
                break  # Break if we've collected enough images

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Determine how many images to collect from this batch
            images_to_collect = min(max_images - images_collected, images.size(0))

            # Update the collections
            collected_images[images_collected:images_collected+images_to_collect] = images[:images_to_collect].cpu()
            collected_labels[images_collected:images_collected+images_to_collect] = labels[:images_to_collect].cpu()
            collected_preds[images_collected:images_collected+images_to_collect] = predicted[:images_to_collect].cpu()

            images_collected += images_to_collect

    # Now that we have exactly 36 images, labels, and predictions, generate the grid
    img_grid = make_grid(collected_images, nrow=6)  # Creates a 6x6 grid

    # Save the grid of images
    save_image(img_grid, file_name)
    print(f"Images saved to {file_name}")

    # For demonstration, print out the true and predicted labels for debugging
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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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
            
            if i % 500 == 499:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        with torch.no_grad():
            model.eval()

            correct = 0
            total = 0
            for i, batch in enumerate(test_dataloader):
                images, labels = mix.mix_up(batch)
                images = images.to(device)

                labels = labels.type(torch.long)
                labels = labels.to(device)

                hot_labels = torch.nn.functional.one_hot(labels, num_classes=10).to(torch.float32)

                outputs = model(images)
                loss = criterion(outputs, hot_labels)

                _, predicted = torch.max(outputs.data, 1)

                total += hot_labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print(f"Epoch: {epoch} - Accuracy: {100 * correct / total} %")

    print('Training done.')

    save_test_images(test_dataloader, model, device, class_names, results_path)

    torch.save(model.state_dict(), model_path)

    print('Model saved.')
