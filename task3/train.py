import time

from typing import Any

import torch
import torchvision

import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, save_image

from mixup import BatchMixUp


class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_dataloader(batch_size: int, holdout_size: float = 0.2, train_size: float = 0.9) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    dataset_size = len(dataset)
    holdout_test_size = int(dataset_size * holdout_size)
    development_size = dataset_size - holdout_test_size

    development_set, holdout_test_set = random_split(dataset, [development_size, holdout_test_size])

    train_size = int(development_size * train_size)
    validation_size = development_size - train_size
    train_set, validation_set = random_split(development_set, [train_size, validation_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    holdout_test_loader = DataLoader(holdout_test_set, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, holdout_test_loader


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


def model_eval(
        model: nn.Module,
        data: DataLoader,
        device: torch.device,
        criterion: torch.nn.Module,
        num_classes: int = 10
    ) -> tuple[float, float]:
    with torch.no_grad():
        model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        true_positive = torch.zeros(10, dtype=torch.int64)
        false_positive = torch.zeros(10, dtype=torch.int64)
        false_negative = torch.zeros(10, dtype=torch.int64)

        for batch in data:
            images, labels = batch
            images = images.to(device)
            labels = labels.type(torch.long).to(device)
            
            outputs = model(images)

            hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(torch.float32)

            outputs = model(images)
            loss = criterion(outputs, hot_labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(num_classes):
                true_positive[i] += ((predicted == i) & (labels == i)).sum().item()
                false_positive[i] += ((predicted == i) & (labels != i)).sum().item()
                false_negative[i] += ((predicted != i) & (labels == i)).sum().item()

    accuracy = correct / total

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1 = 2 * (precision * recall) / (precision + recall)

    F1[torch.isnan(F1)] = 0

    macro_F1 = torch.mean(F1)

    accuracy = correct / total
    average_loss = total_loss / len(data)
    average_precision = torch.mean(precision)
    average_recall = torch.mean(recall)

    return {
        "accuracy": accuracy,
        "macro_precision": average_precision.item(),
        "macro_recall": average_recall.item(),
        "macro_F1": macro_F1.item(),
        "loss": total_loss,
        "average_loss": average_loss
    }


def train(
        model: nn.Module,
        model_path: str,
        results_path: str,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        holdout_dataloader: DataLoader,
        epochs: int = 2,
        lr: float = 0.001,
        sm: int = 1,
        alpha: float = 0.4,
        momentum: float = 0.9
):
    print(f"---------- TRAINING {model_path} (sampling method = {sm}, alpha = {alpha}) ----------")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mix = BatchMixUp(sampling_method=sm, alpha=alpha)

    dataiter = iter(train_dataloader)
    batch = next(dataiter)

    mix.visualize(mix.mix_up(batch))

    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    train_start = time.time()

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

        eval_result = model_eval(model, test_dataloader, device, criterion)
        print(f"Epoch: {epoch} - {eval_result}")

    train_end = time.time()
    train_duration = train_end - train_start
    print(f"Training done in {train_duration} seconds")

    train_eval_result = model_eval(model, train_dataloader, device, criterion)
    test_eval_result = model_eval(model, test_dataloader, device, criterion)
    holdout_eval_result = model_eval(model, holdout_dataloader, device, criterion)

    print(f"Train set performance - {train_eval_result}")
    print(f"Test set performance - {test_eval_result}")
    print(f"Holdout set performance - {holdout_eval_result}")

    save_test_images(test_dataloader, model, device, class_names, results_path)

    torch.save(model.state_dict(), model_path)

    print('Model saved.')
