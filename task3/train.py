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


def get_dataloader(batch_size: int, holdout_size: float = 0.2, train_size: float = 0.9) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Gets all dataloaders as per the requirements
    """
    # As per original tutorial
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    # Only use training part
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    # Calculate the holdout
    dataset_size = len(dataset)
    holdout_test_size = int(dataset_size * holdout_size)
    development_size = dataset_size - holdout_test_size

    development_set, holdout_test_set = random_split(dataset, [development_size, holdout_test_size])

    # Calculate the validation
    train_size = int(development_size * train_size)
    validation_size = development_size - train_size
    train_set, validation_set = random_split(development_set, [train_size, validation_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    holdout_test_loader = DataLoader(holdout_test_set, batch_size=batch_size, shuffle=False)

    # Return all three
    return train_loader, validation_loader, holdout_test_loader


def model_eval(
        model: nn.Module,
        data: DataLoader,
        device: torch.device,
        criterion: torch.nn.Module,
        num_classes: int = 10
    ) -> tuple[float, float]:
    with torch.no_grad():
        model.eval()

        # Correct correct predictions as in task2
        correct = 0
        total = 0
        total_loss = 0.0

        # Collect class-dependent true positives, false positives and false negatives
        true_positive = torch.zeros(num_classes, dtype=torch.int64)
        false_positive = torch.zeros(num_classes, dtype=torch.int64)
        false_negative = torch.zeros(num_classes, dtype=torch.int64)

        # For every batch in dataloader we get here, run the eval
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
            
            # Collect the performance of this class
            for i in range(num_classes):
                true_positive[i] += ((predicted == i) & (labels == i)).sum().item()
                false_positive[i] += ((predicted == i) & (labels != i)).sum().item()
                false_negative[i] += ((predicted != i) & (labels == i)).sum().item()

    accuracy = correct / total

    # Class-based precision, recall and F1
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1 = 2 * (precision * recall) / (precision + recall)

    F1[torch.isnan(F1)] = 0

    # We track macro F1 which is a simple average of F1 across all casses
    macro_F1 = torch.mean(F1)

    accuracy = correct / total
    average_loss = total_loss / len(data)
    average_precision = torch.mean(precision)
    average_recall = torch.mean(recall)

    # Pack metrics that we are going to report on and return
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
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        holdout_dataloader: DataLoader,
        epochs: int = 2,
        lr: float = 0.001,
        sm: int = 1,
        alpha: float = 0.3,
        momentum: float = 0.9,
        num_classes: int = 10,
        log_step: int = 500
):
    print(f"---------- TRAINING {model_path} (sampling method = {sm}, alpha = {alpha}) ----------")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    # If we have a GPU, use it
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Init MixUp class
    mix = BatchMixUp(sampling_method=sm, alpha=alpha)

    # Move model to device
    model.to(device)

    # Use basic loss for multi-class classification
    criterion = torch.nn.BCEWithLogitsLoss()
    # Optimizer as per original tutorial
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    # Benchmark the training start time
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        
        for i, batch in enumerate(train_dataloader, 0):
            # Apply mixup
            images, labels = mix.mix_up(batch)
            images = images.to(device)

            # Reformat labels into hot encoding
            labels = labels.type(torch.long)
            hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(torch.float32)
            hot_labels = hot_labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, hot_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % log_step == (log_step - 1):
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / log_step))
                running_loss = 0.0

        # Instead of just accuracy, run entire suit of metrics on test dataloader
        eval_result = model_eval(model, test_dataloader, device, criterion, num_classes)
        print(f"Epoch: {epoch} - {eval_result}")

    train_end = time.time()
    train_duration = train_end - train_start
    print(f"Training done in {train_duration} seconds")

    # As per requirements, benchmark across train, validation and holdout
    train_eval_result = model_eval(model, train_dataloader, device, criterion, num_classes)
    test_eval_result = model_eval(model, test_dataloader, device, criterion, num_classes)
    holdout_eval_result = model_eval(model, holdout_dataloader, device, criterion, num_classes)

    print(f"Train set performance - {train_eval_result}")
    print(f"Test set performance - {test_eval_result}")
    print(f"Holdout set performance - {holdout_eval_result}")

    # Save the weights and chillg
    torch.save(model.state_dict(), model_path)

    print('Model saved.')
