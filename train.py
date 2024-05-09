
#### `train.py`
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from time import time
import os
import argparse
import numpy as np


from dataloader import get_data_loaders
from model import initialize_model
from utils import save_model, print_classification_report


# Argument parser for training
parser = argparse.ArgumentParser(description="Training Script")
parser.add_argument("--model_name", type=str, default="resnet50", help="Model name (resnet18, resnet50, etc.)")
parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate

# Model and data loaders
model = initialize_model(args.model_name, num_classes=100, pretrained=False)
model = model.to(device)
train_loader, val_loader = get_data_loaders(batch_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Directory for saving the model
model_save_path = 'saved_models/imagenet_100_resnet50.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Training function
def train_model(num_epochs):
    total_train_time = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time()  # Start time for the epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        training_loss = running_loss / total
        training_accuracy = correct / total
        end_time = time()  # End time for the epoch
        epoch_duration = end_time - start_time
        total_train_time += epoch_duration
        print(f'Epoch [{epoch + 1}/{num_epochs}]: Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.4f}, Time: {epoch_duration:.2f} seconds')

        # Validate the model and compute validation loss and accuracy
        val_loss, val_accuracy = validate_model()
        print(f'Epoch [{epoch + 1}/{num_epochs}]: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, model_save_path)
            print(f'Epoch {epoch + 1}: Model improved and saved to {model_save_path}')

    print(f'Total training time: {total_train_time:.2f} seconds')

# Validation function
def validate_model():
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / total
    accuracy = correct / total
    print_classification_report(all_labels, all_preds)

    return average_loss, accuracy

# Train the model
train_model(num_epochs)
