import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time
import os
import torch.nn.functional as F


## MODEL CREATION
import torch
import torch.nn as nn

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = x.view(batch, channels, -1).mean(dim=2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEModule(out_channels, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, reduction=16):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], reduction=reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, reduction))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, reduction=reduction))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet34_se(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)










model = resnet34_se(num_classes=100)



model.apply(init_weights)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.0001,weight_decay=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)


# Learning rate scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training the model
# Training the model
def train_model(num_epochs):
    total_train_time = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()  # Start time for the epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
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

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # scheduler.step()  # Update the learning rate
        training_loss = running_loss/total
        training_accuracy = correct/total
        end_time = time.time()  # End time for the epoch
        epoch_duration = end_time - start_time
        total_train_time += epoch_duration
        print(f'Time elapsed for epoch {epoch + 1}: {epoch_duration:.2f} seconds')
        print(f'Epoch {epoch+1}/{num_epochs}: Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.4f}')
        # print(f'Time elapsed for epoch {epoch + 1}: {epoch_duration:.2f} seconds')


        
        val_loss, val_accuracy = validate_model()
        print(f'Epoch {epoch+1}/{num_epochs}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        torch.save(model.state_dict(), model_save_path)
        print(f'Epoch {epoch+1}: Model improved and saved to {model_save_path}')
        

    print(f'Total training time: {total_train_time:.2f} seconds')

# Validate the model
def validate_model():
    start_time = time.time()  # Start time for validation
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

    end_time = time.time()  # End time for validation
    validation_duration = end_time - start_time
    print(f'Validation time: {validation_duration:.2f} seconds')

    average_loss = total_loss / total
    accuracy = correct/total
    print(f'Average Validation Loss: {average_loss:.4f}')

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

    return average_loss,accuracy

    

# Directory for saving the model
model_save_path = '/csehome/m22cs053/RADIUS_ASSIGNMENT/saved_models/imagenet_100_resnet34_se_new.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Train and validate
num_epochs = 30  # Set the number of epochs
train_model(num_epochs)