import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_data_loaders(batch_size, num_workers=4, root='/csehome/m22cs053/RADIUS_ASSIGNMENT/imagenet/imagenet100'):
    # Data preprocessing
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load training dataset from multiple parts

    train_dataset = datasets.ImageFolder(root='/csehome/m22cs053/imagenet/imagenet100/train.X1', transform=transform) 

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Load validation dataset
    val_dataset = datasets.ImageFolder(root='/csehome/m22cs053/imagenet/imagenet100/train.X1', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    return train_loader, val_loader
