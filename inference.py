import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from time import time

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# Load the saved model
model_path = '/csehome/m22cs053/RADIUS_ASSIGNMENT/saved_models/imagenet_100_resnet50.pth'

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 100)  # Adjusting for 100 classes
# model.load_state_dict(torch.load("imagenet_100_resnet50.pth"))
model.load_state_dict(torch.load(model_path, map_location=device)) 
model.to(device)
model.eval()

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a function for image inference
def infer_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()  # Return the predicted class index

# Example usage:
image_path = "/csehome/m22cs053/RADIUS_ASSIGNMENT/imagenet/imagenet100/val.X/n01491361/ILSVRC2012_val_00002922.JPEG"
start = time()
predicted_class_index = infer_image(image_path)
end = time()
print("Predicted class index:", predicted_class_index)
print("Inference Time", end-start)