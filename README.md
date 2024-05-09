# A comparative study for ImageNet-100 Classification task with ResNet-50, ResNet-34, ResNet-18, Efficient-NetB0

This Repo contains a comparative analysis of 4 different classifiers trained and evaluated on the bench marks dataset, Imagenet-100 containing 100 classes.

#  Comparison of the Accuracies, Training Times, Inferencing Times

**Baseline Scenario**:  
Each model is trained for 30 epochs, Adam optimizer is used, Loss function used is Cross Entropy Loss, Default Learning Rate used is 0.0001, batch size of 256 is selected,  
Accuracy is calculated on the validation dataset.

**ResNet-34**:  
  - Accuracy: 63%  
  - Training Time: 31906.43 seconds  
  - Inference Time: 0.188 seconds
  - The entire classification report containing the Precision, Recall, and F1 score are included in the log files of each model  

**ResNet-50**:  
  - Accuracy: 58%  
  - Training Time: 27032.31 seconds  
  - Inference Time: 0.637 seconds
  - The entire classification report containing the Precision, Recall, and F1 score are included in the log files of each model  

**EfficientNet-B0**:  
  - Accuracy: 48%  
  - Training Time: 25004.28 seconds  
  - Inference Time: 0.368 seconds
  - The entire classification report containing the Precision, Recall, and F1 score are included in the log files of each model

**ResNet-18**:  
  - Accuracy: 58%  
  - Training Time: 40397.59.28 seconds  
  - Inference Time: 0.237 seconds
  - The entire classification report containing the Precision, Recall, and F1 score are included in the log files of each model



### Training

python train.py

### Resource Used

All the models have been trained on DGX A-100 gpu
