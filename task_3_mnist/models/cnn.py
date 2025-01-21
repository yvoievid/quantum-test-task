import torch
from torchvision import models
import numpy as np
from torch import nn
import torch.nn.functional as F
from interfaces.idigit_classifier import DigitClassificationInterface


class CNN(DigitClassificationInterface):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def predict(self, image: torch.Tensor):
        assert image.shape == (28, 28, 1), f"Expected shape (28, 28, 1), got {image.shape}"
        image = image.clone().detach().permute(2, 0, 1).unsqueeze(0).float()    # Shape: (1, 1, 28, 28)
        with torch.no_grad():
            predictions = self.model(image)[0]
        probabilities = F.softmax(predictions, dim=0)
        return np.argmax(probabilities.numpy())

    def fit(self, features, labels):
        raise NotImplementedError("The `fit` method must be implemented by the subclass.")