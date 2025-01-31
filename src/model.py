import torch
import torch.nn as nn
import torch.optim as optim
from . import config

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(config.CHANNELS, 64, kernel_size=3, padding=1)  # Increased filters
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Increased filters
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Increased filters
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Added new layer
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._calculate_fc1_input_size(), 1024)  # Increased neurons
        self.dropout1 = nn.Dropout(config.DROPOUT_RATE)
        self.fc2 = nn.Linear(1024, 512)  # Added new layer
        self.dropout2 = nn.Dropout(config.DROPOUT_RATE)
        self.fc3 = nn.Linear(512, config.NUM_CLASSES)  # Output layer
        
    def _calculate_fc1_input_size(self):
        # Create a dummy input to calculate the size after convolutions and pooling
        dummy_input = torch.zeros(1, config.CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
        x = self.pool1(nn.ReLU()(self.conv1(dummy_input)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = self.pool3(nn.ReLU()(self.conv3(x)))
        x = self.pool4(nn.ReLU()(self.conv4(x)))  # Added for the new layer
        return x.numel()  # Total number of elements in the tensor
    
    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = self.pool3(nn.ReLU()(self.conv3(x)))
        x = self.pool4(nn.ReLU()(self.conv4(x)))  # Added for the new layer
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # Output layer
        return x

def create_model():
    return CNNModel()

def load_trained_model(model_path):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
