import torch
import torch.nn as nn
import torch.optim as optim
from . import config

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(config.CHANNELS, 32, kernel_size=3)  # Input channels, output channels, kernel size
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Calculate the input size for fc1 dynamically
        self.fc1_input_size = self._calculate_fc1_input_size()

        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.fc2 = nn.Linear(512, config.NUM_CLASSES)
    
    def _calculate_fc1_input_size(self):
        # Create a dummy input to calculate the size after convolutions and pooling
        dummy_input = torch.zeros(1, config.CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
        x = self.pool1(nn.ReLU()(self.conv1(dummy_input)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = self.pool3(nn.ReLU()(self.conv3(x)))
        return x.numel()  # Total number of elements in the tensor
    
    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = self.pool3(nn.ReLU()(self.conv3(x)))
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.Softmax(dim=1)(self.fc2(x))
        return x

def create_model():
    return CNNModel()

def load_trained_model(model_path):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model