import torch
import torch.nn as nn
import torch.optim as optim
from . import config

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(config.CHANNELS, 32, kernel_size=3, activation=nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, activation=nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, activation=nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (config.IMG_HEIGHT // 8) * (config.IMG_WIDTH // 8), 512)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.fc2 = nn.Linear(512, config.NUM_CLASSES)
    
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
