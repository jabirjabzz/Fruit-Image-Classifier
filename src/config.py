import os

"""
Configuration settings for the Fruit Classification project (PyTorch version).
"""

# Data parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
NUM_CLASSES = 3

# Model parameters
LEARNING_RATE = 0.001
EPOCHS = 10
DROPOUT_RATE = 0.5

# Paths
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_SAVE_PATH = 'models/fruit_classifier.pth'
CHECKPOINT_DIR = 'models/checkpoints'
LOG_DIR = 'logs'

# Class names
CLASS_NAMES = ['apples', 'bananas', 'oranges']

# Data augmentation parameters
ROTATION_RANGE = 20
HORIZONTAL_FLIP = True
