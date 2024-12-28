import os

"""
Configuration settings for the Fruit Classification project (PyTorch version).
"""

# Data parameters
IMG_HEIGHT = 100  # Adjust if necessary
IMG_WIDTH = 100
CHANNELS = 3
BATCH_SIZE = 32
NUM_CLASSES = 141  # Adjust to match your dataset

# Model parameters
LEARNING_RATE = 0.001
EPOCHS = 10
DROPOUT_RATE = 0.5
# Paths
DATA_DIR = 'data'
