"""
Configuration settings for the Fruit Classification project.
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
TRAIN_DIR = f'{DATA_DIR}/train'
TEST_DIR = f'{DATA_DIR}/test'
VALIDATION_DIR = f'{DATA_DIR}/validation'
MODEL_SAVE_PATH = 'models/fruit_classifier.h5'
CHECKPOINT_DIR = 'models/checkpoints'
LOG_DIR = 'logs'

# Class names
CLASS_NAMES = ['apples', 'bananas', 'oranges']

# Data augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
FILL_MODE = 'nearest'