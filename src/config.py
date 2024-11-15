"""
Configuration settings for the Fruit Classification project.
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
TRAIN_DIR = '/workspaces/Fruit-Image-Classifier/datasets/moltean/fruits/versions/11/fruits-360_dataset_100x100/fruits-360/Training/train'
TEST_DIR = f'/workspaces/Fruit-Image-Classifier/datasets/moltean/fruits/versions/11/fruits-360_dataset_100x100/fruits-360/Test'
VALIDATION_DIR = "/workspaces/Fruit-Image-Classifier/datasets/moltean/fruits/versions/11/fruits-360_dataset_100x100/fruits-360/Training/validation"
MODEL_SAVE_PATH = 'models/fruit_classifier.h5'
CHECKPOINT_DIR = 'models/checkpoints'
LOG_DIR = 'logs'

# Class names
import os

def get_class_names(dataset_dir):
    class_names = os.listdir(dataset_dir)
    return class_names

# Assuming your dataset directory is in the 'data' folder
dataset_dir = '/workspaces/Fruit-Image-Classifier/datasets/moltean/fruits/versions/11/fruits-360_dataset_100x100/fruits-360/Training'
class_names = get_class_names(dataset_dir)

# Update the CLASS_NAMES list in your config.py file
CLASS_NAMES = class_names

# Data augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
FILL_MODE = 'nearest'