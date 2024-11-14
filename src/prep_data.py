import os
import shutil
from sklearn.model_selection import train_test_split

# Assuming your dataset is in the 'fruit_vegetable_dataset' directory
dataset_dir = '/workspaces/Fruit-Image-Classifier/datasets/moltean/fruits/versions/11/fruits-360_dataset_100x100'

# Get all subdirectories (classes) in the training directory
train_subdirs = [subdir for subdir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, subdir))]

# Split the subdirectories into training and validation sets
train_subdirs, validation_subdirs = train_test_split(
    train_subdirs,
    test_size=0.2,  # Adjust the test_size as needed
    random_state=42
)

# Create the actual training and validation directories
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')

# Create the directories if they don't exist
os.makedirs('/workspaces/Fruit-Image-Classifier/datasets/moltean/fruits/training_dataset/train_dir', exist_ok=True)
os.makedirs('/workspaces/Fruit-Image-Classifier/datasets/moltean/fruits/training_dataset/validation_dir', exist_ok=True)

# Move the corresponding subdirectories to the train and validation directories
for subdir in train_subdirs:
    shutil.move(os.path.join(dataset_dir, subdir), os.path.join(train_dir, subdir))

for subdir in validation_subdirs:
    shutil.move(os.path.join(dataset_dir, subdir), os.path.join(validation_dir, subdir))