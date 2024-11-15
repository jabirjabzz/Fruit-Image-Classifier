import os
import shutil
from sklearn.model_selection import train_test_split

dataset_dir = '/workspaces/Fruit-Image-Classifier/datasets/moltean/fruits/versions/11/fruits-360_dataset_100x100/fruits-360/Training'

def prep_data(dataset_dir):

    """Prepares the dataset by splitting it into training and validation sets.

    Args:
        dataset_dir: Path to the root directory of the dataset.

    Returns:
        train_dir: Path to the training directory.
        validation_dir: Path to the validation directory.
    """

    # Get all subdirectories (classes) in the training directory
    subdirs = [subdir for subdir in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, subdir))]

    # Check if there are enough subdirectories for splitting
    if len(subdirs) < 2:
        raise ValueError("Insufficient number of classes in the dataset.")

    # Split the subdirectories into training and validation sets
    train_subdirs, validation_subdirs = train_test_split(
        subdirs, test_size=0.2, random_state=42
    )

    # Create the actual training and validation directories
    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'validation')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Move the corresponding subdirectories to the train and validation directories
    for subdir in train_subdirs:
        shutil.move(os.path.join(dataset_dir, subdir), os.path.join(train_dir, subdir))

    for subdir in validation_subdirs:
        shutil.move(os.path.join(dataset_dir, subdir), os.path.join(validation_dir, subdir))

    return train_dir, validation_dir



    




