import os
import shutil
import logging
from sklearn.model_selection import train_test_split
from pathlib import Path

def prep_data(dataset_dir):
    """Prepares the dataset by splitting it into training and validation sets."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Convert string path to Path object
    dataset_dir = Path(dataset_dir)
    logging.info(f"Processing dataset directory: {dataset_dir}")
    
    # Verify directory exists
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    # Create train and validation directories
    train_dir = dataset_dir / 'train'
    validation_dir = dataset_dir / 'validation'
    train_dir.mkdir(exist_ok=True, parents=True)
    validation_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all subdirectories (classes) in the dataset directory
    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name not in ['train', 'validation']]
    
    # Log found directories
    logging.info(f"Found {len(subdirs)} subdirectories:")
    for d in subdirs:
        logging.info(f"- {d.name}")
    
    # Check if there are enough subdirectories for splitting
    if len(subdirs) < 2:
        logging.error(f"Found only {len(subdirs)} classes in {dataset_dir}")
        logging.error("Directory contents:")
        for item in dataset_dir.iterdir():
            logging.error(f"- {item.name} ({'directory' if item.is_dir() else 'file'})")
        raise ValueError("Insufficient number of classes in the dataset. Need at least 2 class directories.")
    
    # Process each class directory
    for class_dir in subdirs:
        logging.info(f"Processing class: {class_dir.name}")
        
        # Get all image files
        images = [
            img for img in class_dir.iterdir() 
            if img.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ]
        
        if not images:
            logging.warning(f"No images found in {class_dir}")
            continue
            
        logging.info(f"Found {len(images)} images in {class_dir.name}")
        
        # Split images into training and validation sets
        train_images, validation_images = train_test_split(
            images, test_size=0.2, random_state=42
        )
        
        # Create class directories
        (train_dir / class_dir.name).mkdir(exist_ok=True)
        (validation_dir / class_dir.name).mkdir(exist_ok=True)
        
        # Copy training images
        for img in train_images:
            try:
                shutil.copy2(str(img), str(train_dir / class_dir.name / img.name))
            except Exception as e:
                logging.error(f"Error copying {img}: {str(e)}")
        
        # Copy validation images
        for img in validation_images:
            try:
                shutil.copy2(str(img), str(validation_dir / class_dir.name / img.name))
            except Exception as e:
                logging.error(f"Error copying {img}: {str(e)}")
    
    return train_dir, validation_dir

if __name__ == "__main__":
    # Use the correct Windows path
    dataset_dir = r"C:\Users\Administrator\Documents\GitHub\Fruit-Image-Classifier\datasets\moltean\fruits\versions\11\fruits-360_dataset_100x100\fruits-360\Training\train"
    
    try:
        train_dir, validation_dir = prep_data(dataset_dir)
        print(f"Training data saved to: {train_dir}")
        print(f"Validation data saved to: {validation_dir}")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        # Print additional troubleshooting information
        logging.error(f"\nTroubleshooting information:")
        logging.error(f"Current working directory: {Path.cwd()}")
        if Path(dataset_dir).exists():
            logging.error(f"Contents of {dataset_dir}:")
            for item in Path(dataset_dir).iterdir():

                logging.error(f"- {item.name}")

