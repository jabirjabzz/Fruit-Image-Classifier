from torchvision import transforms, datasets
import os
from . import config

def create_data_generators():
    """Create train, validation, and test data loaders."""
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.RandomRotation(config.ROTATION_RANGE),
        transforms.RandomHorizontalFlip() if config.HORIZONTAL_FLIP else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * config.CHANNELS, [0.5] * config.CHANNELS)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * config.CHANNELS, [0.5] * config.CHANNELS)
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform)
    validation_dataset = datasets.ImageFolder(config.VALIDATION_DIR, transform=test_transform)
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=test_transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    return train_loader, validation_loader, test_loader

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for prediction."""
    from PIL import Image
    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * config.CHANNELS, [0.5] * config.CHANNELS)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension
