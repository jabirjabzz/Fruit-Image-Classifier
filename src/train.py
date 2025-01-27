import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from datetime import datetime
from . import config
from .model import create_model

def train_model():
    # Directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.RandomHorizontalFlip() if config.HORIZONTAL_FLIP else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(config.ROTATION_RANGE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Data loaders
    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_dataset = datasets.ImageFolder(config.VALIDATION_DIR, transform=transform)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE)

    # Model, loss, optimizer
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Training loop
    best_accuracy = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch+1}_accuracy_{accuracy:.2f}.pth"))

        # Logging
        writer.add_scalar("Loss/Train", running_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

    writer.close()
    return model

if __name__ == "__main__":
    train_model()