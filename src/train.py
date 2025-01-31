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
        transforms.RandomRotation(config.ROTATION_RANGE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * config.CHANNELS, [0.5] * config.CHANNELS)
    ])

    # Data loaders
    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_dataset = datasets.ImageFolder(config.VALIDATION_DIR, transform=transform)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE)

    # Model, loss, optimizer
    model = create_model()
    model = model.to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # TensorBoard writer
    log_dir = os.path.join(config.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    try:
        # Log model graph
        dummy_input = torch.randn(1, config.CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH).to('cuda')
        writer.add_graph(model, dummy_input)

        # Training loop
        best_accuracy = 0.0
        for epoch in range(config.EPOCHS):
            # Training phase
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                # Log batch loss every 100 batches
                if batch_idx % 100 == 99:
                    writer.add_scalar('Loss/train_batch', 
                                    loss.item(),
                                    epoch * len(train_loader) + batch_idx)

            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            # Calculate validation metrics
            val_loss = val_loss / len(validation_loader)
            val_accuracy = 100 * correct_val / total_val

            # Log all metrics
            writer.add_scalars('Loss', {
                'train': epoch_loss,
                'validation': val_loss
            }, epoch)
            
            writer.add_scalars('Accuracy', {
                'train': train_accuracy,
                'validation': val_accuracy
            }, epoch)
            
            # Log learning rate
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

            print(f"Epoch {epoch+1}/{config.EPOCHS}")
            print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

            # Save the best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                }, best_checkpoint_path)
                print(f"Best model saved with accuracy {best_accuracy:.2f}%")

        # Save the final model
        final_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
        torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_accuracy': val_accuracy,
        }, final_checkpoint_path)
        print(f"Final model saved to {final_checkpoint_path}")

    finally:
        # Ensure writer is closed properly
        writer.flush()
        writer.close()

    return model, log_dir

if __name__ == "__main__":
    model, log_dir = train_model()
    print(f"\nTraining completed! To view training metrics, run:")
    print(f"tensorboard --logdir={log_dir}")