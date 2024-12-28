import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import make_grid
from . import config

def plot_training_history(training_losses, validation_accuracies):
    """Plot training history."""
    epochs = len(training_losses)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(range(1, epochs + 1), training_losses, label='Training Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(range(1, epochs + 1), validation_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def display_sample_images(dataset, num_images=5):
    """Display sample images from each class."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_images, shuffle=True)
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # Plot a batch of images
    plt.figure(figsize=(15, 5))
    grid = make_grid(images, nrow=num_images)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def create_confusion_matrix(model, dataloader, class_names):
    """Create and plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
