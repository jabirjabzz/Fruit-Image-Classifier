import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from . import config

def plot_training_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def display_sample_images(num_images=5):
    """Display sample images from each class."""
    fig, axes = plt.subplots(len(config.CLASS_NAMES), num_images, figsize=(15, 10))
    
    for i, class_name in enumerate(config.CLASS_NAMES):
        class_path = os.path.join(config.TRAIN_DIR, class_name)
        images = os.listdir(class_path)[:num_images]
        
        for j, image_name in enumerate(images):
            image_path = os.path.join(class_path, image_name)
            img = load_img(image_path, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(class_name)
    
    plt.tight_layout()
    plt.show()

def create_confusion_matrix(model, test_generator):
    """Create and plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Get predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()