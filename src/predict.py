# src/predict.py

import torch
from torchvision import transforms
from PIL import Image
import os
from src.model import create_model  # Changed to import create_model instead
from src import config

def load_trained_model(model_path):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path (str): Path to the checkpoint file
        
    Returns:
        model: Loaded PyTorch model
    """
    # Create a new model instance
    model = create_model()
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, weights_only=True)
    
    # Load just the model state dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model

def predict_fruit(image_path, model_path):
    """
    Predict fruit class from image using trained model.
    
    Args:
        image_path (str): Path to input image
        model_path (str): Path to trained model weights
    
    Returns:
        str: Predicted class name
    """
    # Verify files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * config.CHANNELS, [0.5] * config.CHANNELS)
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Load model and set to evaluation mode
    model = load_trained_model(model_path)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = config.CLASS_NAMES[predicted.item()]
    
    return predicted_class

if __name__ == "__main__":
    # Use your specific image path
    image_path = r"C:\Users\Administrator\Documents\GitHub\Fruit-Image-Classifier\datasets\moltean\fruits\versions\11\fruits-360_dataset_100x100\fruits-360\Test\Apple Granny Smith 1\6_100.jpg"
    
    # Model path
    model_path = os.path.join("checkpoints", "best_model.pth")  # or "final_model.pth"
    
    try:
        predicted_fruit = predict_fruit(image_path, model_path)
        print(f"Predicted fruit: {predicted_fruit}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")