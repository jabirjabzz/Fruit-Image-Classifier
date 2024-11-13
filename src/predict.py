import numpy as np
from tensorflow.keras.models import load_model
from .data_loader import load_and_preprocess_image
from . import config

def predict_image(image_path):
    """
    Predict the class of a single image.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        tuple: (predicted_class, confidence)
    """
    # Load the trained model
    model = load_model(config.MODEL_SAVE_PATH)
    
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img)
    
    # Get the predicted class and confidence
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    return config.CLASS_NAMES[predicted_class_index], confidence

def batch_predict(image_paths):
    """
    Predict classes for multiple images.
    
    Args:
        image_paths (list): List of paths to image files
    
    Returns:
        list: List of (predicted_class, confidence) tuples
    """
    # Load the trained model
    model = load_model(config.MODEL_SAVE_PATH)
    
    results = []
    for image_path in image_paths:
        img = load_and_preprocess_image(image_path)
        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]
        results.append((config.CLASS_NAMES[predicted_class_index], confidence))
    
    return results

if __name__ == '__main__':
    # Example usage
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predicted_class, confidence = predict_image(image_path)
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")