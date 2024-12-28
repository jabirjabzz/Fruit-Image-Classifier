import torch
from torchvision import transforms
from PIL import Image
from .model import load_trained_model
from . import config

def predict_image(image_path, model_path):
    # Load model
    model = load_trained_model(model_path)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return config.CLASS_NAMES[predicted.item()]
