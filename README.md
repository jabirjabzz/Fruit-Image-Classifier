# Fruit Image Classification using TensorFlow

A deep learning project that classifies different types of fruits using TensorFlow and Convolutional Neural Networks (CNN). The model is trained to identify three different types of fruits from images.

## 🎯 Features

- Image classification using CNN
- Support for three different fruit categories
- Data augmentation for better model generalization
- Easy-to-use prediction interface
- Trained model saving and loading functionality

## 🔧 Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pillow (PIL)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/jabirjabzz/Fruit-Image-Classifier.git
cd Fruiit-Image-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
fruit-classification/
│
├── data/                          # Dataset directory
│   ├── train/                     # Training data
│   │   ├── apples/               # Apple images
│   │   ├── bananas/              # Banana images
│   │   └── oranges/              # Orange images
│   │
│   ├── test/                     # Testing data
│   │   ├── apples/
│   │   ├── bananas/
│   │   └── oranges/
│   │
│   └── validation/               # Validation data
│       ├── apples/
│       ├── bananas/
│       └── oranges/
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── data_loader.py           # Data loading utilities
│   ├── model.py                 # Model architecture definition
│   ├── train.py                 # Training script
│   ├── predict.py               # Prediction script
│   └── utils.py                 # Utility functions
│
├── notebooks/                    # Jupyter notebooks
│   ├── data_exploration.ipynb   # Dataset analysis
│   ├── model_training.ipynb     # Training experiments
│   └── results_analysis.ipynb   # Performance analysis
│
├── models/                      # Saved models
│   ├── checkpoints/            # Training checkpoints
│   └── fruit_classifier.h5     # Final trained model
│
├── logs/                       # Training logs
│   └── tensorboard/           # Tensorboard logs
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_utils.py
│
├── docs/                      # Documentation
│   ├── api.md                # API documentation
│   ├── setup.md              # Setup guide
│   └── usage.md              # Usage examples
│
├── scripts/                   # Utility scripts
│   ├── setup.sh              # Environment setup
│   └── download_data.sh      # Dataset download
│
├── .gitignore                # Git ignore file
├── LICENSE                   # License file
├── README.md                 # Project documentation
├── requirements.txt          # Project dependencies
└── setup.py                 # Package setup file

## 🚀 Usage

### Training the Model

1. Prepare your dataset :
```
data/
├── train/
│   ├── fruit1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── fruit2/
│   └── fruit3/
└── test/
    ├── fruit1/
    ├── fruit2/
    └── fruit3/
```

2. Run the training script:
```bash
python src/train.py
```

### Making Predictions

```python
from src.predict import predict_fruit

# Predict a single image
result = predict_fruit('path/to/your/image.jpg')
print(f"Predicted fruit: {result}")
```

## 📊 Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout layer (0.5) to prevent overfitting
- Dense layers for classification
- Softmax output layer for 3 classes

## 📈 Performance

The model achieves the following performance metrics on the test set:
- Accuracy: [Add your accuracy]
- Loss: [Add your loss]

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Mohammed Jabir Hussain - Initial work - [jabirjabzz](https://github.com/jabirjabzz)

## 🙏 Acknowledgments

- TensorFlow team for the amazing framework
- [Add any other acknowledgments]

## 📧 Contact

Mohammed Jabir Hussain  - jabirmoh07@gmail.com

Project Link: [https://github.com/jabirjabzz/Fruit-Image-Classifier]
project done by mohammed jabir hussain
