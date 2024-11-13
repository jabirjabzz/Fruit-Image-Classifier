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
├── data/
│   ├── train/
│   │   ├── fruit1/
│   │   ├── fruit2/
│   │   └── fruit3/
│   └── test/
│       ├── fruit1/
│       ├── fruit2/
│       └── fruit3/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Usage

### Training the Model

1. Prepare your dataset in the following structure:
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
