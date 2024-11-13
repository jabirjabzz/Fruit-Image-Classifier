from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from . import config

def create_model():
    """Create and return the CNN model."""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)),
        MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(config.DROPOUT_RATE),
        Dense(config.NUM_CLASSES, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def load_trained_model(model_path):
    """Load a trained model from disk."""
    from tensorflow.keras.models import load_model
    return load_model(model_path)