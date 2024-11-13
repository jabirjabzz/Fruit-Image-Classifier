import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
from . import config
from .model import create_model
from .data_loader import create_data_generators

def train_model():
    """Train the model and save it."""
    # Create directories if they don't exist
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Create data generators
    train_generator, validation_generator, _ = create_data_generators()

    # Create model
    model = create_model()

    # Callbacks
    callbacks = [
        # Model checkpoint to save best weights
        ModelCheckpoint(
            filepath=os.path.join(config.CHECKPOINT_DIR, 'model_{epoch:02d}_{val_accuracy:.3f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(config.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]

    # Train the model
    history = model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Save the final model
    model.save(config.MODEL_SAVE_PATH)
    
    return history

if __name__ == '__main__':
    train_model()