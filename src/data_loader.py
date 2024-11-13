from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from pathlib import Path
from . import config

def create_data_generators():
    """Create train, validation, and test data generators."""
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT_RANGE,
        height_shift_range=config.HEIGHT_SHIFT_RANGE,
        shear_range=config.SHEAR_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        fill_mode=config.FILL_MODE
    )

    # Validation and test data generator (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        classes=config.CLASS_NAMES
    )

    validation_generator = test_datagen.flow_from_directory(
        config.VALIDATION_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        classes=config.CLASS_NAMES
    )

    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        classes=config.CLASS_NAMES
    )

    return train_generator, validation_generator, test_generator

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for prediction."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=config.CHANNELS)
    img = tf.image.resize(img, [config.IMG_HEIGHT, config.IMG_WIDTH])
    img = img / 255.0
    return tf.expand_dims(img, 0)