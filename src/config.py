import os

"""
Configuration settings for the Fruit Classification project (PyTorch version).
"""

# Data parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
NUM_CLASSES = 3

# Model parameters
LEARNING_RATE = 0.001
EPOCHS = 10
DROPOUT_RATE = 0.5

# Paths
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_SAVE_PATH = 'models/fruit_classifier.pth'
CHECKPOINT_DIR = 'models/checkpoints'
LOG_DIR = 'logs'

# Class names
CLASS_NAMES = [
    'Apple_6', 'Apple_Golden_1', 'Apple_Golden_2', 'Apple_Golden_3', 
    'Apple_Granny_Smith_1', 'Apple_Pink_Lady_1', 'Apple_Red_1', 
    'Apple_Red_2', 'Apple_Red_3', 'Apple_Red_Delicious_1', 
    'Apple_Red_Yellow_1', 'Apple_hit_1', 'Apricot_1', 'Avocado_1', 
    'Avocado_ripe_1', 'Banana_1', 'Banana_Lady_Finger_1', 'Beetroot_1', 
    'Blueberry_1', 'Cabbage_white_1', 'Cactus_fruit_1', 'Cantaloupe_1', 
    'Carambula_1', 'Cauliflower_1', 'Cherry_1', 'Cherry_Rainier_1', 
    'Cherry_Wax_Red_1', 'Cherry_Wax_Yellow_1', 'Chestnut_1', 
    'Clementine_1', 'Cocos_1', 'Corn_1', 'Corn_Husk_1', 'Cucumber_1', 
    'Cucumber_3', 'Cucumber_Ripe_1', 'Cucumber_Ripe_2', 'Dates_1', 
    'Eggplant_long_1', 'Fig_1', 'Ginger_Root_1', 'Granadilla_1', 
    'Grape_Blue_1', 'Grape_Pink_1', 'Grape_White_2', 'Grape_White_3', 
    'Guava_1', 'Hazelnut_1', 'Huckleberry_1', 'Kaki_1', 'Kiwi_1', 
    'Kohlrabi_1', 'Lemon_1', 'Limes_1', 'Mandarine_1', 'Mango_Red_1', 
    'Mangostan_1', 'Maracuja_1', 'Melon_Piel_de_Sapo_1', 'Mulberry_1', 
    'Nectarine_1', 'Nectarine_Flat_1', 'Nut_Forest_1', 'Nut_Pecan_1', 
    'Onion_Red_1', 'Onion_White_1', 'Orange_1', 'Papaya_1', 
    'Peach_2', 'Peach_Flat_1', 'Pear_1', 'Pear_2', 'Pear_3', 
    'Pear_Abate_1', 'Pear_Forelle_1', 'Pear_Kaiser_1', 'Pear_Red_1', 
    'Pear_Williams_1', 'Pepino_1', 'Pepper_Green_1', 'Pepper_Orange_1', 
    'Pepper_Red_1', 'Pepper_Yellow_1', 'Physalis_1', 'Pineapple_1', 
    'Pineapple_Mini_1', 'Pitahaya_Red_1', 'Plum_1', 'Plum_2', 
    'Pomelo_Sweetie_1', 'Potato_Red_1', 'Potato_Red_Washed_1', 
    'Potato_Sweet_1', 'Potato_White_1', 'Rambutan_1', 'Raspberry_1', 
    'Redcurrant_1', 'Salak_1', 'Strawberry_1', 'Tamarillo_1', 
    'Tangelo_1', 'Tomato_1', 'Tomato_4', 'Tomato_Cherry_Red_1', 
    'Tomato_Heart_1', 'Tomato_Maroon_1', 'Tomato_Yellow_1', 
    'Tomato_not_Ripened_1', 'Walnut_1', 'Watermelon_1', 
    'Zucchini_1', 'Zucchini_dark_1'
]

# Data augmentation parameters
ROTATION_RANGE = 20
HORIZONTAL_FLIP = True
