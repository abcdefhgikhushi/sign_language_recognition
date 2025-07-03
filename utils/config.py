import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_IMAGES_DIR = os.path.join(DATA_DIR, 'raw_images')
TRAIN_IMAGES_DIR = os.path.join(RAW_IMAGES_DIR, 'asl_alphabet_train')
TEST_IMAGES_DIR = os.path.join(RAW_IMAGES_DIR, 'asl_alphabet_test')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_data')  # Updated path
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')

# Model parameters
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
MAX_RAM_USAGE = 4  # GB
SAFE_MEMORY_RATIO = 0.7  # Use only 70% of MAX_RAM to be safe

# Sign language classes (updated for ASL dataset)
CLASSES = [
    # Letters (A-Z)
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',

    # Numbers (0-9)
   # '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

    # Control gestures
    'SPACE', 'DEL','NOTHING',

    # Common words
    #'HELLO', 'THANK_YOU', 'YES', 'NO'
]
# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640