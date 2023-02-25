import os


# Image color on gray scale
GRAY_SCALE_BLACK = 0
GRAY_SCALE_WHITE = 255

# Image color in BGR color space
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Extension and image name
AMBERJACK = 'amberjack'
EXTENSION_PNG = '.png'
EXTENSION_CSV = '.csv'

# ML algorithm
LSTM = 'lstm'

# Path
BASE_RESOURCE_DIR = './resource'
BASE_INPUT_DIR = os.path.join(BASE_RESOURCE_DIR, 'input')
BASE_OUTPUT_DIR = os.path.join(BASE_RESOURCE_DIR, 'output')
PARAMS_DIR_NAME = 'params'
IMAGE_DIR_NAME = 'image'
CSV_DIR_NAME = 'csv'
CSV_BASE_NAME = 'created_images_info'
CREATED_IMG_INFO_FILE_NAME = f"{CSV_BASE_NAME}{EXTENSION_CSV}"
PARAMS_DIR_PATH = os.path.join(BASE_INPUT_DIR, PARAMS_DIR_NAME)
OUTPUT_CSV_BASE_DIR_PATH = os.path.join(BASE_OUTPUT_DIR, CSV_DIR_NAME)
IMAGE_DIR_PATH = os.path.join(BASE_OUTPUT_DIR, IMAGE_DIR_NAME)
INFO_FILE_PATH = os.path.join(IMAGE_DIR_PATH, CREATED_IMG_INFO_FILE_NAME)
LSTM_CSV_DIR_PATH = os.path.join(OUTPUT_CSV_BASE_DIR_PATH, LSTM)
OUTPUT_CSV_BASE_DIR_PATH = os.path.join(BASE_OUTPUT_DIR, CSV_DIR_NAME)
