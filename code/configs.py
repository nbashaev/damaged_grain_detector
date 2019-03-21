import os
import re

#WIDTH = (192 * 3) // 16 * 16
#HEIGHT = (108 * 3) // 16 * 16

WIDTH = 320
HEIGHT = 160

INPUT_FOLDER = os.path.join('.', 'data', 'quarters')
PICTURES_FOLDER = os.path.join('.', 'data', 'pictures')
MASK_FOLDER = os.path.join('.', 'data', 'masks')
TRAIN_LIST = os.path.join('.', 'data', 'train.txt')
VAL_LIST = os.path.join('.', 'data', 'val.txt')

TRAIN_RATIO = 0.7
VAL_RATIO = 0.3
BATCH_SIZE = 4
NUM_OF_EPOCHS = 30
IOU_RATIO = 0.2
COLORED_COMPONENT_SIZE_THRESHOLD = 0.00005
COMPONENT_SIZE_THRESHOLD = 0.0001
MIN_COMPONENT_SIZE = 0.00005
ARCHIVE_FOLDER = os.path.join('.', 'models', 'archive')
ONNX_FOLDER = os.path.join('.', 'models', 'onnx')
CHECKPOINT_FOLDER = os.path.join('.', 'models', 'checkpoints')
WEIGHTS_PREFIX = os.path.join(CHECKPOINT_FOLDER, 'unet3')
CUR_MODEL = 'unet3-25-0.547'
