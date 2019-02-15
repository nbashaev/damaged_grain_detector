import os
import re

WIDTH = (192 * 3) // 16 * 16
HEIGHT = (108 * 3) // 16 * 16

INPUT_FOLDER = os.path.join('.', 'data', 'raw2')
PICTURES_FOLDER = os.path.join('.', 'data', 'pictures')
MASK_FOLDER = os.path.join('.', 'data', 'masks')
TRAIN_LIST = os.path.join('.', 'data', 'train.txt')
VAL_LIST = os.path.join('.', 'data', 'val.txt')
PRE_LIST = os.path.join('.', 'presentation', 'pre_list.txt')
PRESENT_LIST = os.path.join('.', 'presentation', 'list.txt')

TRAIN_RATIO = 0.0
VAL_RATIO = 1.0
BATCH_SIZE = 1
NUM_OF_EPOCHS = 30
IOU_RATIO = 0.2
COLORED_COMPONENT_SIZE_THRESHOLD = 0.00005
COMPONENT_SIZE_THRESHOLD = 0.0001
MIN_COMPONENT_SIZE = 0.00005
ARCHIVE_FOLDER = os.path.join('.', 'archive')
CHECKPOINT_FOLDER = os.path.join('.', 'checkpoints')
WEIGHTS_PREFIX = os.path.join(CHECKPOINT_FOLDER, 'unet')


for directory in [PICTURES_FOLDER, MASK_FOLDER, CHECKPOINT_FOLDER, ARCHIVE_FOLDER]:
	if not os.path.isdir(directory):
		os.makedirs(directory)
