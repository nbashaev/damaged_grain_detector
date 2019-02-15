import os
import mxnet as mx
from models import *
import cv2
import numpy as np
from configs import *
from utils import *
import random
from finetune_masks import del_small_components

WEIGHTS_NAME = os.path.join(ARCHIVE_FOLDER, 'unet-14-0.629', 'restored.params')
w = int(2 * WIDTH)
h = int(2 * HEIGHT)

'''
with open(PRESENT_LIST, 'r') as f:
	img_list = [line.strip() + '_led.bmp' for line in f.readlines()]
'''

img_list = ['test.jpg']


folder_name = '../rt/' #INPUT_FOLDER
random.shuffle(img_list)


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

net = UNet2()
net.hybridize()
net.load_parameters(WEIGHTS_NAME, ctx=ctx)

for img_path in img_list:
	'''
	if img_path.split('_')[-1] != 'led.bmp':
		continue
	'''

	pic = cv2.imread(os.path.join(folder_name, img_path), cv2.IMREAD_COLOR)

	prediction = cv2_to_ndarr(cv2.resize(pic, (WIDTH, HEIGHT)), ctx=ctx)
	prediction = net(prediction.expand_dims(axis=0))[0]

	prediction = ndarr_to_cv2(prediction.softmax(axis=0).argmax(axis=0))
	prediction = del_small_components(prediction, MIN_COMPONENT_SIZE)

	pic = cv2.resize(pic, (w, h))
	prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)

	blend3 = draw_mask(pic, [None, None, prediction], 1/3)
	pic = (pic * (2/3)).astype(np.uint8)

	strip = np.zeros((20, w, 3), dtype=np.uint8)
	output_img = np.vstack((pic, strip, blend3))

	cv2.imshow('pic', pic)
	cv2.imshow('blend3', blend3)

	cv2.moveWindow('pic', 0, 0)
	cv2.moveWindow('blend3', 0, 0)

	cv2.waitKey(0)
	#cv2.destroyAllWindows()

