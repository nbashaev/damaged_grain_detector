import os
import mxnet as mx
from models import *
import cv2
import numpy as np
from configs import *
from utils import *
import random
from finetune_masks import get_mask2


weights_name = os.path.join(ARCHIVE_FOLDER, 'unet-GAN-14-0.629.params')


with open(PRESENT_LIST, 'r') as f:
    filename_list = [line.strip() for line in f.readlines()]

img_list = [{
    'pic': os.path.join(INPUT_FOLDER, filename + '_led.bmp'),
    'highlight': os.path.join(INPUT_FOLDER, filename + '_uv.bmp'),
    'mask': os.path.join(MASK_FOLDER, filename + '.png'),
} for filename in filename_list]

random.shuffle(img_list)


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

net = GAN()
net.hybridize()
net.load_parameters(weights_name, ctx=ctx)


for img_paths in img_list:
    pic = cv2.imread(img_paths['pic'], cv2.IMREAD_COLOR)
    mask = cv2.imread(img_paths['mask'], cv2.IMREAD_GRAYSCALE)
    highlight = cv2.imread(img_paths['highlight'], cv2.IMREAD_COLOR)

    raw_mask = get_mask2(cv2.resize(pic, (WIDTH, HEIGHT)), cv2.resize(highlight, (WIDTH, HEIGHT)))

    highlight = cv2.split(highlight)[2]

    prediction = cv2_to_ndarr(cv2.resize(pic, (WIDTH, HEIGHT)), ctx=ctx)
    prediction = net(prediction.expand_dims(axis=0))
    prediction, init_mask = prediction[0][0], prediction[1][0]

    tmp = ndarr_to_cv2(prediction.softmax(axis=0)[1])
    init_mask = ndarr_to_cv2(init_mask.softmax(axis=0)[1])
    prediction = ndarr_to_cv2(prediction.softmax(axis=0).argmax(axis=0))

    prediction = del_small_components(prediction, MIN_COMPONENT_SIZE)

    pic = cv2.resize(pic, (2 * WIDTH, 2 * HEIGHT))
    mask = cv2.resize(mask, (2 * WIDTH, 2 * HEIGHT), interpolation=cv2.INTER_NEAREST)
    prediction = cv2.resize(prediction, (2 * WIDTH,  2 * HEIGHT), interpolation=cv2.INTER_NEAREST)
    highlight = cv2.resize(highlight, (2 * WIDTH,  2 * HEIGHT))
    init_mask = cv2.resize(init_mask, (2 * WIDTH, 2 * HEIGHT), interpolation=cv2.INTER_NEAREST)
    tmp = cv2.resize(tmp, (2 * WIDTH, 2 * HEIGHT))
    raw_mask = cv2.resize(raw_mask, (2 * WIDTH, 2 * HEIGHT), interpolation=cv2.INTER_NEAREST)

    blend1 = draw_mask(pic, [None, mask, prediction], 1/3)
    blend2 = draw_mask(pic, [None, highlight, None], 1/3)
    blend25 = draw_mask(pic, [raw_mask, mask, prediction], 1/3)
    blend3 = draw_mask(pic, [None, None, prediction], 1/3)

    #res = np.hstack((np.vstack((pic, r1)), np.vstack((r2, blend))))

    pic = (pic * (2 / 3)).astype(np.uint8)

    cv2.imshow('pic', pic)
    cv2.imshow('blend1', blend1)
    #cv2.imshow('blend2', blend2)
    #cv2.imshow('blend25', blend25)
    cv2.imshow('blend3', blend3)
    #cv2.imshow('init_mask', init_mask)
    #cv2.imshow('tmp', tmp)

    cv2.moveWindow('pic', 0, 0)
    cv2.moveWindow('blend1', 0, 0)
    #cv2.moveWindow('blend2', 0, 0)
    #cv2.moveWindow('blend25', 0, 0)
    cv2.moveWindow('blend3', 0, 0)
    #cv2.moveWindow('init_mask', 0, 0)
    #cv2.moveWindow('tmp', 0, 0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
