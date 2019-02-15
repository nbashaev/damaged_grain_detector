import mxnet as mx
from mxnet import nd
import numpy as np
import cv2


WIDTH = (192 * 3) // 16 * 16
HEIGHT = (108 * 3) // 16 * 16

MIN_COMPONENT_SIZE = 0.00005


def cv2_to_ndarr(img, ctx):
	if len(img.shape) == 3:
		img = img.transpose((2, 0, 1))

	return mx.nd.array(img / 255.0, ctx=ctx)


def ndarr_to_cv2(img):
	res = img.asnumpy() if type(img) is nd.ndarray.NDArray else img
	return (255 * res).astype(np.uint8)


def del_small_components(mask, ratio, connectivity=4):
	if mask.ndim == 3:
		for i in range(mask.shape[0]):
			mask[i] = del_small_components(mask[i], ratio, connectivity)

		return mask

	num_of_components, labels, stats = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)[0:3]

	for k in range(num_of_components):
		if stats[k][4] < ratio * WIDTH * HEIGHT:
			component_mask = (labels != k).astype(np.uint8)
			mask = cv2.bitwise_and(mask, mask, mask=component_mask)

	return mask


def preprocess(img):
	return cv2_to_ndarr(img, ctx=mx.gpu(0))


def postprocess(img):
	img = ndarr_to_cv2(img)
	return del_small_components(img, MIN_COMPONENT_SIZE)
