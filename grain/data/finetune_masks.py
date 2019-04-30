import os
import numpy as np
import cv2
from configs import *
from utils import *
import random

kernel_2_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel_3_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def get_mask(pic, img):
	r = cv2.split(img)[2]
	red_mask = get_mask2(pic, img)
	red_mask = del_small_components2(r, red_mask, COLORED_COMPONENT_SIZE_THRESHOLD)
	red_mask = del_small_components(red_mask, COMPONENT_SIZE_THRESHOLD)
	red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_3_3, iterations=2)

	return red_mask


def get_mask2(pic, img):
	gray_pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

	boundary = cv2.Laplacian(gray_pic, cv2.CV_32F)
	boundary = cv2.dilate(boundary, kernel_2_2, iterations=1)
	boundary = cv2.normalize(boundary, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
	boundary = adjust_gamma(boundary, 0.5)
	boundary = cv2.adaptiveThreshold(boundary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 5)
	boundary = del_small_components(boundary, 0.0002)

	boundary = boundary.astype(np.float32)
	b, g, r = cv2.split(img.astype(np.float32))

	abba = cv2.adaptiveThreshold(r.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -55)

	red_part = cv2.normalize((2 * r - 1 * b) - 2 * boundary + 3.8 * gray_pic.astype(np.float32) + 0.2 * abba, None, 0,
							 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

	red_mask = cv2.adaptiveThreshold(red_part, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -50)
	red_mask = np.minimum(red_mask, 255 * (r > 45).astype(np.uint8))
	red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_3_3, iterations=1)

	return red_mask


if __name__ == '__main__':
	os.chdir('..')
	folder_name = INPUT_FOLDER

	filename_list = os.listdir(folder_name)
	random.shuffle(filename_list)


	for filename in filename_list:
		parts = re.compile('[_.]').split(filename)

		if not (parts[-1] == 'bmp' and parts[-2] == 'uv'):
			continue

		img = cv2.imread(os.path.join(folder_name, filename), cv2.IMREAD_COLOR)
		img = cv2.resize(img, (WIDTH, HEIGHT))

		pic = cv2.imread(os.path.join(folder_name, filename.replace('uv', 'led')), cv2.IMREAD_COLOR)
		pic = cv2.resize(pic, (WIDTH, HEIGHT))

		mask = get_mask(pic, img)

		#r1 = cv2.bitwise_and(img, img, mask=mask)
		#r2 = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
		r3 = draw_mask(pic, [None, mask, None], 1/5)

		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
		res = np.hstack((np.vstack((pic, mask)), np.vstack((img, r3))))

		res = cv2.resize(res, (4 * WIDTH, 4 * HEIGHT), interpolation=cv2.INTER_NEAREST)

		cv2.imshow('res', res)
		cv2.moveWindow('res', 50, 50)

		if cv2.waitKey(0) == ord('q'):
			break
