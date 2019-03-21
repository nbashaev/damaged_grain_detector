import os
os.chdir('..')

from configs import *
from tqdm import tqdm
import cv2
import numpy as np


if __name__ == '__main__':
	for filename in tqdm(os.listdir(INPUT_FOLDER)):
		parts = re.compile('[_.]').split(filename)

		if parts[-1] != 'bmp' or parts[-2] != 'uv':
			continue

		name = '_'.join(parts[:-2])

		pic_path = os.path.join(INPUT_FOLDER, filename.replace('uv', 'led'))
		uv_path = os.path.join(INPUT_FOLDER, filename)

		pic = cv2.imread(pic_path, cv2.IMREAD_COLOR)
		pic = cv2.resize(pic, (2 * WIDTH, 2 * HEIGHT))
		b, g, r = cv2.split(pic.astype(np.float32))

		dd = ((r - (b + r) / 2) > 45).astype(np.uint8)

		if dd.sum() > 0.0001 * (2 * WIDTH * 2 * HEIGHT):
			#cv2.imshow('abba', pic)
			#cv2.waitKey(0)

			os.remove(pic_path)
			os.remove(uv_path)
