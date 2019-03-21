import os
os.chdir('..')

import cv2
from configs import *
from tqdm import tqdm
from utils import *
from models.models import *


MARKUP_PICTURES_FOLDER = "./markup/pictures/"
MARKUP_MASKS_FOLDER = "./markup/masks/"
LIST_NAME = "./markup/grn.001.txt"

w, h = 1920, 1080


if __name__ == '__main__':
	clear_folder(MARKUP_PICTURES_FOLDER)
	clear_folder(MARKUP_MASKS_FOLDER)


	model = current_model(w, h)


	with open(LIST_NAME) as f:
		processed_pics = [line.strip().split(" ")[0] for line in f.readlines()]


	for filename in tqdm(os.listdir(INPUT_FOLDER)):
		parts = re.compile('[_.]').split(filename)
		name = '_'.join(parts[:-2])

		if parts[-1] != 'bmp' or parts[-2] != 'led':
			continue

		if (name + ".jpg") in processed_pics:
			continue


		pic = cv2.imread(os.path.join(INPUT_FOLDER, filename), cv2.IMREAD_COLOR)
		mask = model.execute(pic)

		if mask.sum() == 0:
			continue

		resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

		cv2.imwrite(os.path.join(MARKUP_PICTURES_FOLDER, name + '.jpg'), pic)
		cv2.imwrite(os.path.join(MARKUP_MASKS_FOLDER, name + '.png'), resized_mask)
