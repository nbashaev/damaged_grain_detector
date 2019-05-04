from grain.utils import *
from grain.data.finetune_masks import get_mask


if __name__ == '__main__':
	clear_folder(PICTURES_FOLDER)
	clear_folder(MASK_FOLDER)

	train_list = open(TRAIN_LIST, 'w')
	val_list = open(VAL_LIST, 'w')

	for filename in tqdm(os.listdir(INPUT_FOLDER)):
		parts = re.compile('[_.]').split(filename)

		if parts[-1] != 'bmp' or parts[-2] != 'uv':
			continue

		name = '_'.join(parts[:-2])

		uv_img = cv2.imread(os.path.join(INPUT_FOLDER, filename), cv2.IMREAD_COLOR)
		pic = cv2.imread(os.path.join(INPUT_FOLDER, filename.replace('uv', 'led')), cv2.IMREAD_COLOR)

		uv_img = cv2.resize(uv_img, (WIDTH, HEIGHT))
		pic = cv2.resize(pic, (WIDTH, HEIGHT))
		mask = get_mask(pic, uv_img)

		if mask.sum() < (2 * MIN_COMPONENT_SIZE) * WIDTH * HEIGHT:
			continue

		x = np.random.rand()

		if x < TRAIN_RATIO + VAL_RATIO:
			(train_list if (x < TRAIN_RATIO) else val_list).write(name + '\n')

			cv2.imwrite(os.path.join(PICTURES_FOLDER, name + '.jpg'), pic)
			cv2.imwrite(os.path.join(MASK_FOLDER, name + '.png'), mask)

	train_list.close()
	val_list.close()
