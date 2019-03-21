import os
os.chdir('..')

from data.finetune_masks import *
from tqdm import tqdm


if __name__ == '__main__':
	with open(VAL_LIST, 'r') as f:
		filename_list = [line.strip() for line in f.readlines()]

	img_list = [{
		'pic': os.path.join(PICTURES_FOLDER, filename + '.jpg'),
		'highlight': os.path.join(INPUT_FOLDER, filename + '_uv.bmp'),
		'mask': os.path.join(MASK_FOLDER, filename + '.png'),
		'filename': filename
	} for filename in filename_list]


	model = current_model(WIDTH, HEIGHT)


	_cc = 1
	_nc = 1
	_npc = 1

	arr = []

	for img_paths in tqdm(img_list):
		pic = cv2.imread(img_paths['pic'], cv2.IMREAD_COLOR)
		highlight = cv2.resize(cv2.imread(img_paths['highlight'], cv2.IMREAD_COLOR), (WIDTH, HEIGHT))
		mask = np.expand_dims(cv2.imread(img_paths['mask'], cv2.IMREAD_GRAYSCALE), axis=0)

		prediction = np.expand_dims(model.execute(pic), axis=0)
		common_clusters, num_of_components, predicted_num_of_components = get_cluster_info(prediction, mask, IOU_RATIO)

		cc = common_clusters.sum()
		nc = num_of_components.sum()
		npc = predicted_num_of_components.sum()

		_cc += cc
		_nc += nc
		_npc += npc

		p = cc / npc if npc > 0 else 1
		r = cc / nc
		f = 2 / (1 / p + 1 / r) if p > 0 and r > 0 else 0

		arr.append((img_paths['filename'], nc, f, p, r))


	arr.sort(key=lambda x: -x[2])

	with open(os.path.join('production', 'presentation', 'log.txt'), 'w') as f:
		for entry in arr:
			f.write('{0} {1} {2} {3} {4}\n'.format(entry[0], entry[1], entry[2], entry[3], entry[4]))


	recall = _cc / _nc
	precision = _cc / _npc
	F_score = 2 / (1 / recall + 1 / precision)

	# 83 73 78
	print(precision, recall, F_score)