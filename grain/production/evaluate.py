from grain.data.finetune_masks import *
from grain.utils import *


if __name__ == '__main__':
	'''
	with open(VAL_LIST, 'r') as f:
		filename_list = [line.strip() for line in f.readlines()]

	img_list = [{
		'pic': os.path.join(PICTURES_FOLDER, filename + '.jpg'),
		'highlight': os.path.join(INPUT_FOLDER, filename + '_uv.bmp'),
		'mask': os.path.join(MASK_FOLDER, filename + '.png'),
		'filename': filename
	} for filename in filename_list]
	
	'''

	filename_list = ["0" * (3 - len(str(ind))) + str(ind) for ind in range(1, 501)]


	img_list = [{
		'pic': os.path.join("../g", filename + '.jpg'),
		'mask': os.path.join("../g", filename + '.png'),
		'filename': filename
	} for filename in filename_list]


	model = current_model(WIDTH, HEIGHT)


	p_list = np.zeros(shape=(len(img_list)))
	r_list = np.zeros(shape=(len(img_list)))
	f_list = np.zeros(shape=(len(img_list)))
	ind =  0


	for img_paths in tqdm(img_list):
		pic = cv2.imread(img_paths['pic'], cv2.IMREAD_COLOR)
		mask = cv2.imread(img_paths['mask'], cv2.IMREAD_GRAYSCALE)
		mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
		mask = np.expand_dims(mask, axis=0)

		prediction = np.expand_dims(model.execute(pic), axis=0)


		# precision
		common_clusters, num_of_components, predicted_num_of_components = get_cluster_info(
			prediction,
			255 * (mask >= 50).astype(np.uint8),
			IOU_RATIO
		)

		cc = common_clusters.sum()
		nc = num_of_components.sum()
		npc = predicted_num_of_components.sum()

		p = cc / npc if npc > 0 else 1


		# recall
		common_clusters, num_of_components, predicted_num_of_components = get_cluster_info(
			prediction,
			255 * (mask >= 150).astype(np.uint8),
			IOU_RATIO
		)

		cc = common_clusters.sum()
		nc = num_of_components.sum()
		npc = predicted_num_of_components.sum()

		r = cc / nc if nc > 0 else 1

		# F score
		f = 2 / (1 / p + 1 / r) if p > 0 and r > 0 else 0

		p_list[ind] = p
		r_list[ind] = r
		f_list[ind] = f
		ind += 1


	print("{0} {1} {2}".format(np.asscalar(p_list.mean()), np.asscalar(r_list.mean()), np.asscalar(f_list.mean())))


	# 83 73 78
	#print(precision, recall, F_score)