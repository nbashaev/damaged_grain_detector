from grain.utils import *


folder = "./grain/data/raw2/"
new_folder = INPUT_FOLDER

clear_folder(new_folder)


w = 3 * (1920 // 4)
h = 3 * (1080 // 4)


if __name__ == '__main__':
	for filename in tqdm(os.listdir(folder)):
		parts = re.compile('[_.]').split(filename)
		name = '_'.join(parts[:-2])

		if parts[-1] != 'bmp':
			continue

		pic = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_COLOR)

		pics = [
			pic[:h, :w],
			pic[:h, -w:],
			pic[-h:, :w],
			pic[-h:, -w:]
		]

		for i, new_pic in enumerate(pics):
			new_filename = '_'.join(parts[:-2] + [str(i), parts[-2]]) + '.bmp'
			cv2.imwrite(os.path.join(new_folder, new_filename), new_pic)
