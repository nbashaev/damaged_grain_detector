from grain.utils import *
import random


if __name__ == '__main__':
	w, h = 5 * WIDTH, 5 * HEIGHT
	model = current_model(w, h)


	with open(VAL_LIST, 'r') as f:
		img_list = [line.strip() + '_led.bmp' for line in f.readlines()]

	folder_name = INPUT_FOLDER
	random.shuffle(img_list)


	for img_path in img_list:
		pic = cv2.imread(os.path.join(folder_name, img_path), cv2.IMREAD_COLOR)
		prediction = model.execute(pic)

		pic = cv2.resize(pic, (w, h))
		blend = draw_mask(pic, [None, None, prediction], 2/3)

		cv2.imshow('pic', pic)
		cv2.imshow('blend', blend)

		cv2.moveWindow('pic', 0, 0)
		cv2.moveWindow('blend', 0, 0)

		if cv2.waitKey(0) == ord('q'):
			break
