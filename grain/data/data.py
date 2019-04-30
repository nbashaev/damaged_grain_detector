import os

import cv2
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data import dataset
from imgaug import augmenters as iaa

from configs import *
from utils import *
import sys
sys.path.append('./data')
from aug_sandbox import seq_default


class ImageWithMaskDataset(dataset.Dataset):
	def __init__(self, list_path, ctx, seq=seq_default):
		self._ctx = ctx
		self._seq = seq

		with open(list_path, 'r') as f:
			filename_list = [line.strip() for line in f.readlines()]

		self._img_list = [{
			'pic': os.path.join(PICTURES_FOLDER, filename + '.jpg'),
			'mask': os.path.join(MASK_FOLDER, filename + '.png')
		} for filename in filename_list]

	def _transform(self, pic, mask):
		seq_det = self._seq.to_deterministic()

		pic = seq_det.augment_image(pic)
		pic = cv2_to_ndarr(pic, ctx=self._ctx)

		mask = seq_det.augment_image(mask)
		mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
		mask = cv2_to_ndarr(mask, ctx=self._ctx)

		return pic, mask

	def __getitem__(self, idx):
		pic = cv2.imread(self._img_list[idx]['pic'], cv2.IMREAD_COLOR)
		mask = cv2.imread(self._img_list[idx]['mask'], cv2.IMREAD_GRAYSCALE)

		return self._transform(pic, mask)

	def __len__(self):
		return len(self._img_list)


def get_data_loaders(ctx):
	train_dataset = ImageWithMaskDataset(TRAIN_LIST, ctx, seq=seq_default)
	val_dataset = ImageWithMaskDataset(VAL_LIST, ctx, seq=iaa.Sequential([]))

	data_samples = {
		'train': train_dataset[0],
		'val': val_dataset[0]
	}

	train_data_loader = mx.gluon.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_data_loader = mx.gluon.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
	
	return train_data_loader, val_data_loader, data_samples
