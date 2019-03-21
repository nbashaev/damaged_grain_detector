from mxnet import nd
import numpy as np
import cv2
from configs import *
import mxnet as mx

import sys
sys.path.append('./models')
from models import *

class LossRecorder:
	def __init__(self, writer):
		self._scalars = {}
		self._pics = {}
		self._writer = writer
	
	def check_key(self, name):
		if name not in self._scalars:
			self._scalars[name] = []
	
	def add_val(self, name, val):
		self.check_key(name)
		
		if type(val) is nd.NDArray:
			val = val.mean().asscalar()

		if type(val) is np.ndarray:
			val = np.asscalar(val.mean())

		
		self._scalars[name].append(val)

	def get_mean(self, name, to_clear=True):
		self.check_key(name)
		
		if len(self._scalars[name]) == 0:
			return 0
		
		mean_val = sum(self._scalars[name]) / len(self._scalars[name])
		
		if to_clear:
			self._scalars[name].clear()
		
		return mean_val

	def add_pic(self, name, pic):
		self._pics[name] = pic

	def update(self, epoch):
		for scalar_name in self._scalars:
			if len(self._scalars[scalar_name]) > 0:
				self._writer.add_scalar(scalar_name, self.get_mean(scalar_name), epoch)

		for pic_name in self._pics:
			self._writer.add_image(pic_name, self._pics[pic_name], epoch)


def draw_mask(img, masks, w):
	channels = cv2.split(img)

	for c in range(3):
		mask = 0 * channels[c] if masks[c] is None else masks[c]
		channels[c] = ((1 - w) * channels[c] + w * np.maximum(channels[c], mask)).astype('uint8')

	return cv2.merge(channels)


def adjust_gamma(image, gamma=1.0):
	inv_gamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

	return cv2.LUT(image, table)


def clear_folder(folder_path):
	for filename in os.listdir(folder_path):
		file_path = os.path.join(folder_path, filename)
		os.unlink(file_path)


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


def del_small_components2(img, mask, ratio, connectivity=4):
	num_of_components, labels, stats = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)[0:3]

	for k in range(num_of_components):
		area = (img * (labels == k)).sum() / 255

		if area < ratio * WIDTH * HEIGHT:
			component_mask = (labels != k).astype(np.uint8)
			mask = cv2.bitwise_and(mask, mask, mask=component_mask)

	return mask


cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
cross_entropy2 = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=1, sparse_label=False)


def dice_loss(net_output, ground_truth):
	net_output = net_output.softmax(axis=1)[:, 1, :, :]

	pr_sz = 1 + (net_output * net_output).sum(axis=0, exclude=True)
	int_sz = 1 + (net_output * ground_truth).sum(axis=0, exclude=True)
	mask_sz = 1 + (ground_truth * ground_truth).sum(axis=0, exclude=True)

	F_score = 2 / (pr_sz / int_sz + mask_sz / int_sz)

	return 1 - F_score


def log_dice_loss(net_output, ground_truth, _eps=1e-12):
	F_score = 1 - dice_loss(net_output, ground_truth)
	return (-nd.log(nd.minimum(F_score + _eps, 1)) + (1 - F_score)) / 2


def overall_loss(net_output, ground_truth):
	loss1 = log_dice_loss(net_output, ground_truth)
	loss2 = focal_loss(net_output, ground_truth)

	return (1/2) * loss1 + (1/2) * loss2


def log_dice_loss2(net_output, ground_truth, _eps=1e-12):
	net_output = net_output.softmax(axis=1)[:, 1, :, :]

	int_sz = 1 + (net_output * ground_truth).sum(axis=0, exclude=True)
	mask_sz = 1 + (ground_truth * ground_truth).sum(axis=0, exclude=True)
	pr_sz = 1 + (net_output * net_output).sum(axis=0, exclude=True)

	F_score = 1 / (0.2 * pr_sz / int_sz + 0.8 * mask_sz / int_sz)

	loss = 0.7 * (-nd.log(nd.minimum(F_score + _eps, 1))) + 0.3 * (1 - F_score)

	return loss


def focal_loss(pred, label, _alpha=0.25, _gamma=2, _eps=1e-12):
	pred = pred.softmax(axis=1)
	one_hot = nd.one_hot(label, 2).transpose((0, 3, 1, 2))

	pt = nd.where(one_hot, pred, 1 - pred)
	t = nd.ones_like(one_hot)
	alpha = nd.where(one_hot, _alpha * t, (1 - _alpha) * t)
	loss = -alpha * ((1 - pt) ** _gamma) * nd.log(nd.minimum(pt + _eps, 1))

	return nd.mean(loss, axis=0, exclude=True)


def overall_loss2(net_output, ground_truth):
	loss1 = log_dice_loss2(net_output, ground_truth)
	loss2 = focal_loss(net_output, ground_truth)

	return (7/10) * loss1 + (3/10) * loss2


def ndarr_to_cv2(img):
	res = img.asnumpy() if type(img) is nd.ndarray.NDArray else img
	return (255 * res).astype(np.uint8)


def cv2_to_ndarr(img, ctx):
	if len(img.shape) == 3:
		img = img.transpose((2, 0, 1))

	return mx.nd.array(img / 255.0, ctx=ctx)


def is_valid(mask, groundtruth, ratio):
	return ((mask > 0) & (groundtruth > 0)).sum() > ratio * (mask > 0).sum()


def IOU(mask, groundtruth):
	if mask.shape[0] == 3:
		return np.array([IOU(m, g) for m, g in zip(mask, groundtruth)])

	intersection = ((mask > 0) & (groundtruth > 0)).sum()
	union = 1 + ((mask > 0) | (groundtruth > 0)).sum()
	return intersection / union


class OneHotEncoder:
	def __init__(self, cw, ctx):
		self._ctx = ctx

		w = (1 - cw) / 12
		self._kernel = nd.array([[    w, 2 * w,     w],
								 [2 * w,    cw, 2 * w],
								 [    w, 2 * w,     w]], ctx=self._ctx)

		self._weight = self._kernel.expand_dims(0).expand_dims(0)

		self._conv = mx.gluon.nn.Conv2D(channels=1, kernel_size=(3, 3), padding=(1, 1))
		self._conv._in_channels = 1
		self._conv.initialize(ctx=self._ctx)
		self._conv.weight.set_data(self._weight)

	def encode(self, y):
		tmp = y + nd.random.normal(0, 0.02, y.shape, ctx=self._ctx)
		tmp = self._conv(tmp.expand_dims(axis=1))
		tmp = nd.maximum(tmp, 0) / nd.maximum(tmp.max(), 1.0)

		res = nd.concat(1 - tmp, tmp, dim=1)

		'''
		cv2.imshow('hey', ndarr_to_cv2(res[0, 0, :, :]))
		cv2.imshow('hey0.5', ndarr_to_cv2(res[0, 1, :, :]))
		cv2.imshow('hey2', ndarr_to_cv2(y[0]))
		cv2.moveWindow('hey', 0, 0)
		cv2.moveWindow('hey0.5', 0, 0)
		cv2.moveWindow('hey2', 0, 0)
		cv2.waitKey(0)
		'''

		return res


def record_images(model, samples, loss_recorder):
	for pref in ['train', 'val']:
		predicted_mask = model(samples[pref][0].expand_dims(axis=0))
		predicted_mask = ndarr_to_cv2(predicted_mask.softmax(axis=1)[0, 1, :, :])

		pic = ndarr_to_cv2(samples[pref][0])[::-1, :, :].transpose((1, 2, 0))
		ground_truth = ndarr_to_cv2(samples[pref][1])

		predicted_mask_binary = cv2.threshold(predicted_mask, 127, 255, cv2.THRESH_BINARY)[1]
		img_with_masks = draw_mask(pic, [predicted_mask_binary, ground_truth, None], 1/5)

		predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2RGB)
		ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2RGB)

		final_output = np.vstack((np.hstack((predicted_mask, predicted_mask)),
								  np.hstack((ground_truth, img_with_masks)))).transpose((2, 0, 1))

		loss_recorder.add_pic('{}_pic'.format(pref), final_output)


def record_images2(model, samples, loss_recorder):
	for pref in ['train', 'val']:
		predicted_mask, confidence_map = model(samples[pref][0].expand_dims(axis=0))
		confidence_map = ndarr_to_cv2(confidence_map.softmax(axis=1)[0, 1, :, :])
		predicted_mask = ndarr_to_cv2(predicted_mask.softmax(axis=1)[0, 1, :, :])

		pic = ndarr_to_cv2(samples[pref][0])[::-1, :, :].transpose((1, 2, 0))
		ground_truth = ndarr_to_cv2(samples[pref][1])

		predicted_mask_binary = cv2.threshold(predicted_mask, 127, 255, cv2.THRESH_BINARY)[1]
		img_with_masks = draw_mask(pic, [predicted_mask_binary, ground_truth, None], 1/5)

		predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2RGB)
		confidence_map = cv2.cvtColor(confidence_map, cv2.COLOR_GRAY2RGB)
		ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2RGB)

		final_output = np.vstack((np.hstack((predicted_mask, confidence_map)),
								  np.hstack((ground_truth, img_with_masks)))).transpose((2, 0, 1))

		loss_recorder.add_pic('{}_pic'.format(pref), final_output)


def calc_pixel_metrics(o_mask, y_mask):
	res_sz = 1 + (o_mask > 0).sum(axis=(1, 2))
	y_sz = 1 + (y_mask > 0).sum(axis=(1, 2))
	int_sz = ((o_mask > 0) & (y_mask > 0)).sum(axis=(1, 2))

	precision = int_sz / res_sz
	recall = int_sz / y_sz
	F_score = 2 * int_sz / (res_sz + y_sz)

	return precision, recall, F_score


def calc_cluster_metrics(o_mask, y_mask, ratio):
	common_clusters, num_of_components, predicted_num_of_components = get_cluster_info(o_mask, y_mask, ratio)
	return calc_metrics(common_clusters, num_of_components, predicted_num_of_components)


def calc_metrics(common_clusters, num_of_components, predicted_num_of_components):
	cluster_precision = np.array((1 + common_clusters.sum()) / (1 + predicted_num_of_components.sum()))
	cluster_recall = np.array((1 + common_clusters.sum()) / (1 + num_of_components.sum()))
	cluster_F_score = np.array(2 / (1 / cluster_precision + 1 / cluster_recall))

	return cluster_precision, cluster_recall, cluster_F_score


def get_cluster_info(o_mask, y_mask, ratio):
	common_clusters = np.zeros(shape=(o_mask.shape[0],))
	num_of_components = np.zeros(shape=(o_mask.shape[0],), dtype=np.uint8)
	predicted_num_of_components = np.zeros(shape=(o_mask.shape[0],), dtype=np.uint8)

	for i in range(o_mask.shape[0]):
		num_of_components[i], labels = cv2.connectedComponentsWithStats(y_mask[i], connectivity=4)[0:2]
		predicted_num_of_components[i] = cv2.connectedComponentsWithStats(o_mask[i], connectivity=4)[0]

		for k in range(1, num_of_components[i]):
			if is_valid(labels == k, o_mask[i], ratio):
				common_clusters[i] += 1

	num_of_components -= 1
	predicted_num_of_components -= 1

	return common_clusters, num_of_components, predicted_num_of_components


def hausdorff_metric(o_mask, y_mask):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	iterations = 1

	for i in range(o_mask.shape[0]):
		o_mask[i] = cv2.dilate(o_mask[i], kernel, iterations=iterations)
		y_mask[i] = cv2.dilate(y_mask[i], kernel, iterations=iterations)

	return np.array(1 - IOU(o_mask, y_mask))


def construct_prepost(w, h):
	def _pre(img, ctx):
		img = cv2.resize(img, (WIDTH, HEIGHT))
		img = cv2_to_ndarr(img, ctx=ctx)
		img = img.expand_dims(axis=0)

		return img


	def _post(prediction, ctx):
		prediction = prediction[0].softmax(axis=0).argmax(axis=0)
		prediction = ndarr_to_cv2(prediction)
		prediction = del_small_components(prediction, MIN_COMPONENT_SIZE)
		prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)

		return prediction

	return _pre, _post


def current_model(w, h):
	os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

	weights_name = os.path.join(ARCHIVE_FOLDER, CUR_MODEL + '.params')
	ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
	_pre, _post = construct_prepost(w, h)

	return ModelWrapper(UNet3, weights_name, ctx, _pre, _post)