from mxnet import autograd
from grain.data.data import get_data_loaders
from tensorboardX import SummaryWriter
from grain.utils import *
import datetime



if __name__ == '__main__':
	clear_folder(CHECKPOINT_FOLDER)
	os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


	weights = os.path.join(ARCHIVE_FOLDER, 'unet3-0.482.params')
	ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
	_pre = lambda img, ctx: img
	_post = lambda prediction, ctx: prediction
	model = ModelWrapper(UNet3, weights, ctx, _pre, _post)


	train_data, val_data, samples = get_data_loaders(ctx)

	scheduler = mx.lr_scheduler.FactorScheduler(step=100, factor=0.9)
	trainer = mx.gluon.Trainer(model.net.collect_params(), 'sgd',
										{'learning_rate': 0.00001, 'momentum': 0.9, 'lr_scheduler': scheduler})

	run_name = "simple_{0}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S"))
	loss_recorder = LossRecorder(SummaryWriter(os.path.join('.', 'training', 'runs', run_name)))


	# initializing weights
	print('''
	New session of training:
	'''.format(model.execute(samples['train'][0].expand_dims(axis=0)).sum().asscalar()))


	tbar = tqdm(range(1, NUM_OF_EPOCHS + 1))
	best_val_loss = 10 ** 9

	CURR_TEXT = ''


	gen_loss_func = overall_loss2
	val_loss_func = dice_loss



	for e in tbar:
		loss_recorder.add_val('learning_rate', trainer.learning_rate)

		for ind, (x, y) in enumerate(train_data):
			with autograd.record():
				predicted_mask = model.execute(x)
				segm_loss = gen_loss_func(predicted_mask, y)
				loss_recorder.add_val('Losses/train_generator', segm_loss)

			segm_loss.backward()
			trainer.step(x.shape[0])

			if ind % 10 == 0:
				tbar.set_description('step {0:>3}; '.format(ind // 10) + CURR_TEXT)

		common_clusters = np.array([])
		num_of_components = np.array([])
		predicted_num_of_components = np.array([])

		for x, y in val_data:
			output = model.execute(x)
			error = val_loss_func(output, y)
			loss_recorder.add_val('Losses/val', error)


			output = output.softmax(axis=1).argmax(axis=1)

			y_mask = ndarr_to_cv2(y >= 0.5)
			o_mask = ndarr_to_cv2(output >= 0.5)

			o_mask = del_small_components(o_mask, MIN_COMPONENT_SIZE)

			precision, recall, F_score = calc_pixel_metrics(o_mask, y_mask)
			cluster_precision, cluster_recall, cluster_F_score = calc_cluster_metrics(o_mask, y_mask, IOU_RATIO)
			hausdorff = hausdorff_metric(o_mask, y_mask)

			_1, _2, _3 = get_cluster_info(o_mask, y_mask, IOU_RATIO)

			common_clusters = np.concatenate((common_clusters, _1), axis=0)
			num_of_components = np.concatenate((num_of_components, _2), axis=0)
			predicted_num_of_components = np.concatenate((predicted_num_of_components, _3), axis=0)

			loss_recorder.add_val('Metrics/F_score', F_score)
			loss_recorder.add_val('Metrics/precision', precision)
			loss_recorder.add_val('Metrics/recall', recall)
			loss_recorder.add_val('Metrics/hausdorff', hausdorff)

		cluster_precision, cluster_recall, cluster_F_score =\
			calc_metrics(common_clusters, num_of_components, predicted_num_of_components)

		loss_recorder.add_val('Metrics/cluster_precision', cluster_precision)
		loss_recorder.add_val('Metrics/cluster_recall', cluster_recall)
		loss_recorder.add_val('Metrics/cluster_F_score', cluster_F_score)

		record_images(model.net, samples, loss_recorder)
		val_loss = loss_recorder.get_mean('Losses/val', to_clear=False)
		CURR_TEXT = 'Epoch {0:>5}, val loss = {1:<.3}'.format(e, val_loss)

		loss_recorder.update(epoch=e)

		if (val_loss < best_val_loss) or (e % 5 == 0):
			model.net.save_parameters((WEIGHTS_PREFIX + '-{0}-{1:.3}.params').format(e, val_loss))
			best_val_loss = min(best_val_loss, val_loss)
