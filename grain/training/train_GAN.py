from mxnet import autograd
from grain.data.data import get_data_loaders
from tensorboardX import SummaryWriter
from grain.utils import *
import datetime


clear_folder(CHECKPOINT_FOLDER)


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
train_data, val_data, samples = get_data_loaders(ctx)


model = UNet3()
model.hybridize()
model.load_parameters(os.path.join(ARCHIVE_FOLDER, 'unet3-0.482.params'), ctx=ctx)
#model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
#model.discriminator.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
#model.generator.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
#model.generator.load_parameters('archive/gen-5-0.464.params', ctx=ctx)

scheduler = mx.lr_scheduler.FactorScheduler(step=100, factor=0.9)
generator_trainer = mx.gluon.Trainer(model.collect_params(), 'sgd',
									{'learning_rate': 0.00001, 'momentum': 0.9, 'lr_scheduler': scheduler})
#discriminator_trainer = mx.gluon.Trainer(model.discriminator.collect_params(), 'sgd',
#									{'learning_rate': 0.001, 'momentum': 0.9, 'lr_scheduler': scheduler})
loss_recorder = LossRecorder(SummaryWriter('runs_GAN/' + datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")))


# initializing the model weights
print('''
New session of training:
'''.format(model(samples['train'][0].expand_dims(axis=0)).sum().asscalar()))

imm = samples['val'][0].expand_dims(axis=0)
mmi = samples['val'][1].expand_dims(axis=0)

tbar = tqdm(range(1, NUM_OF_EPOCHS + 1))
best_val_loss = 10 ** 9

CURR_TEXT = ''


gen_loss_func = overall_loss2
#discr_loss_func = log_dice_loss
val_loss_func = dice_loss


#one_hot_encoder = OneHotEncoder(0.6, ctx)

'''
a, b = model(imm)
cv2.imshow('abba0', ndarr_to_cv2(imm[0]).transpose(1, 2, 0))
cv2.imshow('abba1', ndarr_to_cv2(a.softmax(axis=1)[0, 1, :, :]))
cv2.imshow('abba2', ndarr_to_cv2(b.softmax(axis=1)[0, 1, :, :]))
cv2.moveWindow('abba0', 0, 0)
cv2.moveWindow('abba1', 0, 0)
cv2.moveWindow('abba2', 0, 0)
cv2.waitKey(0)
'''


for e in tbar:
	loss_recorder.add_val('learning_rate', generator_trainer.learning_rate)

	for ind, (x, y) in enumerate(train_data):
		with autograd.record():
			#predicted_mask, confidence_map = model(x)
			predicted_mask = model(x)

			#if np.random.rand() < 0.4:
			segm_loss = gen_loss_func(predicted_mask, y)
			#adv_loss = discr_loss_func(confidence_map, nd.zeros_like(y))
			error = segm_loss #+ 0.1 * adv_loss

			trainer = generator_trainer
			loss_recorder.add_val('Losses/train_generator', segm_loss)
			'''
			else:
				generated_mask_loss = discr_loss_func(confidence_map, nd.ones_like(y))

				confidence_map = model.discriminator(one_hot_encoder.encode(y))
				groundtruth_loss = discr_loss_func(confidence_map, nd.zeros_like(y))
				error = (1/2) * generated_mask_loss + (1/2) * groundtruth_loss

				trainer = discriminator_trainer
				loss_recorder.add_val('Losses/train_discriminator', error)'''

		error.backward()
		trainer.step(x.shape[0])
		'''

		a, b = model(imm)
		cv2.imshow('abba0', ndarr_to_cv2(imm[0]).transpose(1, 2, 0))
		cv2.imshow('abba1', ndarr_to_cv2(a.softmax(axis=1)[0, 1, :, :]))
		cv2.imshow('abba2', ndarr_to_cv2(b.softmax(axis=1)[0, 1, :, :]))
		cv2.moveWindow('abba0', 0, 0)
		cv2.moveWindow('abba1', 0, 0)
		cv2.moveWindow('abba2', 0, 0)
		cv2.waitKey(0)
		'''

		if ind % 10 == 0:
			tbar.set_description('step {0:>3}; '.format(ind // 10) + CURR_TEXT)

	common_clusters = np.array([])
	num_of_components = np.array([])
	predicted_num_of_components = np.array([])

	for x, y in val_data:
		output = model(x)#[0]
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

	record_images(model, samples, loss_recorder)
	val_loss = loss_recorder.get_mean('Losses/val', to_clear=False)
	CURR_TEXT = 'Epoch {0:>5}, val loss = {1:<.3}'.format(e, val_loss)

	loss_recorder.update(epoch=e)

	if (val_loss < best_val_loss) or (e % 1 == 0):
		model.save_parameters((WEIGHTS_PREFIX + '-GAN-{0}-{1:.3}.params').format(e, val_loss))
		best_val_loss = min(best_val_loss, val_loss)
