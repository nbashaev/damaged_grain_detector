import mxnet as mx
from mxnet import autograd
from data import get_data_loaders
from models import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from configs import *
from utils import *
import datetime


clear_folder(CHECKPOINT_FOLDER)


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
train_data, val_data, samples = get_data_loaders(ctx)


model = UNet2()
model.hybridize()
model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
#model.load_parameters('archive/unet2-batchless-0.787.params', ctx=ctx)


schedule = mx.lr_scheduler.PolyScheduler(max_update=80, base_lr=0.001, pwr=1)
trainer = mx.gluon.Trainer(model.collect_params(), 'adam', {'lr_scheduler': schedule})
loss_recorder = LossRecorder(SummaryWriter('runs/' + datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")))


# initializing the model weights
print('''
The new session of training:
'''.format(model(samples['train'][0].expand_dims(axis=0)).sum().asscalar()))

tbar = tqdm(range(1, NUM_OF_EPOCHS + 1))
best_val_loss = 10 ** 9


for e in tbar:
	for x, y in train_data:
		with autograd.record():
			output = model(x)
			error = overall_loss(output, y)

		error.backward()
		trainer.step(x.shape[0])

		loss_recorder.add_val('Losses/train', error)
		loss_recorder.add_val('learning_rate', trainer.learning_rate)

	for x, y in val_data:
		output = model(x)
		error = overall_loss(output, y)
		loss_recorder.add_val('Losses/val', error)

		output = output.softmax(axis=1).argmax(axis=1)

		y_mask = ndarr_to_cv2(y >= 0.5)
		o_mask = ndarr_to_cv2(output >= 0.5)

		precision, recall, F_score = calc_pixel_metrics(o_mask, y_mask)
		cluster_precision, cluster_recall, cluster_F_score = calc_cluster_metrics(o_mask, y_mask, 0.2)
		hausdorff = hausdorff_metric(o_mask, y_mask)

		loss_recorder.add_val('Metrics/F_score', F_score)
		loss_recorder.add_val('Metrics/precision', precision)
		loss_recorder.add_val('Metrics/recall', recall)
		loss_recorder.add_val('Metrics/cluster_precision', cluster_precision)
		loss_recorder.add_val('Metrics/cluster_recall', cluster_recall)
		loss_recorder.add_val('Metrics/cluster_F_score', cluster_F_score)
		loss_recorder.add_val('Metrics/hausdorff', hausdorff)

	#record_images(model, samples, loss_recorder)
	val_loss = loss_recorder.get_mean('Losses/val', to_clear=False)
	tbar.set_description('Epoch {0:>5}, val loss = {1:<.3}'.format(e, val_loss))

	loss_recorder.update(epoch=e)

	if (val_loss < best_val_loss) or (e % 1 == 0):
		model.save_parameters((WEIGHTS_PREFIX + '-{0}-{1:.3}.params').format(e, val_loss))
		best_val_loss = min(best_val_loss, val_loss)
