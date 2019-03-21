import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


class ContractingBlock(gluon.HybridBlock):
	def __init__(self, channels, **kwargs):
		super(ContractingBlock, self).__init__(**kwargs)

		with self.name_scope():
			self.seq = nn.HybridSequential()
			self.seq.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1))
			#self.seq.add(nn.BatchNorm(center=True, scale=True))
			self.seq.add(nn.Activation('relu'))
			self.seq.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1))
			self.seq.add(nn.BatchNorm(center=True, scale=True))
			self.seq.add(nn.Activation('relu'))

			self.pool = nn.MaxPool2D(pool_size=2, strides=2)

	def hybrid_forward(self, F, x, *args, **kwargs):
		x = self.seq(x)
		return x, self.pool(x)


class ExpandingBlock(gluon.HybridBlock):
	def __init__(self, channels, **kwargs):
		super(ExpandingBlock, self).__init__(**kwargs)

		with self.name_scope():
			self.seq = nn.HybridSequential()
			self.seq.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1))
			#self.seq.add(nn.BatchNorm(center=True, scale=True))
			self.seq.add(nn.Activation('relu'))
			self.seq.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1))
			#self.seq.add(nn.BatchNorm(center=True, scale=True))
			self.seq.add(nn.Activation('relu'))
			self.seq.add(nn.Conv2DTranspose(channels=channels//2, kernel_size=2, strides=2))
			self.seq.add(nn.BatchNorm(center=True, scale=True))
			self.seq.add(nn.Activation('relu'))

	def hybrid_forward(self, F, x, *args, **kwargs):
		return self.seq(x)


class EndingBlock(gluon.HybridBlock):
	def __init__(self, channels, num_of_classes=2, **kwargs):
		super(EndingBlock, self).__init__(**kwargs)

		with self.name_scope():
			self.seq = nn.HybridSequential()
			self.seq.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1))
			self.seq.add(nn.BatchNorm(center=True, scale=True))
			self.seq.add(nn.Activation('relu'))
			self.seq.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1))
			self.seq.add(nn.BatchNorm(center=True, scale=True))
			self.seq.add(nn.Activation('relu'))
			self.seq.add(nn.Conv2D(channels=num_of_classes, kernel_size=1))

	def hybrid_forward(self, F, x, *args, **kwargs):
		return self.seq(x)


class UNet(gluon.HybridBlock):
	def __init__(self, **kwargs):
		super(UNet, self).__init__(**kwargs)

		with self.name_scope():
			self.contracting_block_1 = ContractingBlock(2 ** 6)
			self.contracting_block_2 = ContractingBlock(2 ** 7)
			self.contracting_block_3 = ContractingBlock(2 ** 8)
			self.contracting_block_4 = ContractingBlock(2 ** 9)
			
			self.dropout_layer = nn.Dropout(.3)

			self.expanding_block_1 = ExpandingBlock(2 ** 10)
			self.expanding_block_2 = ExpandingBlock(2 ** 9)
			self.expanding_block_3 = ExpandingBlock(2 ** 8)
			self.expanding_block_4 = ExpandingBlock(2 ** 7)
			
			self.ending_block = EndingBlock(2 ** 6)

	def hybrid_forward(self, F, x, *args, **kwargs):
		output_1, x = self.contracting_block_1(x)
		output_2, x = self.contracting_block_2(x)
		output_3, x = self.contracting_block_3(x)
		output_4, x = self.contracting_block_4(x)
		
		x = self.dropout_layer(x)
		
		x = F.concat(output_4, self.expanding_block_1(x))
		x = F.concat(output_3, self.expanding_block_2(x))
		x = F.concat(output_2, self.expanding_block_3(x))
		x = F.concat(output_1, self.expanding_block_4(x))

		return self.ending_block(x)


class UNet2(gluon.HybridBlock):
	def __init__(self, **kwargs):
		super(UNet2, self).__init__(**kwargs)

		with self.name_scope():
			self.contracting_block_1 = ContractingBlock(2 ** 6)
			self.contracting_block_2 = ContractingBlock(2 ** 7)
			self.contracting_block_3 = ContractingBlock(2 ** 8)

			self.dropout_layer = nn.Dropout(.3)

			self.expanding_block_1 = ExpandingBlock(2 ** 9)
			self.expanding_block_2 = ExpandingBlock(2 ** 8)
			self.expanding_block_3 = ExpandingBlock(2 ** 7)

			self.ending_block = EndingBlock(2 ** 6)

	def hybrid_forward(self, F, x, *args, **kwargs):
		output_1, x = self.contracting_block_1(x)
		output_2, x = self.contracting_block_2(x)
		output_3, x = self.contracting_block_3(x)

		x = self.dropout_layer(x)

		x = F.concat(output_3, self.expanding_block_1(x))
		x = F.concat(output_2, self.expanding_block_2(x))
		x = F.concat(output_1, self.expanding_block_3(x))

		return self.ending_block(x)


class UNet3(gluon.HybridBlock):
	def __init__(self, **kwargs):
		super(UNet3, self).__init__(**kwargs)

		with self.name_scope():
			self.contracting_block_1 = ContractingBlock(2 ** 5)
			self.contracting_block_2 = ContractingBlock(2 ** 6)
			self.contracting_block_3 = ContractingBlock(2 ** 7)

			self.dropout_layer = nn.Dropout(.3)

			self.expanding_block_1 = ExpandingBlock(2 ** 8)
			self.expanding_block_2 = ExpandingBlock(2 ** 7)
			self.expanding_block_3 = ExpandingBlock(2 ** 6)

			self.ending_block = EndingBlock(2 ** 5)

	def hybrid_forward(self, F, x, *args, **kwargs):
		output_1, x = self.contracting_block_1(x)
		output_2, x = self.contracting_block_2(x)
		output_3, x = self.contracting_block_3(x)

		x = self.dropout_layer(x)

		x = F.concat(output_3, self.expanding_block_1(x))
		x = F.concat(output_2, self.expanding_block_2(x))
		x = F.concat(output_1, self.expanding_block_3(x))

		return self.ending_block(x)


class FCN(gluon.HybridBlock):
	def __init__(self, channels, **kwargs):
		super(FCN, self).__init__(**kwargs)

		with self.name_scope():
			self._conv1 = nn.Conv2D(channels=channels, kernel_size=2, strides=2)
			self._act1 = nn.LeakyReLU(alpha=0.2)
			self._conv2 = nn.Conv2D(channels=2*channels, kernel_size=2, strides=2)
			self._batch1 = nn.BatchNorm(center=True, scale=True)
			self._act2 = nn.LeakyReLU(alpha=0.2)
			self._conv3 = nn.Conv2D(channels=4*channels, kernel_size=2, strides=2)
			self._act3 = nn.LeakyReLU(alpha=0.2)
			self._conv4 = nn.Conv2D(channels=8*channels, kernel_size=2, strides=2)
			self._batch2 = nn.BatchNorm(center=True, scale=True)
			self._act4 = nn.LeakyReLU(alpha=0.2)

			self._conv5 = nn.Conv2D(channels=4*channels, kernel_size=3, strides=1, padding=1)
			self._act5 = nn.LeakyReLU(alpha=0.2)
			self._conv6 = nn.Conv2D(channels=2*channels, kernel_size=3, strides=1, padding=1)
			self._batch3 = nn.BatchNorm(center=True, scale=True)
			self._act6 = nn.LeakyReLU(alpha=0.2)
			self._conv7 = nn.Conv2D(channels=channels, kernel_size=3, strides=1, padding=1)
			self._act7 = nn.LeakyReLU(alpha=0.2)
			self._conv8 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1)
			self._batch4 = nn.BatchNorm(center=True, scale=True)
			self._act8 = nn.LeakyReLU(alpha=0.2)

	def hybrid_forward(self, F, x, *args, **kwargs):
		x1 = self._act1(self._conv1(x))
		x2 = self._act2(self._batch1(self._conv2(x1)))
		x3 = self._act3(self._conv3(x2))
		x4 = self._act4(self._batch2(self._conv4(x3)))

		y3 = F.UpSampling(x4, scale=2, sample_type='nearest')
		y3 = F.concat(x3, y3)
		y3 = self._act5(self._conv5(y3))

		y2 = F.UpSampling(y3, scale=2, sample_type='nearest')
		y2 = F.concat(x2, y2)
		y2 = self._act6(self._batch3(self._conv6(y2)))

		y1 = F.UpSampling(y2, scale=2, sample_type='nearest')
		y1 = F.concat(x1, y1)
		y1 = self._act7(self._conv7(y1))

		y = F.UpSampling(y1, scale=2, sample_type='nearest')
		y = F.concat(x, y)
		y = self._act8(self._batch4(self._conv8(y)))

		return y


class GAN(gluon.HybridBlock):
	def __init__(self, **kwargs):
		super(GAN, self).__init__(**kwargs)

		with self.name_scope():
			self.generator = UNet2()
			self.discriminator = UNet3() # FCN(2 ** 7)

	def hybrid_forward(self, F, x, *args, **kwargs):
		predicted_mask = self.generator(x)
		x = F.softmax(predicted_mask, axis=1)
		confidence_map = self.discriminator(x)

		return predicted_mask, confidence_map


class GAN2(gluon.HybridBlock):
	def __init__(self, **kwargs):
		super(GAN2, self).__init__(**kwargs)

		with self.name_scope():
			self.generator = UNet3()
			self.discriminator = UNet3()

	def hybrid_forward(self, F, x, *args, **kwargs):
		predicted_mask = self.generator(x)
		x = F.softmax(predicted_mask, axis=1)
		confidence_map = self.discriminator(x)

		return predicted_mask, confidence_map



class ModelWrapper():
	def __init__(self, Model, weights, ctx, preprocess, postprocess):
		self.ctx = ctx
		self.preprocess = preprocess
		self.postprocess = postprocess

		self.net = Model()
		self.net.hybridize()

		if weights is None:
			self.net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)
		else:
			self.net.load_parameters(weights, ctx=self.ctx)

	def execute(self, data):
		data = self.preprocess(data, ctx=self.ctx)
		res = self.net(data)
		res = self.postprocess(res, ctx=self.ctx)

		return res
