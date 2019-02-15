import tensorrt as trt
import os


folder = os.path.join('.') # 'archive', 'unet-14-0.629'
onnx_path = os.path.join(folder, 'unet.onnx')
engine_path = os.path.join(folder, 'unet.engine')

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def GiB(val):
	return val * 1 << 30


def build_engine(path):
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
		builder.max_batch_size = 1
		builder.max_workspace_size = GiB(1)

		with open(path, 'rb') as model:
			parser.parse(model.read())

		return builder.build_cuda_engine(network)


if __name__ == '__main__':
	with build_engine(onnx_path) as engine:
		with open(engine_path, 'wb') as f:
			f.write(engine.serialize())
