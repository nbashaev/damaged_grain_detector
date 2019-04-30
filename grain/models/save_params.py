import os

from grain.utils import *
from mxnet.contrib import onnx as onnx_mxnet
import shutil


if __name__ == '__main__':
	input_shape = (1, 3, HEIGHT, WIDTH)
	model = current_model(WIDTH, HEIGHT)
	model.net.forward(nd.zeros(input_shape, ctx=model.ctx))

	temp_folder = os.path.join(ARCHIVE_FOLDER, 'temp')
	tmp = os.path.join(temp_folder, CUR_MODEL)
	os.mkdir(temp_folder)

	model.net.export(tmp, epoch=0)
	onnx_mxnet.export_model(tmp + '-symbol.json', tmp + '-0000.params', [input_shape], np.float32, tmp + '.onnx')
	os.rename(tmp + '.onnx', os.path.join(ONNX_FOLDER, CUR_MODEL + '.onnx'))

	shutil.rmtree(temp_folder)
