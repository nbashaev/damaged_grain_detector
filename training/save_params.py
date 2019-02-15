import mxnet as mx
from models import *
from configs import *
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np


input_shape = (1, 3, HEIGHT, WIDTH)

GAN_params_path = os.path.join(ARCHIVE_FOLDER, 'unet-GAN-14-0.629.params')
params_folder = os.path.join(ARCHIVE_FOLDER, 'unet-14-0.629')
tmp = os.path.join(params_folder, 'unet')
os.mkdir(params_folder)


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

model = GAN()
model.hybridize()
model.load_parameters(GAN_params_path, ctx=ctx, allow_missing=True, ignore_extra=True)
model.generator.forward(mx.nd.zeros(input_shape, ctx=ctx))

model.generator.export(tmp)
onnx_mxnet.export_model(tmp + '-symbol.json', tmp + '-0000.params', [input_shape], np.float32, tmp + '.onnx')
