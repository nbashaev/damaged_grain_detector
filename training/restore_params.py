import mxnet as mx
from models import *
from configs import *
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np


params_folder = os.path.join(ARCHIVE_FOLDER, 'unet-14-0.629')
tmp = os.path.join(params_folder, 'unet')
onnx_model_file = tmp + '.onnx'
out_file = os.path.join(params_folder, 'restored.params')

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

model = onnx_mxnet.import_to_gluon(onnx_model_file, ctx)
model.save_parameters(out_file)

'''

model = GAN()
model.hybridize()
model.load_parameters(GAN_params_path, ctx=ctx, allow_missing=True, ignore_extra=True)
model.forward(mx.nd.zeros(input_shape, ctx=ctx))

model.export(tmp)
onnx_mxnet.export_model(tmp + '-symbol.json', tmp + '-0000.params', [input_shape], np.float32, tmp + '.onnx')
'''
