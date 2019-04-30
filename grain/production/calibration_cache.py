import pycuda.driver as cuda
import numpy as np
from PIL import Image
import ctypes
import tensorrt as trt
import glob
from random import shuffle
from grain.configs import INPUT_FOLDER, ONNX_FOLDER, CUR_MODEL
import os


folder = os.path.join('grain', 'production')
onnx_path = os.path.join(ONNX_FOLDER, CUR_MODEL + '.onnx')
cache_path = os.path.join(folder, 'cache', CUR_MODEL + '.bin')
CALIBRATION_DATASET = os.path.join(INPUT_FOLDER, '*_led.bmp')


class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator):
	def __init__(self, stream):
		trt.IInt8EntropyCalibrator.__init__(self) 
		self.stream = stream
		self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
		stream.reset()

	def get_batch_size(self):
		return self.stream.batch_size

	def get_batch(self, bindings, names):
		batch = self.stream.next_batch()
		if not batch.size:
			return None

		cuda.memcpy_htod(self.d_input, batch)
		bindings[0] = int(self.d_input)

		return bindings

	def read_calibration_cache(self, length):
		return None

	def write_calibration_cache(self, ptr, size):
		ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
		ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
		void_ptr = ctypes.pythonapi.PyCapsule_GetPointer(ptr, None)
		cache = ctypes.c_char_p(void_ptr)

		with open(cache_path, 'wb') as f:
			f.write(cache.value)

		return None


class ImageBatchStream():
	def __init__(self, batch_shape, calibration_files):
		batch_size, C, H, W = batch_shape

		self.batch_size = batch_size
		self.H, self.W = H, W
		self.files = calibration_files

		self.max_batches = ((len(calibration_files) - 1) // batch_size) + 1
		self.calibration_data = np.zeros((batch_size, C, H, W), dtype=np.float32)
		self.batch = 0

	def read_image_chw(self, path):
		img = Image.open(path).resize((self.W, self.H), Image.NEAREST)
		img = np.array(img, dtype=np.float32, order='C')
		img = img[:, :, ::-1]
		img = img.transpose((2, 0, 1))

		return img

	def preprocess(self, img):
		return img / 255.0

	def reset(self):
		self.batch = 0

	def next_batch(self):
		if self.batch >= self.max_batches:
			return np.array([])

		files_for_batch = self.files[self.batch_size * self.batch : self.batch_size * (self.batch + 1)]

		for i, f in enumerate(files_for_batch):
			print("[ImageBatchStream] Processing ", f)
			self.calibration_data[i] = self.preprocess(self.read_image_chw(f))

		self.batch += 1
		return np.ascontiguousarray(self.calibration_data, dtype=np.float32)


def build_engine(path, calibration_files=None):
	def GiB(val):
		return val * 1 << 30

	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

	with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
		with open(path, 'rb') as model:
			parser.parse(model.read())

		C, H, W = network.get_input(0).shape
		batchstream = ImageBatchStream((1, C, H, W), calibration_files)
		int8_calibrator = PythonEntropyCalibrator(batchstream)

		builder.max_batch_size = 1
		builder.max_workspace_size = GiB(1)
		builder.int8_mode = True
		builder.int8_calibrator = int8_calibrator

		return builder.build_cuda_engine(network)


def create_calibration_dataset():
	calibration_files = glob.glob(CALIBRATION_DATASET)
	shuffle(calibration_files)

	return calibration_files[:500]


if __name__ == '__main__':
	build_engine(onnx_path, create_calibration_dataset())
