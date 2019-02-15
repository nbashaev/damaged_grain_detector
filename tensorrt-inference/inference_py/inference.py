import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import cv2

WIDTH = (192 * 3) // 16 * 16
HEIGHT = (108 * 3) // 16 * 16
MIN_COMPONENT_SIZE = 0.0001

MODEL_NAME = '../unet.engine'
IMAGE_NAME = '../input/test.jpg'


class HostDeviceMem(object):
	def __init__(self, host_mem, device_mem):
		self.host = host_mem
		self.device = device_mem

	def __str__(self):
		return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

	def __repr__(self):
		return self.__str__()


def allocate_buffers(engine):
	inputs = []
	outputs = []
	bindings = []
	stream = cuda.Stream()
	for binding in engine:
		size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
		dtype = trt.nptype(engine.get_binding_dtype(binding))
		# Allocate host and device buffers
		host_mem = cuda.pagelocked_empty(size, dtype)
		device_mem = cuda.mem_alloc(host_mem.nbytes)
		# Append the device buffer to device bindings.
		bindings.append(int(device_mem))
		# Append to the appropriate list.
		if engine.binding_is_input(binding):
			inputs.append(HostDeviceMem(host_mem, device_mem))
		else:
			outputs.append(HostDeviceMem(host_mem, device_mem))
	return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
	# Transfer input data to the GPU.
	[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
	# Run inference.
	context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
	# Transfer predictions back from the GPU.
	[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
	# Synchronize the stream
	stream.synchronize()
	# Return only the host outputs.
	return [out.host for out in outputs]


def load_image(img_name, my_buffer):
	def _(image):
		image = np.transpose(image, (2, 0, 1))
		image = image / 255.0

		return image
	
	img = cv2.imread(img_name, cv2.IMREAD_COLOR)
	img = cv2.resize(img, (WIDTH, HEIGHT))
	
	np.copyto(my_buffer, _(img).ravel())
	return img


def extract_results(my_buffer):
	def _(mask, ratio, connectivity=4):
		if mask.ndim == 3:
			for i in range(mask.shape[0]):
				mask[i] = _(mask[i], ratio, connectivity)

			return mask

		num_of_components, labels, stats = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)[0:3]

		for k in range(num_of_components):
			if stats[k][4] < ratio * WIDTH * HEIGHT:
				component_mask = (labels != k).astype(np.uint8)
				mask = cv2.bitwise_and(mask, mask, mask=component_mask)
		
		return mask
	
	img = np.reshape(my_buffer, (2, HEIGHT, WIDTH))
	img = (255 * img.argmax(axis=0)).astype(np.uint8)
	return _(img, MIN_COMPONENT_SIZE)


def main():
	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

	with open(MODEL_NAME, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
		engine = runtime.deserialize_cuda_engine(f.read())
		inputs, outputs, bindings, stream = allocate_buffers(engine)

		with engine.create_execution_context() as context:
			img = load_image(IMAGE_NAME, inputs[0].host)
			[output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
			mask = extract_results(output)
	
	cv2.imshow('img', img)
	cv2.imshow('mask', mask)
	cv2.waitKey(0)


if __name__ == '__main__':
	main()
