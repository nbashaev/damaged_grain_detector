from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import mxnet as mx
from models import GAN
from configs import *
import cv2
from utils import *


weights_name = os.path.join(ARCHIVE_FOLDER, 'unet-GAN-14-0.629.params')

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

net = GAN()
net.hybridize()
net.load_parameters(weights_name, ctx=ctx)

w = int(1.5 * WIDTH)
h = int(1.5 * HEIGHT)


class ExampleHandler(FileSystemEventHandler):
	def on_created(self, event):
		path = event.src_path

		if path.split('_')[-1] != 'led.bmp':
			return

		pic = cv2.imread(path, cv2.IMREAD_COLOR)
		prediction = cv2_to_ndarr(cv2.resize(pic, (WIDTH, HEIGHT)), ctx=ctx)
		prediction = net(prediction.expand_dims(axis=0))[0][0]

		prediction = ndarr_to_cv2(prediction.softmax(axis=0).argmax(axis=0))
		prediction = del_small_components(prediction, MIN_COMPONENT_SIZE)

		pic = cv2.resize(pic, (w, h))
		prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)

		blend3 = draw_mask(pic, [None, None, prediction], 1 / 3)
		pic = (pic * (2 / 3)).astype(np.uint8)
		strip = np.zeros((20, w, 3), dtype=np.uint8)
		output_img = np.vstack((pic, strip, blend3))

		cv2.imshow('output_img', output_img)
		cv2.moveWindow('output_img', 0, 0)

		cv2.waitKey(2500)
		cv2.destroyAllWindows()



observer = Observer()
event_handler = ExampleHandler()
observer.schedule(event_handler, path='test_folder/')
observer.start()

try:
	while True:
		time.sleep(0.1)
except KeyboardInterrupt:
	observer.stop()

observer.join()
