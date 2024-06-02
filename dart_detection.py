
import threading
import time
from ctypes import c_int, c_float
import cv2
from integrations.darknet_integration import darknet
from autoscorer import ImageUtils, BoardCalculatorHelper, slope , params

class SimpleDarknetThread(threading.Thread):
    def __init__(self, darknet, image_path):
        super().__init__()
        self.darknet = darknet
        self.image_path = image_path
        self.output = None

    def run(self):
        start_time = time.time()
        self.output = self.darknet.process(self.image_path)
        print(f"Required time: {(time.time() - start_time):.2f} s")


class Darknet:
    def __init__(self):
      cfg_file = "object-detection/yolov3.cfg"
      names_file = "object-detection/obj.data"
      weights_file = "object-detection/backup/yolov3_last.weights"

      # First thing we do is load the neural network.
      self.network, self.class_names, self.colours, self.metadata = darknet.load_network(cfg_file, names_file, weights_file)
      self.width = darknet.network_width(self.network)
      self.height = darknet.network_height(self.network)

      self.prediction_threshold = 0.25


    def process(self, image_name):
        image_bgr = cv2.imread(image_name)
        darknet_image = darknet.convert_cv2_image2darknet_image(image_bgr)
        resized_image = darknet.resize_image(darknet_image, self.width, self.height)

        detections = darknet.detect_image(self.network, self.class_names, resized_image,
                                          darknet.ImageDimension(width=darknet_image.w, height=darknet_image.h),
                                          thresh=self.prediction_threshold)

        darknet.free_image(darknet_image)
        darknet.free_image(resized_image)

        return detections




if __name__ == "main":
    image_path = "D:/dart/images_test_cases/b_SING_BOARD_20240325_092516_452t_SING_BOARD_20240325_092711_842/throw/0.jpg"
    darknet_instance_1 = SimpleDarknetThread(Darknet(), image_path)
    darknet_instance_2 = SimpleDarknetThread(Darknet(), image_path)

    darknet_instance_3 = SimpleDarknetThread(Darknet(), image_path)
    darknet_instance_4 = SimpleDarknetThread(Darknet(), image_path)


    darknet_instance_1.start()
    darknet_instance_2.start()

    darknet_instance_3.start()
    darknet_instance_4.start()

    darknet_instance_1.join()
    darknet_instance_2.join()

    darknet_instance_3.join()
    darknet_instance_4.join()

    print(f"Darknet 1 output: {darknet_instance_1.output}")
    print(f"Darknet 2 output: {darknet_instance_2.output}")
    print(f"Darknet 3 output: {darknet_instance_3.output}")
    print(f"Darknet 4 output: {darknet_instance_4.output}")
        