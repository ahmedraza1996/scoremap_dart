import logging
import threading

from .config import YOLOV3, OBJ_DATA, WEIGHTS
from integrations.darknet_integration import darknet



class Darknet: 
  def __init__(self):
      cfg_file = YOLOV3
      names_file = OBJ_DATA
      weights_file = WEIGHTS

      # First thing we do is load the neural network.
      self.network, self.class_names, self.colours, self.metadata = darknet.load_network(cfg_file, names_file,
                                                                                         weights_file)
      self.width = darknet.network_width(self.network)
      self.height = darknet.network_height(self.network)

      self.prediction_threshold = 0.25

      self.available = True

  def process(self, image):
      darknet_image = darknet.convert_cv2_image2darknet_image(image)
      resized_image = darknet.resize_image(darknet_image, self.width, self.height)

      detections = darknet.detect_image(self.network, self.class_names, resized_image,
                                        darknet.ImageDimension(width=darknet_image.w, height=darknet_image.h),
                                        thresh=self.prediction_threshold)

      darknet.free_image(darknet_image)
      darknet.free_image(resized_image)

      return detections