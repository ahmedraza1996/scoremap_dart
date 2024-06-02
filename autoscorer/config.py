import logging
import os

OBJ_DATA = os.getenv("OBJ_DATA", "/app/object-detection/obj.data")
YOLOV3 = os.getenv("YOLOV3", "/app/object-detection/yolov3.cfg")
WEIGHTS = os.getenv("WEIGHTS", "/app/object-detection/backup/yolov3_last.weights")
# WEIGHTS = "/app/object-detection/backup/yolov3_last_01_03.weights"

logging.basicConfig(
    filename=os.getenv("SERVICE_LOG", "server.log"),
    level=logging.DEBUG,
    format="%(levelname)s: %(asctime)s \
        pid:%(process)s module:%(module)s %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)
