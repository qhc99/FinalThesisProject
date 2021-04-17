from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors
import cv2


MODEL_NUM = 3

# 1: 639
# 3: 381
CASCADE_FILE_PATH = "../../dataset/TrafficBlockSign/models/model" + str(MODEL_NUM) + "/cascade.xml"

YOLOV5L_PATH = "./parameters/original/yolov5l.pt"
YOLOV5M_PATH = "./parameters/original/yolov5m.pt"
YOLOV5S_PATH = "./parameters/original/yolov5s.pt"
YOLO_SIGN_PATH = "./parameters/trained/best1.pt"

GPU_DEVICE = select_device('')
TRAFFIC_MODEL = load_model(YOLOV5S_PATH, GPU_DEVICE)
SIGN_MODEL = load_model(YOLO_SIGN_PATH, GPU_DEVICE)

TRAFFIC_NAMES = get_names(TRAFFIC_MODEL)
SIGN_NAMES = get_names(SIGN_MODEL)

TRAFFIC_COLOR = get_colors(TRAFFIC_NAMES)
SIGN_COLOR = [255, 153, 0]

INTRESTED_CLASSES = {0, 1, 2, 3, 5, 7}
# BGR
for _cls in INTRESTED_CLASSES:
    if _cls == 0:
        TRAFFIC_COLOR[_cls] = [0, 255, 0]  # green person
    elif _cls == 11:
        TRAFFIC_COLOR[_cls] = [255, 153, 0]  # sign blue
    else:
        TRAFFIC_COLOR[_cls] = [0, 0, 255]   # cars red

SIGN_CLASSIFIER = cv2.CascadeClassifier(CASCADE_FILE_PATH)
