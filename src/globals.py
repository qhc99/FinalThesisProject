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

GPU_DEVICE = select_device('')
YOLO_MODEL = load_model(YOLOV5S_PATH, GPU_DEVICE)
MODEL_OUTPUT_NAMES = get_names(YOLO_MODEL)
MODEL_OUTPUT_COLOR = get_colors(MODEL_OUTPUT_NAMES)
INTRESTED_CLASSES = {0, 1, 2, 3, 5, 7, 11}
# BGR
for _cls in INTRESTED_CLASSES:
    if _cls == 0:
        MODEL_OUTPUT_COLOR[_cls] = [0, 255, 0]  # green person
    elif _cls == 11:
        MODEL_OUTPUT_COLOR[_cls] = [255, 153, 0]  # sign blue
    else:
        MODEL_OUTPUT_COLOR[_cls] = [0, 0, 255]   # cars red

SIGN_CLASSIFIER = cv2.CascadeClassifier(CASCADE_FILE_PATH)