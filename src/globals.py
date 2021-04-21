from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors
import cv2


MODEL_NUM = "1"

# 1: 381
OVERTRAINED_SIGN_MODEL_PATH = "./parameters/trained/lbp_overtrained_no_break.xml"
LBP_33000 = "./parameters/trained/lbp_24_33000_break.xml"
LBP_EXP = "./parameters/trained/lbp_fix_pos_12000.xml"

YOLOV5L_PATH = "./parameters/original/yolov5l.pt"
YOLOV5M_PATH = "./parameters/original/yolov5m.pt"
YOLOV5S_PATH = "./parameters/original/yolov5s.pt"

GPU_DEVICE = select_device('')
TRAFFIC_MODEL = load_model(YOLOV5S_PATH, GPU_DEVICE)

TRAFFIC_NAMES = get_names(TRAFFIC_MODEL)
TRAFFIC_NAMES[0] = "行人"
TRAFFIC_NAMES[1] = "自行车"
TRAFFIC_NAMES[2] = "汽车"
TRAFFIC_NAMES[3] = "摩托车"
TRAFFIC_NAMES[5] = "公交车"
TRAFFIC_NAMES[7] = "卡车"
TRAFFIC_NAMES[11] = "禁止标志"

TRAFFIC_COLOR = get_colors(TRAFFIC_NAMES)

INTRESTED_CLASSES = {0, 1, 2, 3, 5, 7, 11}
# BGR
for _cls in INTRESTED_CLASSES:
    if _cls == 0:
        TRAFFIC_COLOR[_cls] = [0, 255, 0]  # person green
    elif _cls == 11:
        TRAFFIC_COLOR[_cls] = [255, 153, 0]  # sign blue
    else:
        TRAFFIC_COLOR[_cls] = [0, 0, 255]   # cars red

SIGN_CLASSIFIER = cv2.CascadeClassifier()
SIGN_CLASSIFIER.load(OVERTRAINED_SIGN_MODEL_PATH)
