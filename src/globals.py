from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors
import cv2


NEG_IMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/neg_imgs/imgs"

CONFI_THRES = 0.25
IOU_THRES = 0.45

FONT = cv2.FONT_HERSHEY_SIMPLEX


MODEL_NUM = "1"

# 1: 381
CASCADE = "./parameters/trained/lbp.xml"
YOLOV5S_PATH = "./parameters/original/yolov5s.pt"

GPU_DEVICE = select_device('')
TRAFFIC_MODEL = load_model(YOLOV5S_PATH, GPU_DEVICE)
TRAFFIC_MODEL.eval()

TRAFFIC_NAMES = get_names(TRAFFIC_MODEL)
TRAFFIC_NAMES[11] = "prohibit"

TRAFFIC_COLOR = get_colors(TRAFFIC_NAMES)

INTERESTED_CLASSES = {0, 1, 2, 3, 5, 7, 11}  # 11
# BGR
for _cls in INTERESTED_CLASSES:
    if _cls == 0:
        TRAFFIC_COLOR[_cls] = [0, 255, 0]  # person green
    elif _cls == 11:
        TRAFFIC_COLOR[_cls] = [255, 153, 0]  # sign blue
    else:
        TRAFFIC_COLOR[_cls] = [0, 0, 255]   # cars red

SIGN_CLASSIFIER = cv2.CascadeClassifier(CASCADE)
