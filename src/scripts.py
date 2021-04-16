import cv2
import time
import os
import numpy as np
from pathlib import Path
from enum import Enum
from PIL import Image
import torch.backends.cudnn as cudnn
from multiprocessing import Process, Queue

from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors, img_resize, img_transform
from utils.general import non_max_suppression
from utils.plots import plot_one_box
from utils.general import scale_coords

NEG_IMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/neg_imgs/imgs"
POS_IMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/pos_imgs/img"
POS_HARDIMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/pos_imgs/hard_imgs"

MODEL_NUM = 3

# 1: 639
# 3: 381
CASCADE_FILE_PATH = "../../dataset/TrafficBlockSign/models/model" + str(MODEL_NUM) + "/cascade.xml"

YOLOV5L_PATH = "./parameters/original/yolov5l.pt"
YOLOV5M_PATH = "./parameters/original/yolov5m.pt"
YOLOV5S_PATH = "./parameters/original/yolov5s.pt"

CONFI_THRES = 0.25
IOU_THRES = 0.45

SIGN_CLASSIFIER = cv2.CascadeClassifier(CASCADE_FILE_PATH)

GPU_DEVICE = select_device('')
YOLO_MODEL = load_model(YOLOV5S_PATH, GPU_DEVICE)
MODEL_OUTPUT_NAMES = get_names(YOLO_MODEL)
MODEL_OUTPUT_COLOR = get_colors(MODEL_OUTPUT_NAMES)
cudnn.benchmark = True

FONT = cv2.FONT_HERSHEY_SIMPLEX

INTRESTED_CLASSES = {0, 1, 2, 3, 5, 7, 11}

# BGR
for _cls in INTRESTED_CLASSES:
    if _cls == 0:
        MODEL_OUTPUT_COLOR[_cls] = [0, 255, 0]  # green person
    elif _cls == 11:
        MODEL_OUTPUT_COLOR[_cls] = [255, 153, 0]  # sign blue
    else:
        MODEL_OUTPUT_COLOR[_cls] = [0, 0, 255]  # cars red


def yoloPredictionPaint(pred, tensor_img, origin_img, path_img='', img_window=False, webcam=False):
    # Process detections
    painted = False
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0 = path_img[i], '%g: ' % i, origin_img[i].copy()
        else:
            p, s, im0 = path_img, '', origin_img
        p = Path(p)  # to Path
        s += '%gx%g ' % tensor_img.shape[2:]  # print string
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(tensor_img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for (*xyxy, conf, cls) in reversed(det):
                if int(cls.item()) in INTRESTED_CLASSES:
                    painted = True
                    label = f'{MODEL_OUTPUT_NAMES[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=MODEL_OUTPUT_COLOR[int(cls)], line_thickness=2)

            if img_window:
                if not webcam:
                    cv2.imshow(str(p), im0)
                    k = cv2.waitKey(0) & 0xFF  # standard grammar for 64-bit machine
                    if k == 27:  # enter ESC to close window
                        cv2.destroyAllWindows()
                else:
                    raise Exception("not implement.")
    return painted


def signPredictionPaint(img, sign_pred):
    if len(sign_pred) > 0:
        label = "prohibit"
        for (x, y, w, h) in sign_pred:
            xyxy = [x, y, x + w, y + h]
            plot_one_box(xyxy, img, label=label, color=[255, 153, 0], line_thickness=2)


class ImgsSource(Enum):
    CAMERA = 0
    FILE = 1
    VIDEO = 2


def RunModels(SOURCE=ImgsSource.CAMERA, IMG_FOLDER_PATH=None):
    if (SOURCE == ImgsSource.FILE or SOURCE == ImgsSource.VIDEO) and (IMG_FOLDER_PATH is None):
        raise Exception("path is None")

    yolo_in_queue = Queue()
    yolo_out_queue = Queue()
    yolo_process = Process(target=yoloPredict, args=(yolo_in_queue, yolo_out_queue))
    yolo_process.start()

    sign_in_queue = Queue()
    sign_out_queue = Queue()
    sign_process = Process(target=signPredict, args=(sign_in_queue, sign_out_queue))
    sign_process.start()

    if SOURCE == ImgsSource.CAMERA:
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        cv2.namedWindow("camera")
        cv2.moveWindow('camera', 300, 115)
        last_time = time.time()

        while cap.isOpened():
            read_succ, img = cap.read()
            if not read_succ:
                break

            pil_img = cv2_to_pil(img)

            yolo_in_queue.put(pil_img, True)
            sign_in_queue.put(pil_img, True)

            yolo_painted = yolo_out_queue.get(True)
            yolo_painted = pil_to_cv2(yolo_painted)
            sign_pred = sign_out_queue.get(True)

            signPredictionPaint(yolo_painted, sign_pred)
            res_img = yolo_painted

            current_latency = (time.time() - last_time) * 1000
            last_time = time.time()
            cv2.putText(res_img, "FPS:%.1f" % (1000 / current_latency), (0, 20), FONT, 0.5, (255, 80, 80), 1,
                        cv2.LINE_4)

            cv2.imshow('camera', res_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elif SOURCE == ImgsSource.FILE:
        imgs_folder_path = os.path.join(os.getcwd(), IMG_FOLDER_PATH)
        img_names_list = os.listdir(imgs_folder_path)

        for img_name in img_names_list:
            img_path = os.path.join(imgs_folder_path, img_name)
            img = cv2.imread(img_path)

            pil_img = cv2_to_pil(img)

            yolo_in_queue.put(pil_img, True)
            sign_in_queue.put(pil_img, True)

            yolo_painted = yolo_out_queue.get(True)
            yolo_painted = pil_to_cv2(yolo_painted)
            sign_pred = sign_out_queue.get(True)

            signPredictionPaint(yolo_painted, sign_pred)
            res_img = yolo_painted

            cv2.imshow('camera', res_img)
            if cv2.waitKey(3000) & 0xFF == ord('q'):
                break

    elif SOURCE == ImgsSource.VIDEO:
        raise Exception("not implemented")

    else:
        raise Exception("unknown img source")


def yoloPredict(in_queue: Queue, out_queue: Queue):
    while True:
        img = in_queue.get(True)
        img = pil_to_cv2(img)
        tensor_img = img_transform(img_resize(img, 640), GPU_DEVICE)
        yolo_pred = YOLO_MODEL(tensor_img)[0]
        yolo_pred = non_max_suppression(yolo_pred, CONFI_THRES, IOU_THRES)
        yoloPredictionPaint(yolo_pred, tensor_img, img)
        img = cv2_to_pil(img)
        out_queue.put(img, True)


def signPredict(in_queue: Queue, out_queue: Queue):
    while True:
        img = in_queue.get(True)
        img = pil_to_cv2(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sign_detect = SIGN_CLASSIFIER.detectMultiScale(gray, 1.1, 1)
        out_queue.put(sign_detect, True)


def cv2_to_pil(img):  # Since you want to be able to use Pillow (PIL)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    RunModels(SOURCE=ImgsSource.FILE, IMG_FOLDER_PATH="../../dataset/TrafficBlockSign/pos_imgs/img")
    # RunModels()
