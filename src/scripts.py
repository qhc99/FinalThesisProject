import cv2
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors, img_resize, img_transform
from utils.general import non_max_suppression
import os
from utils.plots import plot_one_box
from utils.general import scale_coords
from pathlib import Path
import time
from enum import Enum

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
# 0 1 2 3 5 7
MODEL_OUTPUT_NAMES = get_names(YOLO_MODEL)
MODEL_OUTPUT_COLOR = get_colors(MODEL_OUTPUT_NAMES)
cudnn.benchmark = True

FONT = cv2.FONT_HERSHEY_SIMPLEX


def yolo_paint_interested_result(pred, tensor_img, origin_img, path_img='', img_window=False, webcam=False):
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
                if int(cls.item()) in {0, 1, 2, 3, 5, 7, 11}:
                    painted = True
                    label = f'{MODEL_OUTPUT_NAMES[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=MODEL_OUTPUT_COLOR[int(cls)], line_thickness=3)

            if img_window:
                if not webcam:
                    cv2.imshow(str(p), im0)
                    k = cv2.waitKey(0) & 0xFF  # standard grammar for 64-bit machine
                    if k == 27:  # enter ESC to close window
                        cv2.destroyAllWindows()
                else:
                    raise Exception("not implement.")
    return painted


class ImgsSource(Enum):
    CAMERA = 0
    FILE = 1
    VIDEO = 2


def RunModels(SOURCE=ImgsSource.CAMERA, IMG_FOLDER_PATH=None, SHOW_FPS=False):
    if (SOURCE == ImgsSource.FILE or SOURCE == ImgsSource.VIDEO) and (IMG_FOLDER_PATH is None):
        raise Exception("path is None")

    if SOURCE == ImgsSource.CAMERA:
        cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
        sign_pred = None
        yolo_pred, yolo_tensor_img = None, None

        # warm up
        if cap.isOpened():
            read_succ, img = cap.read()
            sign_pred = signPredict(img)
            yolo_pred, yolo_tensor_img = yoloPredict(img)

        # process trick
        process_yolo = True

        cv2.namedWindow("camera")
        cv2.moveWindow('camera', 300, 115)

        # FPS init
        frame_count = -1
        last_time = time.time()
        latency = 0

        while cap.isOpened():
            read_succ, img = cap.read()
            if not read_succ:
                break

            if process_yolo:
                process_yolo = not process_yolo
                yolo_pred, yolo_tensor_img = yoloPredict(img)
            else:
                process_yolo = not process_yolo
                sign_pred = signPredict(img)

            paint(img, sign_pred, yolo_pred, yolo_tensor_img)

            if SHOW_FPS:
                frame_count += 1
                current_latency = (time.time() - last_time) * 1000
                last_time = time.time()
                if frame_count == 1:
                    latency = current_latency
                elif frame_count > 1:
                    latency = ((frame_count - 1) * latency + current_latency) / frame_count

                if latency > 0:
                    cv2.putText(img, "FPS:%.1f" % (1000 / latency), (0, 20), FONT, 0.5, (255, 80, 80), 1, cv2.LINE_4)

            cv2.imshow('camera', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elif SOURCE == ImgsSource.FILE:
        imgs_folder_path = os.path.join(os.getcwd(), IMG_FOLDER_PATH)
        img_names_list = os.listdir(imgs_folder_path)
        for img_name in img_names_list:
            img_path = os.path.join(imgs_folder_path, img_name)
            img = cv2.imread(img_path)

            sign_pred = signPredict(img)
            yolo_pred, yolo_tensor_img = yoloPredict(img)
            paint(img, sign_pred, yolo_pred, yolo_tensor_img)

            cv2.imshow('camera', img)
            if cv2.waitKey(3000) & 0xFF == ord('q'):
                break

    elif SOURCE == ImgsSource.VIDEO:
        raise Exception("not implemented")

    else:
        raise Exception("unknown img source")


def yoloPredict(img):
    tensor_img = img_transform(img_resize(img, 640), GPU_DEVICE)
    yolo_pred = YOLO_MODEL(tensor_img)[0]
    yolo_pred = non_max_suppression(yolo_pred, CONFI_THRES, IOU_THRES)
    return yolo_pred, tensor_img


def signPredict(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sign_detect = SIGN_CLASSIFIER.detectMultiScale(gray, 1.1, 1)
    return sign_detect


def paint(img, sign_detect, yolo_pred, yolo_tensor_img):
    sign_painted = False
    # paint cascade
    if len(sign_detect) > 0:
        sign_painted = True
        for (x, y, w, h) in sign_detect:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # paint yolo
    yolo_painted = yolo_paint_interested_result(yolo_pred, yolo_tensor_img, img)

    return yolo_painted, sign_painted


def videoWithoutPredict():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, img = cap.read()
        cv2.imshow('camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    RunModels()
