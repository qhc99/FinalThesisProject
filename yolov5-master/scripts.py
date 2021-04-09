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
from skimage import io
from enum import Enum
from PIL import Image
from multiprocessing.dummy import Pool
import asyncio

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


def RunYoloModel(conf_thres=0.25, iou_thres=0.45, compute_exe_time=False):
    cap = cv2.VideoCapture(0)

    GPU_DEVICE = select_device('')
    YOLO_MODEL = load_model(YOLOV5S_PATH, GPU_DEVICE)
    # 0 1 2 3 5 7
    MODEL_OUTPUT_NAMES = get_names(YOLO_MODEL)
    MODEL_OUTPUT_COLOR = get_colors(MODEL_OUTPUT_NAMES)
    cudnn.benchmark = True

    while cap.isOpened():
        _, img = cap.read()
        tensor_img = img_transform(img_resize(img, 640), GPU_DEVICE)
        if compute_exe_time:
            t1 = time.time()
        pred = YOLO_MODEL(tensor_img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        if compute_exe_time:
            t2 = time.time()
            # noinspection PyUnboundLocalVariable
            print((t2 - t1) * 1000, "ms")
        paint_interested_result(pred, tensor_img, img, MODEL_OUTPUT_NAMES, MODEL_OUTPUT_COLOR)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def RunSignModel(IMGS_DIR_PATH=POS_IMGS_FOLDER_PATH, show=False, compute_exe_time=False):
    SIGN_CLASSIFIER = cv2.CascadeClassifier(CASCADE_FILE_PATH)
    imgs_folder_path = os.path.join(os.getcwd(), IMGS_DIR_PATH)
    img_names_list = os.listdir(imgs_folder_path)
    count = 0
    for img_name in img_names_list:
        img_path = imgs_folder_path + "\\" + img_name
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if compute_exe_time:
            t1 = time.time()
        detect = SIGN_CLASSIFIER.detectMultiScale(gray, 1.1, 1)
        if compute_exe_time:
            t2 = time.time()
            # noinspection PyUnboundLocalVariable
            print((t2 - t1) * 1000, "ms")

        if len(detect) > 0:
            for (x, y, w, h) in detect:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print(img_name)
            count += 1
            if show:
                try:
                    cv2.imshow("detect", img)
                    cv2.waitKey(4000)
                except cv2.error:
                    print(img_path)
    print(count)


def paint_interested_result(pred, tensor_img, origin_img, names, colors, path_img='', img_window=False, webcam=False):
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
            # reversed(det)
            for (*xyxy, conf, cls) in reversed(det):
                if int(cls.item()) in {0, 1, 2, 3, 5, 7}:
                    painted = True
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

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


async def RunModel(conf_thres=0.25, iou_thres=0.45, SOURCE=ImgsSource.CAMERA, IMG_FOLDER_PATH=None):
    if (SOURCE == ImgsSource.FILE or SOURCE == ImgsSource.VIDEO) and (IMG_FOLDER_PATH is None):
        raise Exception("path is None")
    SIGN_CLASSIFIER = cv2.CascadeClassifier(CASCADE_FILE_PATH)
    GPU_DEVICE = select_device('')
    YOLO_MODEL = load_model(YOLOV5S_PATH, GPU_DEVICE)
    # 0 1 2 3 5 7
    MODEL_OUTPUT_NAMES = get_names(YOLO_MODEL)
    MODEL_OUTPUT_COLOR = get_colors(MODEL_OUTPUT_NAMES)
    cudnn.benchmark = True

    if SOURCE == ImgsSource.CAMERA:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            _, img = cap.read()
            sign_pred, (yolo_pred, yolo_tensor_img) = await asyncio.gather(
                cascadePredictAsync(img, SIGN_CLASSIFIER),
                yoloPredictAsync(img, YOLO_MODEL, GPU_DEVICE, conf_thres, iou_thres)
            )
            paint(img,
                  sign_pred, yolo_pred, yolo_tensor_img,
                  MODEL_OUTPUT_NAMES, MODEL_OUTPUT_COLOR)

            cv2.imshow('camera', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif SOURCE == ImgsSource.FILE:
        imgs_folder_path = os.path.join(os.getcwd(), IMG_FOLDER_PATH)
        img_names_list = os.listdir(imgs_folder_path)
        for img_name in img_names_list:
            img_path = os.path.join(imgs_folder_path, img_name)
            img = cv2.imread(img_path)

            sign_pred, (yolo_pred, yolo_tensor_img) = await asyncio.gather(
                cascadePredictAsync(img, SIGN_CLASSIFIER),
                yoloPredictAsync(img, YOLO_MODEL, GPU_DEVICE, conf_thres, iou_thres)
            )
            yolo_painted, sign_painted = paint(img,
                                               sign_pred, yolo_pred, yolo_tensor_img,
                                               MODEL_OUTPUT_NAMES, MODEL_OUTPUT_COLOR)

            try:
                if yolo_painted or sign_painted:
                    cv2.imshow("img_file", img)
                    cv2.waitKey(3000)
            except cv2.error:
                print(img_path)
    elif SOURCE == ImgsSource.VIDEO:
        raise Exception("not implemented")
    else:
        raise Exception("unknown img source")


async def yoloPredictAsync(img, YOLO_MODEL, GPU_DEVICE, conf_thres, iou_thres):
    tensor_img = img_transform(img_resize(img, 640), GPU_DEVICE)
    yolo_pred = YOLO_MODEL(tensor_img)[0]
    yolo_pred = non_max_suppression(yolo_pred, conf_thres, iou_thres)
    return yolo_pred, tensor_img


async def cascadePredictAsync(img, SIGN_CLASSIFIER):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sign_detect = SIGN_CLASSIFIER.detectMultiScale(gray, 1.1, 1)
    return sign_detect


def paint(img,
          sign_detect,
          yolo_pred, yolo_tensor_img,
          MODEL_OUTPUT_NAMES, MODEL_OUTPUT_COLOR):
    sign_painted = False
    # paint cascade
    if len(sign_detect) > 0:
        sign_painted = True
        for (x, y, w, h) in sign_detect:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # paint yolo
    yolo_painted = paint_interested_result(yolo_pred, yolo_tensor_img, img, MODEL_OUTPUT_NAMES, MODEL_OUTPUT_COLOR)

    return yolo_painted, sign_painted


def findBroken(image_path: str):
    # noinspection PyBroadException
    try:
        io_img = io.imread(image_path)
        img = cv2.imread(image_path)
        with Image.open(image_path) as p_img:
            if img is None:
                print(image_path[image_path.rfind("\\") + 1:])
            elif io_img is None:
                print(image_path[image_path.rfind("\\") + 1:])
            elif p_img is None:
                print(image_path[image_path.rfind("\\") + 1:])
    except Exception as e:
        print(image_path[image_path.rfind("\\") + 1:])
        return False


def process(img_name: str):
    img_path = os.path.join(NEG_IMGS_FOLDER_PATH, img_name)
    findBroken(img_path)


def printBrokenImages():
    img_names = os.listdir(NEG_IMGS_FOLDER_PATH)

    pool: Pool = Pool()
    try:
        pool.map(process, img_names)
    except Exception as e:
        print(e)


def videoWithoutPredict():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, img = cap.read()
        cv2.imshow('camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    asyncio.run(RunModel())

    print("end")
