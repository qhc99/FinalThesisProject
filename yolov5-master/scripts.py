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


def RunModel(conf_thres=0.25, iou_thres=0.45, CAMERA=True, IMG_FOLDER_PATH=None):
    if not CAMERA and (IMG_FOLDER_PATH is None):
        raise Exception
    SIGN_CLASSIFIER = cv2.CascadeClassifier(CASCADE_FILE_PATH)
    GPU_DEVICE = select_device('')
    YOLO_MODEL = load_model(YOLOV5S_PATH, GPU_DEVICE)
    # 0 1 2 3 5 7
    MODEL_OUTPUT_NAMES = get_names(YOLO_MODEL)
    MODEL_OUTPUT_COLOR = get_colors(MODEL_OUTPUT_NAMES)
    cudnn.benchmark = True

    if CAMERA:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            _, img = cap.read()

            runTwoModelsAndPaint(img,
                                 YOLO_MODEL, GPU_DEVICE,
                                 conf_thres, iou_thres,
                                 MODEL_OUTPUT_NAMES, MODEL_OUTPUT_COLOR,
                                 SIGN_CLASSIFIER)

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        imgs_folder_path = os.path.join(os.getcwd(), IMG_FOLDER_PATH)
        img_names_list = os.listdir(imgs_folder_path)
        for img_name in img_names_list:
            img_path = os.path.join(imgs_folder_path, img_name)
            img = cv2.imread(img_path)

            yolo_painted, sign_painted = runTwoModelsAndPaint(img,
                                                              YOLO_MODEL, GPU_DEVICE,
                                                              conf_thres, iou_thres,
                                                              MODEL_OUTPUT_NAMES, MODEL_OUTPUT_COLOR,
                                                              SIGN_CLASSIFIER)

            try:
                if yolo_painted or sign_painted:
                    cv2.imshow("detect", img)
                    cv2.waitKey(3000)
            except cv2.error:
                print(img_path)


def runTwoModelsAndPaint(img,
                         YOLO_MODEL, GPU_DEVICE,
                         conf_thres, iou_thres,
                         MODEL_OUTPUT_NAMES, MODEL_OUTPUT_COLOR,
                         SIGN_CLASSIFIER):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # run yolo
    tensor_img = img_transform(img_resize(img, 640), GPU_DEVICE)
    pred = YOLO_MODEL(tensor_img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # run cascade
    detect = SIGN_CLASSIFIER.detectMultiScale(gray, 1.1, 1)

    sign_painted = False
    # paint cascade
    if len(detect) > 0:
        sign_painted = True
        for (x, y, w, h) in detect:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # paint yolo
    yolo_painted = paint_interested_result(pred, tensor_img, img, MODEL_OUTPUT_NAMES, MODEL_OUTPUT_COLOR)

    return yolo_painted, sign_painted


if __name__ == "__main__":
    # RunYoloModel(compute_exe_time=True)
    # RunSignModel(NEG_IMGS_FOLDER_PATH, show=True, compute_exe_time=True)
    RunModel()

    print("end")
