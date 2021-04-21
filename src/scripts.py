import cv2
import time
import os
import numpy as np
from enum import Enum
from PIL import Image
import torch.backends.cudnn as cudnn
from multiprocessing import Process, Queue

from predict import img_resize, img_transform
from utils.general import non_max_suppression
from utils.plots import plot_one_box
from utils.general import scale_coords
from globals import TRAFFIC_MODEL, TRAFFIC_COLOR, TRAFFIC_NAMES,\
    INTRESTED_CLASSES, GPU_DEVICE, SIGN_CLASSIFIER, CONFI_THRES, IOU_THRES, FONT
cudnn.benchmark = True


# tensor_img[2:]
def yoloPaint(pred, tensor_shape, origin_img, names, colors):
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', origin_img
        s += '%gx%g ' % tensor_shape  # print string
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(tensor_shape, det[:, :4], im0.shape).round()
            # Write results
            for (*xyxy, conf, cls) in reversed(det):
                if int(cls.item()) in INTRESTED_CLASSES:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)


def opencvPaint(sign_pred, img):
    if len(sign_pred) > 0:
        label = "prohibit"
        for (x, y, w, h) in sign_pred:
            xyxy = [x, y, x + w, y + h]
            plot_one_box(xyxy, img, label=label, color=[255, 153, 0], line_thickness=2)


class ImgsSource(Enum):
    CAMERA = 0
    FILE = 1
    VIDEO = 2


def RunModels(SOURCE=ImgsSource.CAMERA, SOURCE_PATH=None):
    if (SOURCE == ImgsSource.FILE or SOURCE == ImgsSource.VIDEO) and (SOURCE_PATH is None):
        raise Exception("path is None")

    mp = ModelProcesses()
    mp.start()

    if SOURCE == ImgsSource.CAMERA:
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        cv2.namedWindow("camera")
        cv2.moveWindow('camera', 300, 115)
        last_time = time.time()

        while cap.isOpened():
            read_succ, cv2_img = cap.read()
            if not read_succ:
                break
            pil_img = cv2_to_pil(cv2_img)
            mp.traffic_in.put(pil_img, True)
            mp.sign_in.put(pil_img, True)

            sign_pred = mp.sign_out.get(True)
            opencvPaint(sign_pred, cv2_img)
            (traffic_pred, tensor_shape) = mp.traffic_out.get(True)
            yoloPaint(traffic_pred, tensor_shape, cv2_img, TRAFFIC_NAMES, TRAFFIC_COLOR)

            current_latency = (time.time() - last_time) * 1000
            last_time = time.time()
            cv2.putText(cv2_img, "FPS:%.1f" % (1000 / current_latency), (0, 15), FONT, 0.5, (255, 80, 80), 1,
                        cv2.LINE_4)

            cv2.imshow('camera', cv2_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        mp.terminate()

    elif SOURCE == ImgsSource.FILE:
        imgs_folder_path = os.path.join(os.getcwd(), SOURCE_PATH)
        img_names_list = os.listdir(imgs_folder_path)

        for img_name in img_names_list:
            img_path = os.path.join(imgs_folder_path, img_name)

            cv2_img = cv2.imread(img_path)
            pil_img = cv2_to_pil(cv2_img)
            mp.traffic_in.put(pil_img, True)
            mp.sign_in.put(pil_img, True)

            sign_pred = mp.sign_out.get(True)
            opencvPaint(sign_pred, cv2_img)
            (traffic_pred, tensor_shape) = mp.traffic_out.get(True)
            yoloPaint(traffic_pred, tensor_shape, cv2_img, TRAFFIC_NAMES, TRAFFIC_COLOR)

            cv2.imshow('camera', cv2_img)
            if cv2.waitKey(3000) & 0xFF == ord('q'):
                break
        mp.terminate()

    elif SOURCE == ImgsSource.VIDEO:
        cap = cv2.VideoCapture(SOURCE_PATH)
        cv2.namedWindow("video")
        cv2.moveWindow('video', 300, 115)
        last_time = time.time()

        while cap.isOpened():
            read_succ, cv2_img = cap.read()
            if not read_succ:
                break
            pil_img = cv2_to_pil(cv2_img)
            mp.traffic_in.put(pil_img, True)
            mp.sign_in.put(pil_img, True)

            sign_pred = mp.sign_out.get(True)
            opencvPaint(sign_pred, cv2_img)
            (traffic_pred, tensor_shape) = mp.traffic_out.get(True)
            yoloPaint(traffic_pred, tensor_shape, cv2_img, TRAFFIC_NAMES, TRAFFIC_COLOR)

            current_latency = (time.time() - last_time) * 1000
            last_time = time.time()
            cv2.putText(cv2_img, "FPS:%.1f" % (1000 / current_latency), (0, 15), FONT, 0.5, (255, 80, 80), 1,
                        cv2.LINE_4)

            cv2.imshow('video', cv2_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        mp.terminate()

    else:
        raise Exception("unknown img source")


def trafficPredict(in_queue: Queue, out_queue: Queue):
    while True:
        img = in_queue.get(True)
        img = pil_to_cv2(img)
        tensor_img = img_transform(img_resize(img, 640), GPU_DEVICE)
        yolo_pred = TRAFFIC_MODEL(tensor_img)[0]
        yolo_pred = non_max_suppression(yolo_pred, CONFI_THRES, IOU_THRES)
        for i, data in enumerate(yolo_pred):
            yolo_pred[i] = data.cpu().detach()
        out_queue.put((yolo_pred, tensorShape(tensor_img)), True)


def signPredict(in_queue: Queue, out_queue: Queue):
    while True:
        img = in_queue.get(True)
        img = pil_to_cv2(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sign_detect = SIGN_CLASSIFIER.detectMultiScale(gray, 1.1, 1)
        out_queue.put(sign_detect, True)


class ModelProcesses:
    def __init__(self):
        self.traffic_in = Queue()
        self.traffic_out = Queue()
        self.__traffic_process = Process(target=trafficPredict, args=(self.traffic_in, self.traffic_out))

        self.sign_in = Queue()
        self.sign_out = Queue()
        self.__sign_process = Process(target=signPredict, args=(self.sign_in, self.sign_out))

    def start(self):
        self.__traffic_process.start()
        self.__sign_process.start()

    def terminate(self):
        self.__traffic_process.terminate()
        self.__sign_process.terminate()


def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def tensorShape(tensor_img):
    return tensor_img.shape[2:]


if __name__ == "__main__":
    RunModels(SOURCE=ImgsSource.VIDEO, SOURCE_PATH="./resources/sign_demo_1.avi")
    # RunModels()
    print("success")
