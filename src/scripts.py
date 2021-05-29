from cv2 import VideoCapture, putText, LINE_4, namedWindow, imread, moveWindow, waitKey, destroyWindow, \
    CAP_DSHOW, imshow, cvtColor, COLOR_BGR2RGB, COLOR_BGR2GRAY, COLOR_RGB2BGR
from time import time
from os import getcwd, listdir
from os.path import join
from numpy import array
from enum import Enum

from torch.cuda import synchronize
from PIL.Image import fromarray
from torch.backends import cudnn
from multiprocessing import Process, Queue

from predict import img_resize, img_transform
from utils.general import non_max_suppression
from utils.plots import plot_one_box
from utils.general import scale_coords
from globals import TRAFFIC_MODEL, TRAFFIC_COLOR, TRAFFIC_NAMES, \
    INTRESTED_CLASSES, GPU_DEVICE, SIGN_CLASSIFIER, CONFI_THRES, IOU_THRES, FONT

cudnn.benchmark = True


def yoloPaint(pred, tensor_shape, origin_img, names, colors, interested_class=INTRESTED_CLASSES):
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', origin_img
        s += '%gx%g ' % tensor_shape  # print string
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(tensor_shape, det[:, :4], im0.shape).round()
            # Write results
            for (*xyxy, conf, cls) in reversed(det):
                cls = int(cls)
                if cls in interested_class:
                    label = f'{names[cls]} {conf:.2f}' if cls != 11 else f"{names[cls]}"
                    plot_one_box(xyxy, im0, label=label, color=colors[cls], line_thickness=2)


def opencvPaint(sign_pred, img):
    if len(sign_pred) > 0:
        for (x, y, w, h) in sign_pred:
            xyxy = [x, y, x + w, y + h]
            label = "prohibit"
            plot_one_box(xyxy, img, label=label, color=[255, 153, 0], line_thickness=2)


class ImgsSource(Enum):
    CAMERA = 0
    FILE = 1
    VIDEO = 2


# noinspection DuplicatedCode
def RunModels(SOURCE=ImgsSource.CAMERA, SOURCE_PATH=None):
    if (SOURCE == ImgsSource.FILE or SOURCE == ImgsSource.VIDEO) and (SOURCE_PATH is None):
        raise Exception("path is None")

    sign_in = Queue()
    sign_out = Queue()
    sign_process = Process(target=signPredict, args=(sign_in, sign_out))
    sign_process.start()

    if SOURCE == ImgsSource.CAMERA:
        cap = VideoCapture(0 + CAP_DSHOW)
        windows_show = False
        last_time = time()

        while cap.isOpened():
            read_succ, cv2_img = cap.read()
            if not read_succ:
                break
            pil_img = cv2_to_pil(cv2_img)
            sign_in.put(pil_img, True)

            tensor_img = img_transform(img_resize(cv2_img, 32 * 15), GPU_DEVICE)
            traffic_pred = TRAFFIC_MODEL(tensor_img)[0]
            traffic_pred = non_max_suppression(traffic_pred, CONFI_THRES, IOU_THRES)
            yoloPaint(traffic_pred, tensorShape(tensor_img), cv2_img, TRAFFIC_NAMES, TRAFFIC_COLOR)

            sign_pred = sign_out.get(True)
            opencvPaint(sign_pred, cv2_img)

            synchronize()
            t = time()
            current_latency = (t - last_time) * 1000
            last_time = t
            putText(cv2_img, "FPS:%.1f" % (1000 / current_latency), (0, 15), FONT, 0.5, (255, 80, 80), 1,
                    LINE_4)

            if not windows_show:
                namedWindow("camera")
                moveWindow('camera', 300, 115)
                windows_show = True

            imshow('camera', cv2_img)
            if waitKey(1) & 0xFF == ord('q'):
                break
        sign_process.terminate()
    elif SOURCE == ImgsSource.FILE:
        imgs_folder_path = join(getcwd(), SOURCE_PATH)
        img_names_list = listdir(imgs_folder_path)

        for img_name in img_names_list:
            img_path = join(imgs_folder_path, img_name)
            cv2_img = imread(img_path)

            pil_img = cv2_to_pil(cv2_img)
            sign_in.put(pil_img, True)

            tensor_img = img_transform(img_resize(cv2_img, 640), GPU_DEVICE)
            traffic_pred = TRAFFIC_MODEL(tensor_img)[0]
            traffic_pred = non_max_suppression(traffic_pred, CONFI_THRES, IOU_THRES)
            yoloPaint(traffic_pred, tensorShape(tensor_img), cv2_img, TRAFFIC_NAMES, TRAFFIC_COLOR,
                      interested_class={0, 1, 2, 3, 5, 7})

            sign_pred = sign_out.get(True)
            opencvPaint(sign_pred, cv2_img)

            namedWindow("file")
            moveWindow('file', 300, 115)
            imshow('file', cv2_img)
            waitKey()
            destroyWindow("file")
        sign_process.terminate()

    elif SOURCE == ImgsSource.VIDEO:
        count = 0
        cap = VideoCapture(SOURCE_PATH)
        window_show = False
        last_time = time()

        while cap.isOpened():
            read_succ, cv2_img = cap.read()
            count += 1
            if not read_succ:
                break
            pil_img = cv2_to_pil(cv2_img)
            sign_in.put(pil_img, True)

            tensor_img = img_transform(img_resize(cv2_img, 480), GPU_DEVICE)
            traffic_pred = TRAFFIC_MODEL(tensor_img)[0]
            traffic_pred = non_max_suppression(traffic_pred, CONFI_THRES, IOU_THRES)
            yoloPaint(traffic_pred, tensorShape(tensor_img), cv2_img, TRAFFIC_NAMES, TRAFFIC_COLOR)

            sign_pred = sign_out.get(True)
            opencvPaint(sign_pred, cv2_img)

            synchronize()
            t = time()
            current_latency = (t - last_time) * 1000
            last_time = t
            putText(cv2_img, "FPS:%.1f" % (1000 / current_latency), (0, 15), FONT, 0.5, (255, 80, 80), 1,
                    LINE_4)

            if not window_show:
                namedWindow("video")
                moveWindow('video', 300, 115)
                window_show = True

            imshow('video', cv2_img)
            if waitKey(1) & 0xFF == ord('q'):
                break
        sign_process.terminate()
    else:
        raise Exception("unknown img source")


# def trafficPredict(in_queue: Queue, out_queue: Queue):
#     while True:
#         img = in_queue.get(True)
#         img = pil_to_cv2(img)
#         tensor_img = img_transform(img_resize(img, 640), GPU_DEVICE)
#         yolo_pred = TRAFFIC_MODEL(tensor_img)[0]
#         yolo_pred = non_max_suppression(yolo_pred, CONFI_THRES, IOU_THRES)
#         for i, data in enumerate(yolo_pred):
#             yolo_pred[i] = data.cpu().detach()
#         out_queue.put((yolo_pred, tensorShape(tensor_img)), True)


def signPredict(in_queue: Queue, out_queue: Queue):
    while True:
        img = in_queue.get(True)
        img = pil_to_cv2(img)
        gray = cvtColor(img, COLOR_BGR2GRAY)
        sign_detect = SIGN_CLASSIFIER.detectMultiScale(gray, 1.1, 2)
        out_queue.put(sign_detect, True)


def cv2_to_pil(img):
    return fromarray(cvtColor(img, COLOR_BGR2RGB))


def pil_to_cv2(img):
    return cvtColor(array(img), COLOR_RGB2BGR)


def tensorShape(tensor_img):
    return tensor_img.shape[2:]


if __name__ == "__main__":
    # RunModels(SOURCE=ImgsSource.VIDEO, SOURCE_PATH="./resources/demo.mov")
    # RunModels(SOURCE=ImgsSource.FILE, SOURCE_PATH="D:\\data")
    RunModels(SOURCE=ImgsSource.CAMERA)
    print("success")
