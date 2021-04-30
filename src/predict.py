from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def pipeline(source, weights, img_size=640,
             img_window=False, augment_arg=False, conf_thres_arg=0.25,
             iou_thres_arg=0.45, classes_arg=None, agnostic_nms_arg=False):
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    set_logging()
    device = select_device('')
    model = load_model(weights, device)
    imgsz = check_img_size(img_size, s=model.stride.max())

    if webcam:
        cudnn.benchmark = True  # speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    names = get_names(model)
    colors = get_colors(names)

    for path, img, im0s, vid_cap in dataset:
        img = img_transform(img, device)
        pred = model(img, augment=augment_arg)[0]
        pred = non_max_suppression(pred, conf_thres_arg, iou_thres_arg, classes=classes_arg, agnostic=agnostic_nms_arg)
        paint_result(pred, img, im0s, names, colors, path, img_window, webcam)


def get_names(model):
    return model.module.names if hasattr(model, 'module') else model.names


def get_colors(names):
    return [[random.randint(0, 255) for _ in range(3)] for _ in names]


def load_model(weights_path, device):
    return attempt_load(weights_path, map_location=device).half()


def img_resize(img_origin, img_size=640):
    # Padded resize
    img = letterbox(img_origin, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img


def img_transform(img, device):
    img = torch.from_numpy(img).to(device)
    img = img.half()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def paint_result(pred, tensor_img, origin_img, names, colors, path_img='', img_window=False, webcam=False):
    # Process detections
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
            for *xyxy, conf, cls in reversed(det):
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
    return origin_img
