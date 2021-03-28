import cv2
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors, img_resize, img_transform
from utils.general import non_max_suppression
import os
from utils.plots import plot_one_box
from utils.general import scale_coords
from pathlib import Path


NEG_IMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/neg/img"
POS_IMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/pos/img"
MODEL_NUM = 1
CASCADE_FILE_PATH = "../../dataset/TrafficBlockSign/model" + str(MODEL_NUM) + "/cascade.xml"

YOLOV5L_PATH = "./parameters/original/yolov5l.pt"
YOLOV5M_PATH = "./parameters/original/yolov5m.pt"
YOLOV5S_PATH = "./parameters/original/yolov5s.pt"

STATIC_GPU_DEVICE = select_device('')
STATIC_YOLO_MODEL = load_model(YOLOV5S_PATH, STATIC_GPU_DEVICE)
# 0 1 2 3 5 7
STATIC_MODEL_OUTPUT_NAMES = get_names(STATIC_YOLO_MODEL)
STATIC_MODEL_OUTPUT_COLOR = get_colors(STATIC_MODEL_OUTPUT_NAMES)
cudnn.benchmark = True

STATIC_SIGN_CLASSIFIER = cv2.CascadeClassifier(CASCADE_FILE_PATH)


def paint_interested_result(pred, tensor_img, origin_img, names, colors, path_img='', img_window=False, webcam=False):
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
            # reversed(det)
            for (*xyxy, conf, cls) in reversed(det):
                if cls in [0, 1, 2, 3, 5, 7]:
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


def RunYoloModel(conf_thres=0.25, iou_thres=0.45, ):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _, img = cap.read()
        tensor_img = img_transform(img_resize(img, 640), STATIC_GPU_DEVICE)
        pred = STATIC_YOLO_MODEL(tensor_img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        paint_interested_result(pred, tensor_img, img, STATIC_MODEL_OUTPUT_NAMES, STATIC_MODEL_OUTPUT_COLOR)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def RunSignModel():
    pos_imgs_folder_path = os.path.join(os.getcwd(), POS_IMGS_FOLDER_PATH)
    img_names_list = os.listdir(pos_imgs_folder_path)
    for img_name in img_names_list:
        img_path = pos_imgs_folder_path + "\\" + img_name
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = STATIC_SIGN_CLASSIFIER.detectMultiScale(gray)

        for (x, y, w, h) in res:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        try:
            cv2.imshow("detect", img)
            cv2.waitKey(3000)
        except cv2.error:
            print(img_path)


if __name__ == "__main__":
    RunYoloModel()
    # RunSignModel()

    print("end")
