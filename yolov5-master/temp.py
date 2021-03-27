import cv2
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors, img_resize, img_transform, paint_result
from utils.general import non_max_suppression
import os
from skimage import io
from PIL import Image

NEG_IMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/neg/img"
NEG_T_IMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/neg/t_imgs"
POS_IMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/pos/img"
CASCADE_FILE_PATH = "../../dataset/TrafficBlockSign/model/cascade.xml"


# FPS test
def RunYoloModel(weights_path, conf_thres=0.25, iou_thres=0.45, ):
    cap = cv2.VideoCapture(0)
    device = select_device('')
    model = load_model(weights_path, device)
    cudnn.benchmark = True
    names = get_names(model)
    colors = get_colors(names)

    while cap.isOpened():
        _, img = cap.read()
        tensor_img = img_transform(img_resize(img, 640), device)
        pred = model(tensor_img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        paint_result(pred, tensor_img, img, names, colors)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def RunSignModel():
    pos_imgs_folder_path = os.path.join(os.getcwd(), POS_IMGS_FOLDER_PATH)
    img_names_list = os.listdir(pos_imgs_folder_path)
    sign_classifier = cv2.CascadeClassifier(CASCADE_FILE_PATH)
    for img_name in img_names_list:
        img_path = pos_imgs_folder_path + "\\" + img_name
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        res = sign_classifier.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in res:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        try:
            cv2.imshow("detect", img)
            cv2.waitKey(2500)
        except cv2.error:
            print(img_path)


def ImgCheck(IMGS_FILE=NEG_IMGS_FOLDER_PATH):
    img_names = os.listdir(IMGS_FILE)
    for img_name in img_names:
        img_path = os.path.join(IMGS_FILE, img_name)
        img = cv2.imread(img_path)
        shape = img.shape
        if shape[0] > shape[1]:
            print(img_name)
        if (shape[0] > 700) or (shape[1] > 700):
            if shape[0] > shape[1]:
                factor = shape[0]/600
            else:
                factor = shape[1]/600
            new_img = cv2.resize(img, (int(shape[1]/factor+0.5), int(shape[0]/factor+0.5)))
            cv2.imshow("t", new_img)
            cv2.waitKey(500)


if __name__ == "__main__":
    # RunYoloModel('parameters/original/yolov5s.pt')
    # RunSignModel()
    ImgCheck("../../dataset/TrafficBlockSign/pos/new_imgs")

    print("succ")
