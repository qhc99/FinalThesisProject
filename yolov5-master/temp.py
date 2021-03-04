import cv2
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors, img_resize, img_transform, paint_result
from utils.general import non_max_suppression
import os
import hashlib


# FPS test
def FPS_Test(weights_path, conf_thres=0.25, iou_thres=0.45, ):
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


def ModelCheck():
    imgs_folder_path = os.getcwd() + "\\dataset\\TrafficBlockSign\\pos\\img"
    img_names_list = os.listdir(imgs_folder_path)
    block_sign_cascade_path = "./dataset/TrafficBlockSign/model/cascade.xml"
    sign_classifier = cv2.CascadeClassifier(block_sign_cascade_path)
    for img_name in img_names_list:
        img_path = imgs_folder_path + "\\" + img_name
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


def GetParamNumNeg(window_size=24):
    neg_imgs_folder_path = "./dataset/TrafficBlockSign/neg/img"
    img_names_list = os.listdir(neg_imgs_folder_path)
    res = 0
    for img_name in img_names_list:
        img = cv2.imread(neg_imgs_folder_path + "/" + img_name, cv2.IMREAD_GRAYSCALE)
        row_count, col_count = img.shape
        r = row_count // window_size
        c = col_count // window_size
        res = res + r * c
    print("window size: ", window_size)
    print("availabe neg images in training should be ", res)


def ImgFormatCheck():
    imgs_folder_path = "./dataset/TrafficBlockSign/neg/img"
    img_names = os.listdir(imgs_folder_path)
    for img_name in img_names:
        img_path = os.path.join(imgs_folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(img_name)


def removeImgDuplicate():

    def md5sum(filename):
        f = open(filename, 'rb')
        md5 = hashlib.md5()
        while True:
            fb = f.read(8096)
            if not fb:
                break
            md5.update(fb)
        f.close()
        return md5.hexdigest()
    all_md5 = {}

    filedir = os.walk(os.path.join(os.getcwd(), "./dataset/TrafficBlockSign/neg/img"))
    for i in filedir:
        for tlie in i[2]:
            file_path = os.path.join(os.getcwd(), "./dataset/TrafficBlockSign/neg/img", tlie)
            if md5sum(file_path) in all_md5.values():
                print(tlie)
                os.remove(file_path)
            else:
                all_md5[tlie] = md5sum(file_path)


if __name__ == "__main__":
    # FPS_Test('parameters/original/yolov5s.pt',)
    ModelCheck()
    # GetParamNumNeg()
    # ImgFormatCheck()
    # removeImgDuplicate()
    print("succ")
