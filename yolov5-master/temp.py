import cv2
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors, img_resize, img_transform, paint_result
from utils.general import non_max_suppression
import os


# FPS test
def FPS_Test(weights_path, conf_thres=0.25, iou_thres=0.45,):
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
    for img_name in img_names_list:
        img_path = imgs_folder_path + "\\" + img_name
        img = cv2.imread(img_path, 0)
        try:
            cv2.imshow("temp", img)
            cv2.waitKey(3000)
        except cv2.error:
            print(img_path)


if __name__ == "__main__":
    # FPS_Test('parameters/original/yolov5s.pt',)
    ModelCheck()
