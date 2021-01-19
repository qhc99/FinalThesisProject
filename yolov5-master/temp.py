import cv2
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors, img_resize, img_transform, paint_result
from utils.general import non_max_suppression


def temp1(weights_path, conf_thres=0.25, iou_thres=0.45,):
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


if __name__ == "__main__":
    temp1('parameters/original/yolov5s.pt',)
