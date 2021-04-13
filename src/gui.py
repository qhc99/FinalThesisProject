import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from scripts import paint, yoloPredict, signPredict
import cv2
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors
import os
from pathlib import Path
import time
import asyncio

MODEL_NUM = 3
# 1: 639
# 3: 381
CASCADE_FILE_PATH = "../../dataset/TrafficBlockSign/models/model" + str(MODEL_NUM) + "/cascade.xml"
SIGN_CLASSIFIER = cv2.CascadeClassifier(CASCADE_FILE_PATH)

YOLOV5S_PATH = "./parameters/original/yolov5s.pt"
CONFI_THRES = 0.25
IOU_THRES = 0.45
GPU_DEVICE = select_device('')
YOLO_MODEL = load_model(YOLOV5S_PATH, GPU_DEVICE)
MODEL_OUTPUT_NAMES = get_names(YOLO_MODEL)  # 0 1 2 3 5 7
MODEL_OUTPUT_COLOR = get_colors(MODEL_OUTPUT_NAMES)
cudnn.benchmark = True

FONT = cv2.FONT_HERSHEY_SIMPLEX


class TrafficSystemGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.initMainWindow()
        self.initImageBox()
        self.initButtons()
        self.initFPSText()
        self.threadPool = QThreadPool()

    def initMainWindow(self):
        self.__height = 579
        self.__width = 765
        self.__top = 300
        self.__left = 610
        self.__title = "交通路况系统"
        self.setWindowTitle(self.__title)
        self.setGeometry(self.__left, self.__top, self.__width, self.__height)
        self.setFixedWidth(self.__width)
        self.setFixedHeight(self.__height)
        self.setWindowIcon(QIcon("resources/hohai.png"))

    def initImageBox(self):
        self.ImageBox = QLabel(self)
        self.ImageBox.setGeometry(QRect(10, 10, 640, 480))
        self.ImageBox.setObjectName("ImageBox")

    def initButtons(self):
        self.widget = QWidget(self)
        self.widget.setGeometry(QRect(660, 250, 95, 126))
        self.widget.setObjectName("widget")

        self.buttonsLayout = QVBoxLayout(self.widget)
        self.buttonsLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonsLayout.setSpacing(20)
        self.buttonsLayout.setObjectName("verticalLayout")

        self.CameraButton = QPushButton(self.widget)
        self.CameraButton.setObjectName("CameraButton")
        self.buttonsLayout.addWidget(self.CameraButton)
        self.CameraButton.setText("摄像头:off")
        # noinspection PyUnresolvedReferences
        self.CameraButton.clicked.connect(self.clickCameraButton)
        self.cap = None

        self.VideoButtion = QPushButton(self.widget)
        self.VideoButtion.setObjectName("VideoButtion")
        self.buttonsLayout.addWidget(self.VideoButtion)
        self.VideoButtion.setText("视频:off")

        self.FileButton = QPushButton(self.widget)
        self.FileButton.setObjectName("FileButton")
        self.buttonsLayout.addWidget(self.FileButton)
        self.FileButton.setText("文件:off")

    def initFPSText(self):
        self.FPSTextLabel = QLabel(self)
        self.FPSTextLabel.setObjectName("FPSTextLabel")
        self.FPSTextLabel.setGeometry(500, 500, 75, 23)
        self.FPSTextLabel.setText("FPS:")

    @pyqtSlot()
    def FPS_SwitchPressed(self):
        if not self.FPSSwitch.isChecked():
            self.FPSTextLabel.setText("FPS:")

    @pyqtSlot()
    def clickCameraButton(self):
        if self.CameraButton.text().endswith("off"):
            self.VideoButtion.setEnabled(False)
            self.CameraButton.setText(self.CameraButton.text().replace("off", "on"))
            self.threadPool.start(self.cameraRunModels)
        else:
            self.CameraButton.setText(self.CameraButton.text().replace("on", "off"))
            self.cap.release()
            self.FPSTextLabel.setText("FPS:")
            self.ImageBox.clear()
            self.VideoButtion.setEnabled(True)

    @pyqtSlot()
    def cameraRunModels(self):
        self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        sign_pred = None
        yolo_pred, yolo_tensor_img = None, None

        # warm up
        if self.cap.isOpened():
            _, img = self.cap.read()
            sign_pred = signPredict(img)
            yolo_pred, yolo_tensor_img = yoloPredict(img)

        # process trick
        process_yolo = True

        # FPS init
        last_time = time.time()
        yolo_latency, sign_latency = 0, 0
        while self.cap.isOpened():
            read_succ, img = self.cap.read()
            if not read_succ:
                break

            if process_yolo:
                process_yolo = not process_yolo
                yolo_pred, yolo_tensor_img = yoloPredict(img)
            else:
                process_yolo = not process_yolo
                sign_pred = signPredict(img)

            paint(img, sign_pred, yolo_pred, yolo_tensor_img)

            current_latency = (time.time() - last_time)
            last_time = time.time()
            if process_yolo:
                yolo_latency = current_latency
            else:
                sign_latency = current_latency
                self.FPSTextLabel.setText("FPS:%.1f" % (2.0 / (yolo_latency + sign_latency)))

            img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped()
            self.ImageBox.setPixmap(QPixmap.fromImage(img))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = TrafficSystemGUI()
    ui.show()
    sys.exit(app.exec_())
