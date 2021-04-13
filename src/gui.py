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
# 0 1 2 3 5 7
MODEL_OUTPUT_NAMES = get_names(YOLO_MODEL)
MODEL_OUTPUT_COLOR = get_colors(MODEL_OUTPUT_NAMES)
cudnn.benchmark = True

FONT = cv2.FONT_HERSHEY_SIMPLEX


class TrafficSystemGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.initMainWindow()
        self.initImageBox()
        self.initButtons()
        self.initFPSGroup()
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
        self.scene = QGraphicsScene()
        self.ImageBox = QGraphicsView(self.scene)
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

    def initFPSGroup(self):
        self.widget1 = QWidget(self)
        self.widget1.setGeometry(QRect(440, 500, 250, 24))
        self.widget1.setObjectName("widget1")

        self.FPS_StatusLayout = QHBoxLayout(self.widget1)
        self.FPS_StatusLayout.setContentsMargins(0, 0, 0, 0)
        self.FPS_StatusLayout.setSpacing(30)
        self.FPS_StatusLayout.setObjectName("horizontalLayout")

        self.FPSSwitch = QRadioButton(self.widget1)
        self.FPSSwitch.setObjectName("FPSSwitch")
        self.FPS_StatusLayout.addWidget(self.FPSSwitch)
        self.FPSSwitch.setText("FPS监测")
        # noinspection PyUnresolvedReferences
        self.FPSSwitch.toggled.connect(self.FPS_SwitchPressed)
        self.FPSSwitchOn = False

        self.FPSTextLabel = QLabel(self.widget1)
        self.FPSTextLabel.setObjectName("FPSTextLabel")
        self.FPS_StatusLayout.addWidget(self.FPSTextLabel)
        self.FPSTextLabel.setText("FPS:")

    @pyqtSlot()
    def FPS_SwitchPressed(self):
        if not self.FPSSwitch.isChecked():
            self.FPSTextLabel.setText("FPS:")

    @pyqtSlot()
    def clickCameraButton(self):
        if self.CameraButton.text().endswith("off"):
            self.CameraButton.setText(self.CameraButton.text().replace("off", "on"))
            self.threadPool.start(self.cameraRunModels)
        else:
            self.CameraButton.setText(self.CameraButton.text().replace("on", "off"))
            self.cap.release()

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

        # cv2.namedWindow("camera")
        # cv2.moveWindow('camera', 300, 115)

        # FPS init
        frame_count = -1
        last_time = time.time()
        latency = 0

        while self.cap.isOpened():
            _, img = self.cap.read()
            if process_yolo:
                process_yolo = not process_yolo
                yolo_pred, yolo_tensor_img = yoloPredict(img)
            else:
                process_yolo = not process_yolo
                sign_pred = signPredict(img)

            paint(img, sign_pred, yolo_pred, yolo_tensor_img)

            if self.FPSSwitch.isChecked():
                frame_count += 1
                current_latency = (time.time() - last_time) * 1000
                last_time = time.time()
                if frame_count == 1:
                    latency = current_latency
                elif frame_count > 1:
                    latency = ((frame_count - 1) * latency + current_latency) / frame_count

                if latency > 0:
                    # cv2.putText(img, "FPS:%.1f" % (1000 / latency), (0, 20), FONT, 0.5, (255, 80, 80), 1, cv2.LINE_4)
                    self.FPSTextLabel.setText("FPS:%.1f" % (1000 / latency))

            img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped()
            self.scene.clear()
            self.scene.addPixmap(QPixmap.fromImage(img))
            self.scene.update()
            # cv2.imshow('camera', img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = TrafficSystemGUI()
    ui.show()
    sys.exit(app.exec_())
