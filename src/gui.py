import sys
import time

import cv2
import torch.backends.cudnn as cudnn
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from globals import TRAFFIC_NAMES, TRAFFIC_COLOR
from scripts import cv2_to_pil, opencvPaintPrediction, yoloPaintPrediction, FONT, ModelProcesses

cudnn.benchmark = True


class TrafficSystemGUI(QWidget):

    # noinspection PyArgumentList
    def __init__(self):
        super().__init__()
        self.initMainWindow()
        self.initImageGroup()
        self.initButtons()
        self.initReadMe()
        self.threadPool = QThreadPool()

    def initMainWindow(self):
        self.__height = 963
        self.__width = 1894
        self.__top = 45
        self.__left = 20
        self.__title = "交通路况系统"
        self.setWindowTitle(self.__title)
        self.setGeometry(self.__left, self.__top, self.__width, self.__height)
        self.setFixedWidth(self.__width)
        self.setFixedHeight(self.__height)
        self.setWindowIcon(QIcon("resources/hohai.png"))

    def initImageGroup(self):
        self.ImageBox = QGraphicsView(self)
        self.ImageBox.setGeometry(QRect(10, 10, 1521, 941))

        self.ImageScreen = QLabel(self)
        rect = QRect(10, 10, 200, 200)
        self.ImageScreen.setGeometry(rect)
        self.ImageScreen.setText("")

        self.ImageScreenTop = rect.top()
        self.ImageScreenLeft = rect.left()
        self.ImageScreenWidth = rect.width()
        self.ImageScreenHeight = rect.height()

    # noinspection PyArgumentList,PyUnresolvedReferences
    def initButtons(self):
        self.VideoButtion = QPushButton(self)
        self.VideoButtion.setGeometry(QRect(1580, 630, 241, 101))
        self.VideoButtion.setText("视频:off")
        self.VideoButtion.clicked.connect(self.clickVideoButton)

        font = QFont()
        font.setPointSize(24)
        font.setFamily("楷体")
        self.VideoButtion.setFont(font)

        self.CameraButton = QPushButton(self)
        self.CameraButton.setGeometry(QRect(1580, 850, 241, 101))
        self.CameraButton.setText("摄像头:off")
        self.CameraButton.clicked.connect(self.clickCameraButton)
        self.CameraButton.setFont(font)

        self.cap = None

    def initReadMe(self):
        self.ReadMeLabel = QLabel(self)
        self.ReadMeLabel.setGeometry(QRect(1540, 270, 351, 131))
        font = QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.ReadMeLabel.setFont(font)
        self.ReadMeLabel.setText("<html><head/><body><p>绿色:行人</p>"
                                 "<p>红色:汽车(轿车、卡车、摩托车)</p><p>蓝色:禁止标志</p></body></html>")

    @pyqtSlot()
    def clickVideoButton(self):
        if self.VideoButtion.text().endswith("off"):
            self.CameraButton.setEnabled(False)
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            # noinspection PyTypeChecker
            file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "",
                                                       "All Files (*);;Python Files (*.py)", options=options)
            if len(file_path) > 0:
                self.VideoButtion.setText(self.VideoButtion.text().replace("off", "on"))
                self.threadPool.start(lambda: self.videoRunmodels(video_path=file_path))
            else:
                self.CameraButton.setEnabled(True)
        else:
            self.VideoButtion.setText(self.VideoButtion.text().replace("on", "off"))
            self.cap.release()
            self.ImageScreen.clear()
            self.CameraButton.setEnabled(True)

    @pyqtSlot()
    def videoRunmodels(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

        mp = ModelProcesses()
        mp.start()

        last_time = time.time()

        while self.cap.isOpened():
            read_succ, cv2_img = self.cap.read()
            if not read_succ:
                break

            pil_img = cv2_to_pil(cv2_img)
            mp.traffic_in.put(pil_img, True)
            mp.sign_in.put(pil_img, True)

            sign_pred = mp.sign_out.get(True)
            opencvPaintPrediction(sign_pred, cv2_img)
            (traffic_pred, tensor_shape) = mp.traffic_out.get(True)
            yoloPaintPrediction(traffic_pred, tensor_shape, cv2_img, TRAFFIC_NAMES, TRAFFIC_COLOR)

            current_latency = (time.time() - last_time) * 1000
            last_time = time.time()
            cv2.putText(cv2_img, "FPS:%.1f" % (1000 / current_latency), (0, 15), FONT, 0.5, (255, 80, 80), 1,
                        cv2.LINE_4)

            img = QImage(cv2_img.data, cv2_img.shape[1], cv2_img.shape[0], QImage.Format_RGB888).rgbSwapped()
            if not (img.width() == self.ImageScreenWidth and img.height() == self.ImageScreenHeight):
                self.ImageScreen.resize(img.width(), img.height())
                self.ImageScreenWidth = img.width()
                self.ImageScreenHeight = img.height()

            if self.VideoButtion.text().endswith("on"):
                # noinspection PyArgumentList
                self.ImageScreen.setPixmap(QPixmap.fromImage(img))
        mp.terminate()

    @pyqtSlot()
    def clickCameraButton(self):
        if self.CameraButton.text().endswith("off"):
            self.VideoButtion.setEnabled(False)
            self.CameraButton.setText(self.CameraButton.text().replace("off", "on"))
            self.threadPool.start(self.cameraRunModels)
        else:
            self.CameraButton.setText(self.CameraButton.text().replace("on", "off"))
            self.cap.release()
            self.ImageScreen.clear()
            self.VideoButtion.setEnabled(True)

    # noinspection PyArgumentList
    @pyqtSlot()
    def cameraRunModels(self):
        self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

        mp = ModelProcesses()
        mp.start()
        last_time = time.time()

        while self.cap.isOpened():
            read_succ, cv2_img = self.cap.read()
            if not read_succ:
                break

            pil_img = cv2_to_pil(cv2_img)
            mp.traffic_in.put(pil_img, True)
            mp.sign_in.put(pil_img, True)

            sign_pred = mp.sign_out.get(True)
            opencvPaintPrediction(sign_pred, cv2_img)
            (traffic_pred, tensor_shape) = mp.traffic_out.get(True)
            yoloPaintPrediction(traffic_pred, tensor_shape, cv2_img, TRAFFIC_NAMES, TRAFFIC_COLOR)

            current_latency = (time.time() - last_time) * 1000
            last_time = time.time()
            cv2.putText(cv2_img, "FPS:%.1f" % (1000 / current_latency), (0, 15), FONT, 0.5, (255, 80, 80), 1,
                        cv2.LINE_4)

            img = QImage(cv2_img.data, cv2_img.shape[1], cv2_img.shape[0], QImage.Format_RGB888).rgbSwapped()

            if not (img.width() == self.ImageScreenWidth and img.height() == self.ImageScreenHeight):
                self.ImageScreen.resize(img.width(), img.height())
                self.ImageScreenWidth = img.width()
                self.ImageScreenHeight = img.height()

            if self.CameraButton.text().endswith("on"):
                self.ImageScreen.setPixmap(QPixmap.fromImage(img))
        mp.terminate()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = TrafficSystemGUI()
    ui.show()
    sys.exit(app.exec_())
