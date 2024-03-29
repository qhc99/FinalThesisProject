from sys import argv, exit
from time import time

from cv2 import VideoCapture, LINE_4, putText, CAP_DSHOW
from torch.backends import cudnn
from torch.cuda import synchronize
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from globals import TRAFFIC_NAMES, TRAFFIC_COLOR, GPU_DEVICE, TRAFFIC_MODEL
from scripts import cv2_to_pil, opencvPaint, yoloPaint, FONT, CONFI_THRES, IOU_THRES, tensorShape, signPredict
from predict import img_resize, img_transform, non_max_suppression
from multiprocessing import Queue, Process

cudnn.benchmark = True


# noinspection PyAttributeOutsideInit,DuplicatedCode
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
        self.__height = 898
        self.__width = 1536
        self.__top = int(1080 / 2 - self.__height / 2)
        self.__left = int(1920 / 2 - self.__width / 2)
        self.__title = "交通路况系统"
        self.setWindowTitle(self.__title)
        self.setGeometry(self.__left, self.__top, self.__width, self.__height)
        self.setFixedWidth(self.__width)
        self.setFixedHeight(self.__height)
        self.setWindowIcon(QIcon("resources/hohai.png"))

    def initImageGroup(self):
        self.ImageBackGround = QGraphicsView(self)
        self.ImageBackGround.setGeometry(QRect(10, 10, 1152, 854))

        self.ImageScreen = QLabel(self)
        rect = QRect(10, 10, 200, 200)
        self.ImageScreen.setGeometry(rect)
        self.ImageScreen.setText("")

        self.ImageScreenTop = rect.top()
        self.ImageScreenLeft = rect.left()
        self.ImageScreenWidth = rect.width()
        self.ImageScreenHeight = rect.height()

    def imageBGMidPoint(self):
        bg = self.ImageBackGround.geometry()
        return bg.left() + int(bg.width() / 2), bg.top() + int(bg.height() / 2)

    def imageMidPoint(self):
        ig = self.ImageScreen.geometry()
        return ig.left() + int(ig.width() / 2), ig.top() + int(ig.height() / 2)

    def alignImage(self, img_mid: tuple, bg_mid: tuple):
        w_diff = bg_mid[0] - img_mid[0]
        h_diff = bg_mid[1] - img_mid[1]
        self.ImageScreenTop += h_diff
        self.ImageScreenLeft += w_diff
        self.ImageScreen.setGeometry(
            QRect(self.ImageScreenLeft, self.ImageScreenTop, self.ImageScreenWidth, self.ImageScreenHeight))

    # noinspection PyArgumentList,PyUnresolvedReferences
    def initButtons(self):
        self.VideoButton = QPushButton(self)
        self.VideoButton.setGeometry(QRect(1180, 640, 241, 101))
        self.VideoButton.setText("视频:off")
        self.VideoButton.clicked.connect(self.clickVideoButton)

        font = QFont()
        font.setPointSize(24)
        font.setFamily("楷体")
        self.VideoButton.setFont(font)

        self.CameraButton = QPushButton(self)
        self.CameraButton.setGeometry(QRect(1180, 770, 241, 101))
        self.CameraButton.setText("摄像头:off")
        self.CameraButton.clicked.connect(self.clickCameraButton)
        self.CameraButton.setFont(font)

        self.cap = None

    def initReadMe(self):
        self.ReadMeLabel = QLabel(self)
        self.ReadMeLabel.setGeometry(QRect(1180, 490, 351, 131))
        font = QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.ReadMeLabel.setFont(font)
        self.ReadMeLabel.setText("<html><head/><body><p>绿色:行人</p>"
                                 "<p>红色:汽车(轿车、卡车、摩托车)</p><p>蓝色:禁止标志</p></body></html>")

    @pyqtSlot()
    def clickVideoButton(self):
        if self.VideoButton.text().endswith("off"):
            self.CameraButton.setEnabled(False)
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            # noinspection PyTypeChecker
            file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "",
                                                       "All Files (*);;Python Files (*.py)", options=options)
            if len(file_path) > 0:
                self.VideoButton.setText(self.VideoButton.text().replace("off", "on"))
                self.threadPool.start(lambda: self.videoRunModels(video_path=file_path))
            else:
                self.CameraButton.setEnabled(True)
        else:
            self.VideoButton.setText(self.VideoButton.text().replace("on", "off"))

    @pyqtSlot()
    def videoRunModels(self, video_path):
        self.cap = VideoCapture(video_path)

        sign_in = Queue()
        sign_out = Queue()
        sign_process = Process(target=signPredict, args=(sign_in, sign_out))
        sign_process.start()

        last_time = time()

        while self.cap.isOpened():
            read_succ, cv2_img = self.cap.read()
            if not read_succ:
                break

            pil_img = cv2_to_pil(cv2_img)
            sign_in.put(pil_img, True)

            tensor_img = img_transform(img_resize(cv2_img, 480), GPU_DEVICE)
            yolo_pred = TRAFFIC_MODEL(tensor_img)[0]
            yolo_pred = non_max_suppression(yolo_pred, CONFI_THRES, IOU_THRES)
            yoloPaint(yolo_pred, tensorShape(tensor_img), cv2_img, TRAFFIC_NAMES, TRAFFIC_COLOR)

            sign_pred = sign_out.get(True)
            opencvPaint(sign_pred, cv2_img)

            synchronize()
            t = time()
            current_latency = (t - last_time) * 1000
            last_time = t
            putText(cv2_img, "FPS:%.1f" % (1000 / current_latency), (0, 15), FONT, 0.5, (255, 80, 80), 1, LINE_4)

            img = QImage(cv2_img.data, cv2_img.shape[1], cv2_img.shape[0], QImage.Format_RGB888).rgbSwapped()
            if not (img.width() == self.ImageScreenWidth and img.height() == self.ImageScreenHeight):
                self.ImageScreen.resize(img.width(), img.height())
                self.ImageScreenWidth = img.width()
                self.ImageScreenHeight = img.height()

            bg_mid = self.imageBGMidPoint()
            img_mid = self.imageMidPoint()
            if not (bg_mid == img_mid):
                self.alignImage(img_mid, bg_mid)

            if self.VideoButton.text().endswith("on"):
                # noinspection PyArgumentList
                self.ImageScreen.setPixmap(QPixmap.fromImage(img))
            else:
                break
        self.ImageScreen.clear()
        self.cap.release()
        sign_process.terminate()
        self.CameraButton.setEnabled(True)

    @pyqtSlot()
    def clickCameraButton(self):
        if self.CameraButton.text().endswith("off"):
            self.VideoButton.setEnabled(False)
            self.CameraButton.setText(self.CameraButton.text().replace("off", "on"))
            self.threadPool.start(self.cameraRunModels)
        else:
            self.CameraButton.setText(self.CameraButton.text().replace("on", "off"))

    # noinspection PyArgumentList,DuplicatedCode
    @pyqtSlot()
    def cameraRunModels(self):
        self.cap = VideoCapture(0 + CAP_DSHOW)

        sign_in = Queue()
        sign_out = Queue()
        sign_process = Process(target=signPredict, args=(sign_in, sign_out))
        sign_process.start()

        last_time = time()

        while self.cap.isOpened():
            read_success, cv2_img = self.cap.read()
            if not read_success:
                break

            pil_img = cv2_to_pil(cv2_img)
            sign_in.put(pil_img, True)

            tensor_img = img_transform(img_resize(cv2_img, 480), GPU_DEVICE)
            yolo_pred = TRAFFIC_MODEL(tensor_img)[0]
            yolo_pred = non_max_suppression(yolo_pred, CONFI_THRES, IOU_THRES)
            yoloPaint(yolo_pred, tensor_img.shape[2:], cv2_img, TRAFFIC_NAMES, TRAFFIC_COLOR)

            sign_pred = sign_out.get(True)
            opencvPaint(sign_pred, cv2_img)

            synchronize()
            t = time()
            current_latency = (t - last_time) * 1000
            last_time = t
            putText(cv2_img, "FPS:%.1f" % (1000 / current_latency), (0, 15), FONT, 0.5, (255, 80, 80), 1, LINE_4)

            img = QImage(cv2_img.data, cv2_img.shape[1], cv2_img.shape[0], QImage.Format_RGB888).rgbSwapped()

            if not (img.width() == self.ImageScreenWidth and img.height() == self.ImageScreenHeight):
                self.ImageScreen.resize(img.width(), img.height())
                self.ImageScreenWidth = img.width()
                self.ImageScreenHeight = img.height()

            bg_mid = self.imageBGMidPoint()
            img_mid = self.imageMidPoint()
            if not (bg_mid == img_mid):
                self.alignImage(img_mid, bg_mid)

            if self.CameraButton.text().endswith("on"):
                self.ImageScreen.setPixmap(QPixmap.fromImage(img))
            else:
                break
        self.cap.release()
        self.ImageScreen.clear()
        sign_process.terminate()
        self.VideoButton.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(argv)
    ui = TrafficSystemGUI()
    ui.show()
    exit(app.exec_())
