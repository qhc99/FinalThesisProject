import sys
import cv2
import time
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torch.backends.cudnn as cudnn

from scripts import cv2_to_pil, pil_to_cv2, yoloPredict, signPredict, yolo_paint_interested_result, FONT
# from utils.torch_utils import select_device
# from predict import load_model, get_names, get_colors
from multiprocessing import Process, Queue

cudnn.benchmark = True


class TrafficSystemGUI(QWidget):

    # noinspection PyArgumentList
    def __init__(self):
        super().__init__()
        self.initMainWindow()
        self.initImageGroup()
        self.initButtons()
        self.threadPool = QThreadPool()

    def initMainWindow(self):
        self.__height = 719
        self.__width = 1097
        self.__top = 180
        self.__left = 400
        self.__title = "交通路况系统"
        self.setWindowTitle(self.__title)
        self.setGeometry(self.__left, self.__top, self.__width, self.__height)
        self.setFixedWidth(self.__width)
        self.setFixedHeight(self.__height)
        self.setWindowIcon(QIcon("resources/hohai.png"))

    def initImageGroup(self):
        self.ImageBox = QGraphicsView(self)
        self.ImageBox.setGeometry(QRect(10, 10, 941, 671))

        self.ImageScreen = QLabel(self)
        rect = QRect(10, 10, 200, 200)
        self.ImageScreen.setGeometry(rect)
        self.ImageScreen.setText("")

        self.ImageScreen.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        self.ImageScreenTop = rect.top()
        self.ImageScreenLeft = rect.left()
        self.ImageScreenWidth = rect.width()
        self.ImageScreenHeight = rect.height()

    # noinspection PyArgumentList,PyUnresolvedReferences
    def initButtons(self):
        self.VideoButtion = QPushButton(self)
        self.VideoButtion.setGeometry(QRect(970, 510, 111, 51))
        self.VideoButtion.setText("视频:off")
        self.VideoButtion.clicked.connect(self.clickVideoButton)

        self.CameraButton = QPushButton(self)
        self.CameraButton.setGeometry(QRect(970, 630, 111, 51))
        self.CameraButton.setText("摄像头:off")
        self.CameraButton.clicked.connect(self.clickCameraButton)

        self.cap = None

    @pyqtSlot()
    def clickVideoButton(self):
        if self.VideoButtion.text().endswith("off"):
            self.CameraButton.setEnabled(False)
            dialog = QFileDialog(self)
            selected_info = dialog.getOpenFileName(caption="选择视频文件")
            file_path = selected_info[0]
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

        while self.cap.isOpened():
            read_succ, img = self.cap.read()
            if not read_succ:
                break

            yolo_pred, yolo_tensor_img = yoloPredict(img)
            yolo_paint_interested_result(yolo_pred, yolo_tensor_img, img)

            img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped()
            if not (img.width() == self.ImageScreenWidth and img.height() == self.ImageScreenHeight):
                self.ImageScreen.resize(img.width(), img.height())
                self.ImageScreenWidth = img.width()
                self.ImageScreenHeight = img.height()

            self.ImageScreen.setPixmap(QPixmap.fromImage(img))

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

        yolo_in_queue = Queue()
        yolo_out_queue = Queue()
        yolo_process = Process(target=yoloPredict, args=(yolo_in_queue, yolo_out_queue))
        yolo_process.start()

        sign_in_queue = Queue()
        sign_out_queue = Queue()
        sign_process = Process(target=signPredict, args=(sign_in_queue, sign_out_queue))
        sign_process.start()

        last_time = time.time()

        while self.cap.isOpened():
            read_succ, img = self.cap.read()
            if not read_succ:
                break

            pil_img = cv2_to_pil(img)

            yolo_in_queue.put(pil_img, True)
            sign_in_queue.put(pil_img, True)

            yolo_painted = yolo_out_queue.get(True)
            yolo_painted = pil_to_cv2(yolo_painted)
            sign_pred = sign_out_queue.get(True)

            if len(sign_pred) > 0:
                for (x, y, w, h) in sign_pred:
                    cv2.rectangle(yolo_painted, (x, y), (x + w, y + h), (255, 0, 0), 2)

            current_latency = (time.time() - last_time) * 1000
            last_time = time.time()
            cv2.putText(yolo_painted, "FPS:%.1f" % (1000 / current_latency), (0, 20), FONT, 0.5, (255, 80, 80), 1,
                        cv2.LINE_4)

            res_img = QImage(yolo_painted.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped()

            if not (res_img.width() == self.ImageScreenWidth and res_img.height() == self.ImageScreenHeight):
                self.ImageScreen.resize(res_img.width(), res_img.height())
                self.ImageScreenWidth = res_img.width()
                self.ImageScreenHeight = res_img.height()

            self.ImageScreen.setPixmap(QPixmap.fromImage(res_img))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = TrafficSystemGUI()
    ui.show()
    sys.exit(app.exec_())
