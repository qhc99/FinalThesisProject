import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from scripts import paint, yoloPredict, signPredict
import cv2
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device
from predict import load_model, get_names, get_colors

# 1: 639
# 3: 381
MODEL_NUM = 3
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

        self.CameraButton = QPushButton(self)
        self.CameraButton.setGeometry(QRect(970, 630, 111, 51))
        self.CameraButton.setText("摄像头:off")
        self.CameraButton.clicked.connect(self.clickCameraButton)

        self.cap = None

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
        sign_pred = None
        yolo_pred, yolo_tensor_img = None, None

        # warm up
        if self.cap.isOpened():
            _, img = self.cap.read()
            sign_pred = signPredict(img)
            yolo_pred, yolo_tensor_img = yoloPredict(img)

        # process trick
        process_yolo = True

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
            img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped()

            if not(img.width() == self.ImageScreenWidth and img.height() == self.ImageScreenHeight):
                self.ImageScreen.resize(img.width(), img.height())
                self.ImageScreenWidth = img.width()
                self.ImageScreenHeight = img.height()

            self.ImageScreen.setPixmap(QPixmap.fromImage(img))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = TrafficSystemGUI()
    ui.show()
    sys.exit(app.exec_())
