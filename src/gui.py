import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class TrafficSystemGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.initMainWindow()
        self.initImageBox()
        self.initButtons()
        self.initFPSGroup()

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
        self.ImageBox = QGraphicsView(self)
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
        self.CameraButton.setText("摄像头")

        self.VideoButtion = QPushButton(self.widget)
        self.VideoButtion.setObjectName("VideoButtion")
        self.buttonsLayout.addWidget(self.VideoButtion)
        self.VideoButtion.setText("视频")

        self.FileButton = QPushButton(self.widget)
        self.FileButton.setObjectName("FileButton")
        self.buttonsLayout.addWidget(self.FileButton)
        self.FileButton.setText("文件")
        
    def initFPSGroup(self):
        self.widget1 = QWidget(self)
        self.widget1.setGeometry(QRect(440, 500, 158, 24))
        self.widget1.setObjectName("widget1")

        self.FPS_StatusLayout = QHBoxLayout(self.widget1)
        self.FPS_StatusLayout.setContentsMargins(0, 0, 0, 0)
        self.FPS_StatusLayout.setSpacing(30)
        self.FPS_StatusLayout.setObjectName("horizontalLayout")

        self.FPSSwitch = QRadioButton(self.widget1)
        self.FPSSwitch.setObjectName("FPSSwitch")
        self.FPS_StatusLayout.addWidget(self.FPSSwitch)
        self.FPSSwitch.setText("FPS监测")

        self.FPSTextLabel = QLabel(self.widget1)
        self.FPSTextLabel.setObjectName("FPSTextLabel")
        self.FPS_StatusLayout.addWidget(self.FPSTextLabel)
        self.FPSTextLabel.setText("FPS:")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = TrafficSystemGUI()
    ui.show()
    sys.exit(app.exec_())
