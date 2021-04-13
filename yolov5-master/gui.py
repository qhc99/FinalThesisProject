import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import pyqtSlot


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.initWindow()
        self.initLabel()
        self.initButton()

    def initWindow(self):
        self.__height = 480
        self.__width = 640
        self.__top = 120
        self.__left = 300
        self.__title = "交通路况系统"
        self.setWindowTitle(self.__title)
        self.setGeometry(self.__left, self.__top, self.__width, self.__height)
        self.setWindowIcon(QIcon(""))

    def initButton(self):
        # attach button
        self.__button = QPushButton('PyQt5 button', self)
        self.__button.setToolTip('This is an example button')
        self.__button.move(100, 70)
        self.__button.clicked.connect(self.on_click)

    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')

    def initLabel(self):
        # attach button
        self.__label = QLabel(self)
        self.__label.setText("Hello World")
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.__label.setFont(font)
        self.__label.move(50, 20)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
