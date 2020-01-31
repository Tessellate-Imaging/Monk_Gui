import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowObj7Yolov3(QtWidgets.QWidget):

    backward_main = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Object Detection - Yolo-v3'
        self.left = 300
        self.top = 300
        self.width = 500
        self.height = 220
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);

        # Forward
        self.b1 = QPushButton('Next', self)
        self.b1.move(300,180)
        self.b1.clicked.connect(self.forward)
        self.b1.setEnabled(False)

        # Backward
        self.b2 = QPushButton('Back', self)
        self.b2.move(200,180)
        self.b2.clicked.connect(self.backward)



        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(400,180)
        self.b3.clicked.connect(self.close)

    def forward(self):
    	pass;

    def backward(self):
    	self.backward_main.emit();

