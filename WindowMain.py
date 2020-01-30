import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *




class WindowMain(QtWidgets.QWidget):

    forward_classification = QtCore.pyqtSignal();
    forward_detection = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Monk'
        self.left = 100
        self.top = 100
        self.width = 300
        self.height = 200
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);

        # Image classification
        self.b1 = QPushButton('Image Classification', self)
        self.b1.move(30,10)
        self.b1.resize(200,25)
        self.b1.clicked.connect(self.image_classification);

        # Object Detection
        self.b2 = QPushButton('Object Detection', self)
        self.b2.move(30,50)
        self.b2.resize(200,25)
        self.b2.clicked.connect(self.object_detection);

        # Exit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(100,150)
        self.b3.clicked.connect(self.close)



    def image_classification(self):
        self.forward_classification.emit();

    def object_detection(self):
        self.forward_detection.emit();