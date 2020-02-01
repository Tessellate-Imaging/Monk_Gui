import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowObj3MxrcnnHyperParam(QtWidgets.QWidget):

    backward_model_param = QtCore.pyqtSignal();
    forward_train = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Mxrcnn - Hyper Param'
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 220
        self.load_cfg();
        self.initUI()


    def load_cfg(self):
        if(os.path.isfile("obj_3_mxrcnn.json")):
            with open('obj_3_mxrcnn.json') as json_file:
                self.system = json.load(json_file)


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);

        # Forward
        self.b1 = QPushButton('Next', self)
        self.b1.move(300,180)
        self.b1.clicked.connect(self.forward)

        # Backward
        self.b2 = QPushButton('Back', self)
        self.b2.move(200,180)
        self.b2.clicked.connect(self.backward)


        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(400,180)
        self.b3.clicked.connect(self.close)


        self.l1 = QLabel(self);
        self.l1.setText("1. learning_rate:");
        self.l1.move(20, 20);

        self.e1 = QLineEdit(self)
        self.e1.move(150, 20);
        self.e1.setText(self.system["lr"]);
        self.e1.resize(200, 25);

        self.l2 = QLabel(self);
        self.l2.setText("2. Learning rate decay epochs (Comma separated):");
        self.l2.move(20,70);

        self.e2 = QLineEdit(self)
        self.e2.move(30, 90);
        self.e2.setText(self.system["lr_decay_epoch"]);


    def forward(self):
        self.system["lr"] = self.e1.text();
        self.system["lr_decay_epoch"] = self.e2.text();

        with open('obj_3_mxrcnn.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.forward_train.emit();

    def backward(self):
        self.system["lr"] = self.e1.text();
        self.system["lr_decay_epoch"] = self.e2.text();

        with open('obj_3_mxrcnn.json', 'w') as outfile:
            json.dump(self.system, outfile)
        
        self.backward_model_param.emit();


'''
app = QApplication(sys.argv)
screen = WindowObj3MxrcnnHyperParam()
screen.show()
sys.exit(app.exec_())
'''