import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowObj3MxrcnnDataPreproc(QtWidgets.QWidget):

    backward_3_mxrcnn_data_param = QtCore.pyqtSignal();
    forward_model_param = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Mxrcnn - Data Preprocessing'
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 400
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
        self.b1.move(300,350)
        self.b1.clicked.connect(self.forward)

        # Backward
        self.b2 = QPushButton('Back', self)
        self.b2.move(200,350)
        self.b2.clicked.connect(self.backward)

        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(400,350)
        self.b3.clicked.connect(self.close)


        self.l1 = QLabel(self);
        self.l1.setText("1. Image short side :");
        self.l1.move(20, 20);

        self.e1 = QLineEdit(self)
        self.e1.move(170, 20);
        self.e1.setText(self.system["img_short_side"]);
        self.e1.resize(200, 25);


        self.l2 = QLabel(self);
        self.l2.setText("2. Image long side :");
        self.l2.move(20, 70);

        self.e2 = QLineEdit(self)
        self.e2.move(170, 70);
        self.e2.setText(self.system["img_long_side"]);
        self.e2.resize(200, 25);


        self.l3 = QLabel(self);
        self.l3.setText("3. Normalization mean :");
        self.l3.move(20, 120);

        self.e3 = QLineEdit(self)
        self.e3.move(190, 120);
        self.e3.setText(self.system["mean"]);
        self.e3.resize(200, 25);


        self.l4 = QLabel(self);
        self.l4.setText("4. Normalization std :");
        self.l4.move(20, 170);

        self.e4 = QLineEdit(self)
        self.e4.move(190, 170);
        self.e4.setText(self.system["std"]);
        self.e4.resize(200, 25);





    def forward(self):
        self.system["img_short_side"] = str(self.e1.text())
        self.system["img_long_side"] = str(self.e2.text())
        self.system["mean"] = str(self.e3.text())
        self.system["std"] = str(self.e4.text())

        with open('obj_3_mxrcnn.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.forward_model_param.emit();

    def backward(self):
        self.system["img_short_side"] = str(self.e1.text())
        self.system["img_long_side"] = str(self.e2.text())
        self.system["mean"] = str(self.e3.text())
        self.system["std"] = str(self.e4.text())

        with open('obj_3_mxrcnn.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.backward_3_mxrcnn_data_param.emit();


'''
app = QApplication(sys.argv)
screen = WindowObj3MxrcnnDataPreproc()
screen.show()
sys.exit(app.exec_())
'''