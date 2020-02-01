import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowObj4EfficientdetModelParam(QtWidgets.QWidget):

    backward_4_efficientdet_valdata_param = QtCore.pyqtSignal();
    forward_hyper_param = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'EfficientDet - Model Param'
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 400
        self.load_cfg();
        self.initUI()
        

    def load_cfg(self):
        if(os.path.isfile("obj_4_efficientdet.json")):
            with open('obj_4_efficientdet.json') as json_file:
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


        self.l2 = QLabel(self);
        self.l2.setText("1. Image size (model input size):");
        self.l2.move(20, 20);

        self.e2 = QLineEdit(self)
        self.e2.move(270, 20);
        self.e2.setText(self.system["image_size"]);
        self.e2.resize(200, 25);



        self.l3 = QLabel(self);
        self.l3.setText("2. Use GPU:");
        self.l3.move(20, 60);

        self.cb3 = QComboBox(self);
        self.use_gpu = ["yes", "no"];
        self.cb3.addItems(self.use_gpu);
        index = self.cb3.findText(self.system["use_gpu"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb3.setCurrentIndex(index)
        self.cb3.move(120, 60);
        self.cb3.activated.connect(self.gpu)


        self.l4 = QLabel(self);
        self.l4.setText("3. GPU devices (Comma separated):");
        self.l4.move(20, 100);


        self.e4 = QLineEdit(self)
        self.e4.move(270, 100);
        self.e4.setText(self.system["devices"]);
        self.e4.resize(200, 25);



        if(self.system["use_gpu"] == "no"):
            self.l4.setEnabled(False);
            self.e4.setEnabled(False); 



    def gpu(self):
        if str(self.cb3.currentText()) == "no":
            self.l4.setEnabled(False);
            self.e4.setEnabled(False);

        else:
            self.l4.setEnabled(True);
            self.e4.setEnabled(True);




    def forward(self):
        self.system["image_size"] = self.e2.text()
        self.system["use_gpu"] = str(self.cb3.currentText())
        self.system["devices"] = self.e4.text();

        with open('obj_4_efficientdet.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.forward_hyper_param.emit();

    def backward(self):
        self.system["image_size"] = self.e2.text()
        self.system["use_gpu"] = str(self.cb3.currentText())
        self.system["devices"] = self.e4.text();

        with open('obj_4_efficientdet.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.backward_4_efficientdet_valdata_param.emit();


'''
app = QApplication(sys.argv)
screen = WindowObj4EfficientdetModelParam()
screen.show()
sys.exit(app.exec_())
'''