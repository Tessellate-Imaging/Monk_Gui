import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowObj7Yolov3ModelParam(QtWidgets.QWidget):

    backward_7_yolov3_valdata_param = QtCore.pyqtSignal();
    forward_hyper_param = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Yolo V3 - Model Param'
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 400
        self.load_cfg();
        self.initUI()
        

    def load_cfg(self):
        if(os.path.isfile("obj_7_yolov3.json")):
            with open('obj_7_yolov3.json') as json_file:
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
        self.l1.setText("1. Model :");
        self.l1.move(20, 20);

        self.cb1 = QComboBox(self);
        self.models = ["yolov3", "yolov3s", "yolov3-spp", "yolov3-spp3", "yolov3-tiny",
                        "yolov3-spp-matrix", "csresnext50-panet-spp"];
        self.cb1.addItems(self.models);
        index = self.cb1.findText(self.system["model"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb1.setCurrentIndex(index)
        self.cb1.move(120, 20);


        self.l3 = QLabel(self);
        self.l3.setText("2. Use GPU:");
        self.l3.move(20, 70);

        self.cb3 = QComboBox(self);
        self.use_gpu = ["yes", "no"];
        self.cb3.addItems(self.use_gpu);
        index = self.cb3.findText(self.system["use_gpu"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb3.setCurrentIndex(index)
        self.cb3.move(120, 70);
        self.cb3.activated.connect(self.gpu)


        self.l4 = QLabel(self);
        self.l4.setText("3. GPU devices (Comma separated):");
        self.l4.move(20, 110);


        self.e4 = QLineEdit(self)
        self.e4.move(270, 110);
        self.e4.setText(self.system["devices"]);
        self.e4.resize(200, 25);



    def gpu(self):
        if str(self.cb3.currentText()) == "no":
            self.l4.setEnabled(False);
            self.e4.setEnabled(False);

        else:
            self.l4.setEnabled(True);
            self.e4.setEnabled(True);
          



    def forward(self):
        self.system["model"] = str(self.cb1.currentText())
        self.system["use_gpu"] = str(self.cb3.currentText())
        self.system["devices"] = self.e4.text();

        with open('obj_7_yolov3.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.forward_hyper_param.emit();

    def backward(self):
        self.system["model"] = str(self.cb1.currentText())
        self.system["use_gpu"] = str(self.cb3.currentText())
        self.system["devices"] = self.e4.text();

        with open('obj_7_yolov3.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.backward_7_yolov3_valdata_param.emit();


'''
app = QApplication(sys.argv)
screen = WindowObj7Yolov3ModelParam()
screen.show()
sys.exit(app.exec_())
'''