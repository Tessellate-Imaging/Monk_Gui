import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowObj1GluoncvFinetuneModelParam(QtWidgets.QWidget):

    backward_1_gluoncv_finetune_data_param = QtCore.pyqtSignal();
    forward_hyper_param = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'GluonCV Finetune - Model Param'
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 400
        self.load_cfg();
        self.initUI()
        

    def load_cfg(self):
        if(os.path.isfile("obj_1_gluoncv_finetune.json")):
            with open('obj_1_gluoncv_finetune.json') as json_file:
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
        self.models = ["Select", "ssd_300_vgg16_atrous_coco", "ssd_300_vgg16_atrous_voc", "ssd_512_vgg16_atrous_coco", 
                        "ssd_512_vgg16_atrous_voc", "ssd_512_resnet50_v1_coco", "ssd_512_resnet50_v1_voc",
                        "ssd_512_mobilenet1.0_voc", "ssd_512_mobilenet1.0_coco", "yolo3_darknet53_voc", "yolo3_darknet53_coco",
                        "yolo3_mobilenet1.0_voc", "yolo3_mobilenet1.0_coco"];
        self.cb1.addItems(self.models);
        index = self.cb1.findText(self.system["model"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb1.setCurrentIndex(index)
        self.cb1.move(120, 20);


        self.l2 = QLabel(self);
        self.l2.setText("2. Use Pretrained Model:");
        self.l2.move(20, 60);

        self.cb2 = QComboBox(self);
        self.use_pretrained = ["yes", "no"];
        self.cb2.addItems(self.use_pretrained);
        index = self.cb2.findText(self.system["use_pretrained"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb2.setCurrentIndex(index)
        self.cb2.move(200, 60);


        self.l3 = QLabel(self);
        self.l3.setText("3. Use GPU:");
        self.l3.move(20, 100);

        self.cb3 = QComboBox(self);
        self.use_gpu = ["yes", "no"];
        self.cb3.addItems(self.use_gpu);
        index = self.cb3.findText(self.system["use_gpu"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb3.setCurrentIndex(index)
        self.cb3.move(120, 100);
        self.cb3.activated.connect(self.gpu)


        self.l4 = QLabel(self);
        self.l4.setText("4. GPU devices (Comma separated):");
        self.l4.move(20, 140);


        self.e4 = QLineEdit(self)
        self.e4.move(270, 140);
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
        self.system["model"] = str(self.cb1.currentText())
        self.system["use_pretrained"] = str(self.cb2.currentText())
        self.system["use_gpu"] = str(self.cb3.currentText())
        self.system["devices"] = self.e4.text();

        with open('obj_1_gluoncv_finetune.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.forward_hyper_param.emit();

    def backward(self):
        self.system["model"] = str(self.cb1.currentText())
        self.system["use_pretrained"] = str(self.cb2.currentText())
        self.system["use_gpu"] = str(self.cb3.currentText())
        self.system["devices"] = self.e4.text();

        with open('obj_1_gluoncv_finetune.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.backward_1_gluoncv_finetune_data_param.emit();

