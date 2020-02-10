import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainQuickModelParam(QtWidgets.QWidget):

    forward_train = QtCore.pyqtSignal();
    backward_data_param = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Model Params'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 400
        self.initUI()

    def cfg_setup(self):
        with open('base_classification.json') as json_file:
            self.system = json.load(json_file)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);


        # Backward
        self.b1 = QPushButton('Back', self)
        self.b1.move(200, 350)
        self.b1.clicked.connect(self.backward)

        # Backward
        self.b2 = QPushButton('Next', self)
        self.b2.move(300, 350)
        self.b2.clicked.connect(self.forward)

        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(400, 350)
        self.b3.clicked.connect(self.close)


        self.l1 = QLabel(self);
        self.l1.setText("1. Select Model:");
        self.l1.move(20, 20);


        self.cb1 = QComboBox(self);
        self.cb1.move(150, 20);
        self.cb1.activated.connect(self.select_model);

        self.cb2 = QComboBox(self);
        self.cb2.move(150, 20);
        self.cb2.activated.connect(self.select_model);

        self.cb3 = QComboBox(self);
        self.cb3.move(150, 20);
        self.cb3.activated.connect(self.select_model);


        self.l2 = QLabel(self);
        self.l2.setText("2. Freeze base model:");
        self.l2.move(20, 100);


        self.cb4 = QComboBox(self);
        self.cb4.move(200, 100);
        self.cb4.activated.connect(self.select_freeze_base);
        self.freeze_base = ["yes", "no"];
        self.cb4.addItems(self.freeze_base);
        index = self.cb4.findText(self.system["freeze_base_model"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.cb4.setCurrentIndex(index)


        if(self.system["backend"] == "Mxnet-1.5.1"):
            set1 = ["alexnet", "darknet53", "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201", "InceptionV3", "MobileNet1.0", "MobileNet0.75", 
                        "MobileNet0.25", "MobileNet0.5", "ResNet18_v1", "ResNet34_v1", "ResNet50_v1", "ResNet101_v1", "ResNet152_v1", "ResNext50_32x4d", 
                        "ResNext101_32x4d", "ResNext101_64x4d_v1", "SE_ResNext50_32x4d", "SE_ResNext101_32x4d", "SE_ResNext101_64x4d", "SENet_154", 
                        "VGG11", "VGG13", "VGG16", "VGG19", "VGG11_bn", "VGG13_bn", "VGG16_bn", "VGG19_bn", "ResNet18_v2", "ResNet34_v2", 
                        "ResNet50_v2", "ResNet101_v2", "ResNet152_v2"];
            set2 = ["MobileNetV2_1.0", "MobileNetV2_0.75", "MobileNetV2_0.5", "MobileNetV2_0.25", "SqueezeNet1.0", "SqueezeNet1.1", "MobileNetV3_Large", "MobileNetV3_Small"];
            set3 = ["ResNet18_v1b", "ResNet34_v1b", "ResNet50_v1b", "ResNet50_v1b_gn", "ResNet101_v1b", "ResNet152_v1b", "ResNet50_v1c", 
                        "ResNet101_v1c", "ResNet152_v1c", "ResNet50_v1d", "ResNet101_v1d", "ResNet152_v1d", "ResNet18_v1d", "ResNet34_v1d", 
                        "ResNet50_v1d", "ResNet101_v1d", "ResNet152_v1d", "resnet18_v1b_0.89", "resnet50_v1d_0.86", "resnet50_v1d_0.48", 
                        "resnet50_v1d_0.37", "resnet50_v1d_0.11", "resnet101_v1d_0.76", "resnet101_v1d_0.73", "Xception"];
            combined_list = set1+set2+set3
            combined_list_lower = list(map(str.lower, combined_list))

            self.cb1.addItems(combined_list_lower);
            index = self.cb1.findText(self.system["model"], QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.cb1.setCurrentIndex(index)

            self.cb1.show();
            self.cb2.hide();
            self.cb3.hide();

            
        elif(self.system["backend"] == "Pytorch-1.3.1"):
            set1 = ["alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]
            set2 = ["densenet121", "densenet161", "densenet169", "densenet201"]
            set3 = ["googlenet", "inception_v3", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d",
                        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0, shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "wide_resnet101_2", "wide_resnet50_2"]
            set4 = ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3", "mobilenet_v2", "squeezenet1_0", "squeezenet1_1"]
            combined_list = set1+set2+set3+set4
            combined_list_lower = list(map(str.lower, combined_list))

            self.cb2.addItems(combined_list_lower);
            index = self.cb2.findText(self.system["model"], QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.cb2.setCurrentIndex(index)

            self.cb1.hide();
            self.cb2.show();
            self.cb3.hide();


        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            set1 = ["mobilenet", "densenet121", "densenet169", "densenet201", "inception_v3", 
                        "inception_resnet_v3", "mobilenet_v2", "nasnet_mobile", "nasnet_large", "resnet50",
                        "resnet101", "resnet152", "resnet50_v2", "resnet101_v2", "resnet152_v2", "vgg16",
                        "vgg19", "xception"];
            combined_list = set1
            combined_list_lower = list(map(str.lower, combined_list))

            self.cb3.addItems(combined_list_lower);
            index = self.cb3.findText(self.system["model"], QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.cb3.setCurrentIndex(index)

            self.cb1.hide();
            self.cb2.hide();
            self.cb3.show();



    def select_freeze_base(self):
        self.system["freeze_base_model"] = self.cb4.currentText();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)



    def select_model(self):
        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.system["model"] = self.cb1.currentText();

        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.system["model"] = self.cb2.currentText();            

        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.system["model"] = self.cb3.currentText();

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
            



    def forward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_train.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_data_param.emit();



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainQuickModelParam()
screen.show()
sys.exit(app.exec_())
'''




