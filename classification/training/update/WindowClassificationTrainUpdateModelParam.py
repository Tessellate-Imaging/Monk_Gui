import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateModelParam(QtWidgets.QWidget):

    forward_layer_param = QtCore.pyqtSignal();
    forward_analyse_model_list = QtCore.pyqtSignal();
    forward_analyse_use_pretrained = QtCore.pyqtSignal();
    forward_analyse_freeze_base = QtCore.pyqtSignal();
    forward_analyse_freeze_layer = QtCore.pyqtSignal();
    backward_transform_param = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Update Model Params'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 600
        self.initUI()

    def cfg_setup(self):
        with open('base_classification.json') as json_file:
            self.system = json.load(json_file)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);


        # Backward
        self.b1 = QPushButton('Back', self)
        self.b1.move(600,550)
        self.b1.clicked.connect(self.backward)

        # Forward
        self.b2 = QPushButton('Next', self)
        self.b2.move(700,550)
        self.b2.clicked.connect(self.forward)

        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(800,550)
        self.b3.clicked.connect(self.close)


        self.l1 = QLabel(self);
        self.l1.setText("1. Model List:");
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

            self.cb1.hide();
            self.cb2.hide();
            self.cb3.show();

        self.btn3 = QPushButton('Autotune this hyperparam', self)
        self.btn3.move(350, 20)
        self.btn3.clicked.connect(self.analyse_model_list)


        self.l4 = QLabel(self);
        self.l4.setText("2. Use Gpu:");
        self.l4.move(20, 100);


        self.cb4 = QComboBox(self);
        self.cb4.move(120, 100);
        self.cb4.activated.connect(self.select_gpu);
        self.items = ["True", "False"];
        self.cb4.addItems(self.items);
        index = self.cb4.findText(self.system["update"]["use_gpu"]["value"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.cb4.setCurrentIndex(index)


        self.l5 = QLabel(self);
        self.l5.setText("3. Use Pretrained Weights:");
        self.l5.move(20, 150);


        self.cb5 = QComboBox(self);
        self.cb5.move(210, 150);
        self.cb5.activated.connect(self.select_pretrained);
        self.items = ["True", "False"];
        self.cb5.addItems(self.items);
        index = self.cb5.findText(self.system["update"]["use_pretrained"]["value"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.cb5.setCurrentIndex(index)

        self.btn5 = QPushButton('Autotune this hyperparam', self)
        self.btn5.move(310, 150)
        self.btn5.clicked.connect(self.analyse_use_pretrained)



        self.l6 = QLabel(self);
        self.l6.setText("4. Freeze base network:");
        self.l6.move(20, 200);


        self.cb6 = QComboBox(self);
        self.cb6.move(200, 200);
        self.cb6.activated.connect(self.select_freeze_base);
        self.items = ["True", "False"];
        self.cb6.addItems(self.items);
        index = self.cb6.findText(self.system["update"]["freeze_base_network"]["value"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.cb6.setCurrentIndex(index)

        self.btn6 = QPushButton('Autotune this hyperparam', self)
        self.btn6.move(300, 200)
        self.btn6.clicked.connect(self.analyse_freeze_base)



        self.l7 = QLabel(self);
        self.l7.setText("5. Number of layers to freeze:");
        self.l7.move(20, 250);

        self.e7 = QLineEdit(self)
        self.e7.move(240, 250);
        self.e7.setText(self.system["update"]["freeze_layers"]["value"]);


        self.btn7 = QPushButton('Autotune this hyperparam', self)
        self.btn7.move(420, 250)
        self.btn7.clicked.connect(self.analyse_freeze_layer)











    def select_freeze_base(self):
        self.system["update"]["freeze_base_network"]["active"] = True;
        self.system["update"]["freeze_base_network"]["value"] = self.cb4.currentText();



    def select_pretrained(self):
        self.system["update"]["use_pretrained"]["active"] = True;
        self.system["update"]["use_pretrained"]["value"] = self.cb4.currentText();



    def select_gpu(self):
        self.system["update"]["use_gpu"]["active"] = True;
        self.system["update"]["use_gpu"]["value"] = self.cb3.currentText();


        

    def select_model(self):
        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.system["update"]["model_name"]["active"] = True;
            self.system["update"]["model_name"]["value"] = self.cb1.currentText();

        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.system["update"]["model_name"]["active"] = True;
            self.system["update"]["model_name"]["value"] = self.cb2.currentText();            

        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.system["update"]["model_name"]["active"] = True;
            self.system["update"]["model_name"]["value"] = self.cb3.currentText();

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def analyse_model_list(self):
        if(self.e7.text() != "None"):
            self.system["update"]["freeze_layers"]["active"] = True;
            self.system["update"]["freeze_layers"]["value"] = self.e7.text();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_analyse_model_list.emit();


    def analyse_use_pretrained(self):
        if(self.e7.text() != "None"):
            self.system["update"]["freeze_layers"]["active"] = True;
            self.system["update"]["freeze_layers"]["value"] = self.e7.text();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_analyse_use_pretrained.emit();

    def analyse_freeze_base(self):
        if(self.e7.text() != "None"):
            self.system["update"]["freeze_layers"]["active"] = True;
            self.system["update"]["freeze_layers"]["value"] = self.e7.text();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_analyse_freeze_base.emit();

    def analyse_freeze_layer(self):
        if(self.e7.text() != "None"):
            self.system["update"]["freeze_layers"]["active"] = True;
            self.system["update"]["freeze_layers"]["value"] = self.e7.text();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_analyse_freeze_layer.emit();


    def forward(self):
        if(self.e7.text() != "None"):
            self.system["update"]["freeze_layers"]["active"] = True;
            self.system["update"]["freeze_layers"]["value"] = self.e7.text();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_layer_param.emit();


    def backward(self):
        if(self.e7.text() != "None"):
            self.system["update"]["freeze_layers"]["active"] = True;
            self.system["update"]["freeze_layers"]["value"] = self.e7.text();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_transform_param.emit();



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateModelParam()
screen.show()
sys.exit(app.exec_())
'''