import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateAnalysisModelList(QtWidgets.QWidget):

    forward_visualize = QtCore.pyqtSignal();
    backward_model_param = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Analyse Base Models'.format(self.system["experiment"])
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
        self.l1.setText("1. Analysis name:");
        self.l1.move(20, 20);

        self.e1 = QLineEdit(self)
        self.e1.move(170, 20);
        self.e1.resize(200, 25);
        self.e1.setText(self.system["analysis"]["model_list"]["analysis_name"]);


        self.l2 = QLabel(self);
        self.l2.setText("2. Percent Data:");
        self.l2.move(420, 20);

        self.e2 = QLineEdit(self)
        self.e2.move(570, 20);
        self.e2.setText(self.system["analysis"]["model_list"]["percent"]);



        self.l3 = QLabel(self);
        self.l3.setText("3. Models list:");
        self.l3.move(20, 70);

        self.e3 = QLineEdit(self)
        self.e3.move(170, 70);
        self.e3.setText(self.system["analysis"]["model_list"]["list"]);


        self.l4 = QLabel(self);
        self.l4.setText("4. Num epochs:");
        self.l4.move(420, 70);

        self.e4 = QLineEdit(self)
        self.e4.move(570, 70);
        self.e4.setText(self.system["analysis"]["model_list"]["epochs"]);


        self.tb0 = QTextEdit(self);
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
            self.tb0.setText("Models available - {}".format(", ".join(combined_list_lower)))

        elif(self.system["backend"] == "Pytorch-1.3.1"):
            set1 = ["alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]
            set2 = ["densenet121", "densenet161", "densenet169", "densenet201"]
            set3 = ["googlenet", "inception_v3", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d",
                        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0, shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "wide_resnet101_2", "wide_resnet50_2"]
            set4 = ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3", "mobilenet_v2", "squeezenet1_0", "squeezenet1_1"]
            combined_list = set1+set2+set3+set4
            combined_list_lower = list(map(str.lower, combined_list))
            self.tb0.setText("Models available - {}".format(", ".join(combined_list_lower)))

        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            set1 = ["mobilenet", "densenet121", "densenet169", "densenet201", "inception_v3", 
                        "inception_resnet_v3", "mobilenet_v2", "nasnet_mobile", "nasnet_large", "resnet50",
                        "resnet101", "resnet152", "resnet50_v2", "resnet101_v2", "resnet152_v2", "vgg16",
                        "vgg19", "xception"];
            combined_list = set1
            combined_list_lower = list(map(str.lower, combined_list))
            self.l0.setText("Models available - {}".format(", ".join(combined_list_lower)))

        self.tb0.move(20, 110);
        self.tb0.resize(800, 80)



        self.b5 = QPushButton('Start Experiment', self)
        self.b5.move(20, 200)
        self.b5.clicked.connect(self.start)


        self.l5 = QLabel(self);
        self.l5.setText("Experiment Not Started");
        self.l5.resize(200, 25)
        self.l5.move(250, 200);


        self.b6 = QPushButton('Stop Experiment', self)
        self.b6.move(600, 200)
        self.b6.clicked.connect(self.stop)



        self.te1 = QTextBrowser(self);
        self.te1.move(20, 240);
        self.te1.setFixedSize(800, 300);
        self.te1.setText(self.system["analysis"]["model_list"]["analysis"]);

    

        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.readyReadStandardError.connect(self.stderrReady)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.finished.connect(self.finished)



    def start(self):
        self.l5.setText("Experiment Running")
        self.te1.setText("");

        self.system["analysis"]["model_list"]["analysis_name"] = self.e1.text();
        self.system["analysis"]["model_list"]["percent"] = self.e2.text();
        self.system["analysis"]["model_list"]["list"] = self.e3.text();
        self.system["analysis"]["model_list"]["epochs"] = self.e4.text();

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

        if self.system["datatype"] == "image" and self.system["labeltype"] == "single":
            os.system("cp cfg/classification/update/analyse_model_list.py .");
            os.system("cp cfg/classification/update/analyse_model_list.sh .");

        self.process.start('bash', ['analyse_model_list.sh'])
        self.append("Process PID: " + str(self.process.pid()) + "\n");


    def stop(self):
        self.l5.setText("Experiment Interrupted")
        self.process.kill();
        self.append("Experiment Stopped\n")
        QMessageBox.about(self, "Experiment Status", "Interrupted");


    def finished(self):
        pass;


    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')
        if("Completed" in text):
            self.l5.setText("Experiment Completed");
        self.system["analysis"]["model_list"]["analysis"] += text;
        self.append(text)


    def stderrReady(self):
        text = str(self.process.readAllStandardError().data(), encoding='utf-8')
        QMessageBox.about(self, "Experiment Status", "Errors Found");
        self.tb1.setText("Errors Found");
        self.append(text)


    def append(self, text):
        cursor = self.te1.textCursor()  
        self.te1.ensureCursorVisible() 
        cursor.movePosition(cursor.End)
        cursor.insertText(text)






    def forward(self):        
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_visualize.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_model_param.emit();


    



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateAnalysisModelList()
screen.show()
sys.exit(app.exec_())
'''