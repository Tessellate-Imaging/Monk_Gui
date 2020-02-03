import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowDetectionMain(QtWidgets.QWidget):

    backward_main = QtCore.pyqtSignal();
    forward_obj_1_gluoncv_finetune = QtCore.pyqtSignal();
    forward_obj_2_pytorch_finetune = QtCore.pyqtSignal();
    forward_obj_3_mxrcnn = QtCore.pyqtSignal();
    forward_obj_4_efficientdet = QtCore.pyqtSignal();
    forward_obj_5_pytorch_retinanet = QtCore.pyqtSignal();
    forward_obj_6_cornernet_lite = QtCore.pyqtSignal();
    forward_obj_7_yolov3 = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.title = 'Monk Detection'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 400
        self.initUI()




    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);


        self.te1 = QTextBrowser(self);
        self.te1.move(450, 100);
        self.te1.setFixedSize(300, 200);

        
        # General Object Detection
        self.l1 = QLabel(self);
        self.l1.setText("General Object Detection:");
        self.l1.move(30, 10);


        # General Object Detection
        self.cb1 = QComboBox(self);
        self.obj1 = ["Select", "GluonCV Finetune", "Pytorch Finetune", "MX-Rcnn", "Efficient-Det",
                            "Pytorch RetinaNet", "CornerNet-Lite", "Yolo-V3"];
        self.cb1.addItems(self.obj1);
        self.cb1.move(300, 10);


        # General Image Segmentation
        self.l2 = QLabel(self);
        self.l2.setText("General Image Segmentation:");
        self.l2.move(30, 50);


        # General Image Segmentation
        self.cb2 = QComboBox(self);
        self.cb2.addItems(["Select"]);
        self.cb2.move(300, 50);
        self.cb2.setEnabled(False);


        # Face Recognition
        self.l3 = QLabel(self);
        self.l3.setText("Face Recognition:");
        self.l3.move(30, 90);


        # Face Recognition
        self.cb3 = QComboBox(self);
        self.cb3.addItems(["Select"]);
        self.cb3.move(300, 90);
        self.cb3.setEnabled(False);


        # Text Detection
        self.l4 = QLabel(self);
        self.l4.setText("Text Detection:");
        self.l4.move(30, 130);


        # Text Detection
        self.cb4 = QComboBox(self);
        self.cb4.addItems(["Select"]);
        self.cb4.move(300, 130);
        self.cb4.setEnabled(False);


        # Forward
        if not os.path.isdir("Monk_Object_Detection"):
            self.b1 = QPushButton('Cloning (Wait)', self)
            self.b1.move(550,350)
            self.b1.clicked.connect(self.forward)
            self.b1.setEnabled(False)
        else:
            self.b1 = QPushButton('Next', self)
            self.b1.move(550,350)
            self.b1.clicked.connect(self.forward)

        # Forward
        self.b2 = QPushButton('Back', self)
        self.b2.move(450,350)
        self.b2.clicked.connect(self.backward)


        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(700,350)
        self.b3.clicked.connect(self.close)

        if not os.path.isdir("Monk_Object_Detection"):
            self.process = QtCore.QProcess(self)
            self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.stdoutReady)
            self.process.readyReadStandardError.connect(self.stderrReady)
            self.process.finished.connect(self.finished)

            self.process.start('git clone https://github.com/Tessellate-Imaging/Monk_Object_Detection.git')
        else:
            self.te1.setText("Monk Object Detection Library Cloned.");


    
    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')
        if("done" in text):
            self.b1.setEnabled(True)
            self.b1.setText("Next")
        self.append(text)

    def finished(self):
        self.b1.setEnabled(True)
        self.b1.setText("Next")
        self.te1.setText("Monk Detection Library Cloned.");  

    def stderrReady(self):
        text = str(self.process.readAllStandardError().data(), encoding='utf-8')
        self.append(text)

    def append(self, text):
        cursor = self.te1.textCursor()  
        self.te1.ensureCursorVisible() 
        cursor.movePosition(cursor.End)
        cursor.insertText(text) 




    def forward(self):
        index = self.obj1.index(str(self.cb1.currentText()))
        if(index == 0):
            QMessageBox.about(self, "Warning", "Select an algorithm");
        elif(index==1):
            self.forward_obj_1_gluoncv_finetune.emit();
        elif(index==2):
            self.forward_obj_2_pytorch_finetune.emit();
        elif(index==3):
            self.forward_obj_3_mxrcnn.emit();
        elif(index==4):
            self.forward_obj_4_efficientdet.emit();
        elif(index==5):
            self.forward_obj_5_pytorch_retinanet.emit();
        elif(index==6):
            self.forward_obj_6_cornernet_lite.emit();
        elif(index==7):
            self.forward_obj_7_yolov3.emit();
            


    def backward(self):
        self.backward_main.emit()


'''
app = QApplication(sys.argv)
screen = WindowDetectionMain()
screen.show()
sys.exit(app.exec_())
'''