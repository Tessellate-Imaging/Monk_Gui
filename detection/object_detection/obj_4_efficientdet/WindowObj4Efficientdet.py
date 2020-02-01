import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowObj4Efficientdet(QtWidgets.QWidget):

    backward_obj = QtCore.pyqtSignal();
    forward_train = QtCore.pyqtSignal();
    forward_infer = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Object Detection - Efficient Detection'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);

        # Forward-Train
        self.b1 = QPushButton('Train', self)
        self.b1.move(600,200)
        self.b1.clicked.connect(self.forward1)


        # Forward-Infer
        self.b2 = QPushButton('Infer', self)
        self.b2.move(600,250)
        self.b2.clicked.connect(self.forward2)

        # Backward
        self.b3 = QPushButton('Back', self)
        self.b3.move(600,550)
        self.b3.clicked.connect(self.backward)


        # Quit
        self.b4 = QPushButton('Quit', self)
        self.b4.move(700,550)
        self.b4.clicked.connect(self.close)


        
        self.l1 = QLabel(self);
        self.l1.setText("Run Installation");
        self.l1.move(20, 20);


        self.l2 = QLabel(self);
        self.l2.setText("Ignore if already ran once for this detector");
        self.l2.move(20, 50);


        #Install
        self.cb4 = QComboBox(self);
        self.cudas = ["Cuda"];
        self.cb4.addItems(self.cudas);
        self.cb4.move(20, 110);

        self.b4 = QPushButton('Start Installation', self)
        self.b4.move(120,110)
        self.b4.clicked.connect(self.install)

        self.tb1 = QTextEdit(self)
        self.tb1.move(250, 110)
        self.tb1.setText("");
        self.tb1.resize(300, 25)
        self.tb1.setReadOnly(True)

        self.te1 = QTextBrowser(self);
        self.te1.move(20, 150);
        self.te1.setFixedSize(400, 400);


        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.readyReadStandardError.connect(self.stderrReady)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)


    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')
        if("Completed" in text):
            QMessageBox.about(self, "Installation Status", "Completed");
            self.tb1.setText("Installation Complete");
        self.append(text)



    def stderrReady(self):
        text = str(self.process.readAllStandardError().data(), encoding='utf-8')
        QMessageBox.about(self, "Installation Status", "Errors Found");
        self.tb1.setText("Error in installation");
        self.append(text)


    def append(self, text):
        cursor = self.te1.textCursor()  
        self.te1.ensureCursorVisible() 
        cursor.movePosition(cursor.End)
        cursor.insertText(text)


    def install(self):
        self.tb1.setText("Installation Running");
        if(self.cb4.currentText() == "Cuda"):
            os.system("cp cfg/detection/object_detection/obj_4_efficientdet/install_obj_4_efficientdet_cuda.sh .");
            os.system("chmod +x install_obj_4_efficientdet_cuda.sh");
            self.process.start('bash', ['install_obj_4_efficientdet_cuda.sh'])
            self.append("Process PID: " + str(self.process.pid()) + "\n");


    def forward1(self):
        self.forward_train.emit();

    def forward2(self):
        self.forward_infer.emit();

    def backward(self):
        self.backward_obj.emit();


'''
app = QApplication(sys.argv)
screen = WindowObj4Efficientdet()
screen.show()
sys.exit(app.exec_())
'''

