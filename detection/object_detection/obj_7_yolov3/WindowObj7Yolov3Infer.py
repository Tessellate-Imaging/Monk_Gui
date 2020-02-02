import os
import sys
import json
import time
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSignal, pyqtSlot

class WindowObj7Yolov3Infer(QtWidgets.QWidget):

    backward_7_yolov3 = QtCore.pyqtSignal();



    def __init__(self):
        super().__init__()
        self.title = 'Yolo-V3 - Infer'
        self.left = 100
        self.top = 100
        self.width = 900
        self.height = 600
        self.cfg_setup();
        self.initUI()

    def cfg_setup(self):
        if(os.path.isfile("obj_7_yolov3_infer.json")):
            with open('obj_7_yolov3_infer.json') as json_file:
                self.system = json.load(json_file)
        else:
            self.system = {};
            self.system["model"] = "yolov3";
            self.system["weights"] = "weights/last.pt";
            self.system["img_file"] = "Monk_Object_Detection/example_notebooks/sample_dataset/ship/test/img1.jpg";
            self.system["conf_thresh"] = "0.7";
            self.system["iou_thresh"] = "0.5";
            self.system["class_file"] = "Monk_Object_Detection/example_notebooks/sample_dataset/ship/annotations/classes.txt"

            with open('obj_7_yolov3_infer.json', 'w') as outfile:
                json.dump(self.system, outfile)


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);


        # Backward
        self.b1 = QPushButton('Back', self)
        self.b1.move(700,550)
        self.b1.clicked.connect(self.backward)

        # Quit
        self.bclose = QPushButton('Quit', self)
        self.bclose.move(800,550)
        self.bclose.clicked.connect(self.close)


        self.l2 = QLabel(self);
        self.l2.setText("1. Model name :");
        self.l2.move(20, 20);

        self.cb2 = QComboBox(self);
        self.models = ["yolov3", "yolov3s", "yolov3-spp", "yolov3-spp3", "yolov3-tiny",
                        "yolov3-spp-matrix", "csresnext50-panet-spp"];
        self.cb2.addItems(self.models);
        index = self.cb2.findText(self.system["model"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb2.setCurrentIndex(index)
        self.cb2.move(140, 20);
        self.cb2.activated.connect(self.select_weight_file)

        self.tb2 = QTextEdit(self)
        self.tb2.move(20, 50)
        self.tb2.resize(300, 80)
        self.tb2.setText(self.system["weights"]);
        self.tb2.setReadOnly(True)



        self.l4 = QLabel(self);
        self.l4.setText("2. Image File: ");
        self.l4.move(20, 190);

        self.b4 = QPushButton('Select File', self)
        self.b4.move(130, 190)
        self.b4.clicked.connect(self.select_img_file);

        self.tb4 = QTextEdit(self)
        self.tb4.move(20, 220)
        self.tb4.resize(300, 80)
        self.tb4.setText(self.system["img_file"]);
        self.tb4.setReadOnly(True)


        self.l5 = QLabel(self);
        self.l5.setText("3. Confidence Threshold:");
        self.l5.move(20, 340);

        self.e4 = QLineEdit(self)
        self.e4.move(200, 340);
        self.e4.setText(self.system["conf_thresh"]);
        self.e4.resize(130, 25);

        self.l6 = QLabel(self);
        self.l6.setText("4. IOU Threshold:");
        self.l6.move(20, 400);

        self.e6 = QLineEdit(self)
        self.e6.move(200, 400);
        self.e6.setText(self.system["iou_thresh"]);
        self.e6.resize(130, 25);


        self.l5 = QLabel(self);
        self.l5.setText("5. Classes File List: ");
        self.l5.move(20, 460);

        self.b5 = QPushButton('Select File', self)
        self.b5.move(150, 460)
        self.b5.clicked.connect(self.select_class_file);

        self.tb5 = QTextEdit(self)
        self.tb5.move(20, 490)
        self.tb5.resize(300, 80)
        self.tb5.setText(self.system["class_file"]);
        self.tb5.setReadOnly(True)

        self.b5 = QPushButton('Predict', self)
        self.b5.move(350, 200)
        self.b5.clicked.connect(self.Predict);


        self.te1 = QTextBrowser(self);
        self.te1.move(450, 20);
        self.te1.setFixedSize(400, 200);


        self.l6 = QLabel(self)
        self.l6.move(450, 250);
        self.l6.resize(400, 300)
        
        


        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.readyReadStandardError.connect(self.stderrReady)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)



    def select_weight_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd() + "/weights/", 
                                                    "Model Files (*.pt);;All Files (*)", 
                                                    options=options)

        self.system["model"] = self.cb2.currentText();
        self.system["weights"] = fileName
        self.tb2.setText(fileName);

        self.system["conf_thresh"] = self.e4.text();

        with open('obj_7_yolov3_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_img_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd(), 
                                                    "All Files (*)", 
                                                    options=options)
        self.system["img_file"] = fileName;
        self.tb4.setText(fileName);

        self.system["conf_thresh"] = self.e4.text();

        with open('obj_7_yolov3_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_class_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd(), 
                                                    "Text Files (*.txt);;All Files (*)", 
                                                    options=options)
        self.system["class_file"] = fileName;

        self.tb5.setText(fileName);

        self.system["conf_thresh"] = self.e4.text();

        with open('obj_7_yolov3_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)



    def Predict(self):
        self.system["conf_thresh"] = self.e4.text();
        self.te1.setText("");
        with open('obj_7_yolov3_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)

        os.system("cp cfg/detection/object_detection/obj_7_yolov3/infer_obj_7_yolov3.py .");
        os.system("cp cfg/detection/object_detection/obj_7_yolov3/infer_obj_7_yolov3.sh .");

        self.process.start('bash', ['infer_obj_7_yolov3.sh'])
        self.append("Process PID: " + str(self.process.pid()) + "\n");


    def stop(self):
        self.process.kill();
        self.append("Prediction Stopped\n")


    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')
        if("Completed" in text):
            pixmap = QPixmap('output.jpg')
            pixmap = pixmap.scaledToWidth(400)
            pixmap = pixmap.scaledToHeight(300)
            self.l6.setPixmap(pixmap)
        self.append(text)


    def stderrReady(self):
        text = str(self.process.readAllStandardError().data(), encoding='utf-8')
        self.append(text)

    def append(self, text):
        cursor = self.te1.textCursor()  
        self.te1.ensureCursorVisible() 
        cursor.movePosition(cursor.End)
        cursor.insertText(text)


    def backward(self):
        self.system["conf_thresh"] = self.e4.text();
        with open('obj_7_yolov3_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_7_yolov3.emit();



'''
app = QApplication(sys.argv)
screen = WindowObj7Yolov3Infer()
screen.show()
sys.exit(app.exec_())
'''