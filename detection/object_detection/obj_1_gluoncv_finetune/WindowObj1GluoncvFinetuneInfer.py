import os
import sys
import json
import time
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSignal, pyqtSlot

class WindowObj1GluoncvFinetuneInfer(QtWidgets.QWidget):

    backward_1_gluoncv_finetune = QtCore.pyqtSignal();



    def __init__(self):
        super().__init__()
        self.title = 'GluonCV Finetune - Infer'
        self.left = 100
        self.top = 100
        self.width = 900
        self.height = 600
        self.cfg_setup();
        self.initUI()

    def cfg_setup(self):
        if(os.path.isfile("obj_1_gluoncv_finetune_infer.json")):
            with open('obj_1_gluoncv_finetune_infer.json') as json_file:
                self.system = json.load(json_file)
        else:
            self.system = {};
            self.system["model"] = "ssd_300_vgg16_atrous_coco";
            self.system["weights"] = "saved_model.params";
            self.system["use_gpu"] = "yes";
            self.system["img_file"] = "Monk_Object_Detection/example_notebooks/sample_dataset/kangaroo/test/kg1.jpeg";
            self.system["conf_thresh"] = "0.7";
            self.system["class_file"] = "Monk_Object_Detection/example_notebooks/sample_dataset/kangaroo/classes.txt"

            with open('obj_1_gluoncv_finetune_infer.json', 'w') as outfile:
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
        self.l2.setText("2. Weights File: ");
        self.l2.move(20, 70);

        self.b2 = QPushButton('Select File', self)
        self.b2.move(130, 70)
        self.b2.clicked.connect(self.select_model_file);

        self.tb2 = QTextEdit(self)
        self.tb2.move(20, 100)
        self.tb2.resize(300, 80)
        self.tb2.setText(self.system["weights"]);
        self.tb2.setReadOnly(True)


        self.l3 = QLabel(self);
        self.l3.setText("3. Use Gpu :");
        self.l3.move(20, 210);

        self.cb3 = QComboBox(self);
        self.use_gpu = ["Yes", "No"];
        self.cb3.addItems(self.use_gpu);
        index = self.cb3.findText(self.system["use_gpu"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb3.setCurrentIndex(index)
        self.cb3.move(120, 210);


        self.l4 = QLabel(self);
        self.l4.setText("4. Image File: ");
        self.l4.move(20, 250);

        self.b4 = QPushButton('Select File', self)
        self.b4.move(130, 250)
        self.b4.clicked.connect(self.select_img_file);

        self.tb4 = QTextEdit(self)
        self.tb4.move(20, 280)
        self.tb4.resize(300, 80)
        self.tb4.setText(self.system["img_file"]);
        self.tb4.setReadOnly(True)


        self.l5 = QLabel(self);
        self.l5.setText("5. Confidence Threshold:");
        self.l5.move(20, 400);

        self.e4 = QLineEdit(self)
        self.e4.move(200, 400);
        self.e4.setText(self.system["conf_thresh"]);
        self.e4.resize(130, 25);


        self.l5 = QLabel(self);
        self.l5.setText("6. Classes File List: ");
        self.l5.move(20, 440);

        self.b5 = QPushButton('Select File', self)
        self.b5.move(150, 440)
        self.b5.clicked.connect(self.select_class_file);

        self.tb5 = QTextEdit(self)
        self.tb5.move(20, 470)
        self.tb5.resize(300, 80)
        self.tb5.setText(self.system["class_file"]);
        self.tb5.setReadOnly(True)

        self.b5 = QPushButton('Predict', self)
        self.b5.move(330, 550)
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



    def select_model_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd(), 
                                                    "Monk Project Files (*.params);;All Files (*)", options=options)
        self.system["weights"] = fileName;
        self.tb2.setText(fileName);

        self.system["model"] = self.cb1.currentText();
        self.system["use_gpu"] = self.cb3.currentText();
        self.system["conf_thresh"] = self.e4.text();

        with open('obj_1_gluoncv_finetune_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_img_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd(), 
                                                    "All Files (*)", 
                                                    options=options)
        self.system["img_file"] = fileName;
        self.tb2.setText(fileName);

        self.system["model"] = self.cb1.currentText();
        self.system["use_gpu"] = self.cb3.currentText();
        self.system["conf_thresh"] = self.e4.text();

        with open('obj_1_gluoncv_finetune_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_class_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd(), 
                                                    "Text Files (*.txt);;All Files (*)", 
                                                    options=options)
        self.system["class_file"] = fileName;

        self.tb5.setText(fileName);

        self.system["model"] = self.cb1.currentText();
        self.system["use_gpu"] = self.cb3.currentText();
        self.system["conf_thresh"] = self.e4.text();

        with open('obj_1_gluoncv_finetune_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)



    def Predict(self):
        self.system["model"] = self.cb1.currentText();
        self.system["use_gpu"] = self.cb3.currentText();
        self.system["conf_thresh"] = self.e4.text();
        self.te1.setText("");
        with open('obj_1_gluoncv_finetune_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)

        os.system("cp cfg/detection/object_detection/obj_1_gluoncv_finetune/infer_obj_1_gluoncv_finetune.py .");
        os.system("cp cfg/detection/object_detection/obj_1_gluoncv_finetune/infer_obj_1_gluoncv_finetune.sh .");

        self.process.start('bash', ['infer_obj_1_gluoncv_finetune.sh'])
        self.append("Process PID: " + str(self.process.pid()) + "\n");


    def stop(self):
        self.process.kill();
        self.append("Prediction Stopped\n")


    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')
        if("Completed" in text):
            pixmap = QPixmap('output.png')
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
        self.system["model"] = self.cb1.currentText();
        self.system["use_gpu"] = self.cb3.currentText();
        self.system["conf_thresh"] = self.e4.text();
        with open('obj_1_gluoncv_finetune_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_1_gluoncv_finetune.emit();



'''
app = QApplication(sys.argv)
screen = WindowObj1GluoncvFinetuneInfer()
screen.show()
sys.exit(app.exec_())
'''