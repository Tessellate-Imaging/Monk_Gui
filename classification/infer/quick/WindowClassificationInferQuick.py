import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationInferQuick(QtWidgets.QWidget):

    backward_exp = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Infer'.format(self.system["experiment"])
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
        self.b1.move(700,560)
        self.b1.clicked.connect(self.backward)

        # Quit
        self.bclose = QPushButton('Quit', self)
        self.bclose.move(800,560)
        self.bclose.clicked.connect(self.close)

        self.createLayout_group_datatype();
        self.createLayout_group_labeltype();


        self.te1 = QTextBrowser(self);
        self.te1.move(20, 430);
        self.te1.setFixedSize(450, 150);

        self.image_single_label = [];

        self.cb1 = QComboBox(self);
        self.data = ["Single Image", "Image Folder"]
        self.cb1.addItems(self.data);
        self.cb1.move(20, 200);
        index = self.cb1.findText(self.system["test_data"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb1.setCurrentIndex(index)
        self.cb1.activated.connect(self.use_folder);
        self.image_single_label.append(self.cb1)

        self.l2 = QLabel(self);
        self.l2.setText("Image File: ");
        self.l2.move(20, 250);
        self.image_single_label.append(self.l2)

        self.b2 = QPushButton('Select File', self)
        self.b2.move(130, 250)
        self.b2.clicked.connect(self.select_img_file);
        self.image_single_label.append(self.b2)


        self.l3 = QLabel(self);
        self.l3.setText("Image Folder: ");
        self.l3.move(20, 250);
        self.image_single_label.append(self.l3)

        self.b3 = QPushButton('Select Folder', self)
        self.b3.move(130, 250)
        self.b3.clicked.connect(self.select_img_folder);
        self.image_single_label.append(self.b3)

        self.tb4 = QTextEdit(self)
        self.tb4.move(20, 280)
        self.tb4.resize(300, 80)
        self.tb4.setReadOnly(True)
        self.image_single_label.append(self.tb4)

        self.b4 = QPushButton('Predict', self)
        self.b4.move(130, 400)
        self.b4.clicked.connect(self.predict);
        self.image_single_label.append(self.b4)


        self.l6 = QLabel(self)
        self.l6.move(480, 20);
        self.l6.resize(400, 300)


        self.index = 0;
        self.img_list = [];

        self.b5 = QPushButton('Previous', self)
        self.b5.move(540, 400)
        self.b5.clicked.connect(self.previous);
        self.b5.setEnabled(False)
        self.image_single_label.append(self.b5);


        self.b6 = QPushButton('Next', self)
        self.b6.move(660, 400)
        self.b6.clicked.connect(self.next);
        self.b6.setEnabled(False)
        self.image_single_label.append(self.b6);


        self.l7_1 = QLabel(self);
        self.l7_1.setText("Img_Name: ");
        self.l7_1.move(480, 440);
        self.image_single_label.append(self.l7_1)


        self.tb7_1 = QTextEdit(self)
        self.tb7_1.move(570, 440)
        self.tb7_1.resize(300, 30)
        self.tb7_1.setReadOnly(True)
        self.image_single_label.append(self.tb7_1)


        self.l7_2 = QLabel(self);
        self.l7_2.setText("Type: ");
        self.l7_2.move(500, 480);
        self.image_single_label.append(self.l7_2)


        self.tb7_2 = QTextEdit(self)
        self.tb7_2.move(570, 480)
        self.tb7_2.resize(300, 30)
        self.tb7_2.setReadOnly(True)
        self.image_single_label.append(self.tb7_2)


        self.l7_3 = QLabel(self);
        self.l7_3.setText("Probability: ");
        self.l7_3.move(480, 520);
        self.image_single_label.append(self.l7_3)


        self.tb7_3 = QTextEdit(self)
        self.tb7_3.move(570, 520)
        self.tb7_3.resize(300, 30)
        self.tb7_3.setReadOnly(True)
        self.image_single_label.append(self.tb7_3)






        if self.system["datatype"] == "image":
            self.datatype_image();
        elif self.system["datatype"] == "npy":
            self.datatype_npy();
        elif self.system["datatype"] == "hdf5":
            self.datatype_hdf5();
        elif self.system["datatype"] == "parquet":
            self.datatype_parquet();

        self.use_folder();


        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.readyReadStandardError.connect(self.stderrReady)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.finished.connect(self.finished)
        self.predictions = "";
    


    def previous(self):
        self.b6.setEnabled(True);
        self.index -= 1;
        self.tb7_1.setText(str(self.data[str(self.index)]["img_name"]));
        self.tb7_2.setText(str(self.data[str(self.index)]["predicted_class"]));
        self.tb7_3.setText(str(self.data[str(self.index)]["score"]));
        pixmap = QPixmap(self.img_list[self.index])
        pixmap = pixmap.scaledToWidth(400)
        pixmap = pixmap.scaledToHeight(300)
        self.l6.setPixmap(pixmap)
        if(self.index == 0):
            self.b5.setEnabled(False);


    def next(self):
        self.b5.setEnabled(True);
        self.index += 1;
        self.tb7_1.setText(str(self.data[str(self.index)]["img_name"]));
        self.tb7_2.setText(str(self.data[str(self.index)]["predicted_class"]));
        self.tb7_3.setText(str(self.data[str(self.index)]["score"]));
        pixmap = QPixmap(self.img_list[self.index])
        pixmap = pixmap.scaledToWidth(400)
        pixmap = pixmap.scaledToHeight(300)
        self.l6.setPixmap(pixmap)
        if(self.index == len(self.img_list)):
            self.b6.setEnabled(False);




    def createLayout_group_datatype(self):
        self.group_datatype = QGroupBox("Data Type", self)
        self.group_datatype.resize(450, 70);
        self.group_datatype.move(20, 20);
        layout_groupbox = QHBoxLayout(self.group_datatype)

        self.datatype_1 = QRadioButton("Image Files", self)
        layout_groupbox.addWidget(self.datatype_1)

        self.datatype_2 = QRadioButton("NPY Files", self)
        layout_groupbox.addWidget(self.datatype_2)

        self.datatype_3 = QRadioButton("HDF5 Files", self)
        layout_groupbox.addWidget(self.datatype_3)

        self.datatype_4 = QRadioButton("Parquet Files", self)
        layout_groupbox.addWidget(self.datatype_4)

        if self.system["datatype"] == "image":
            self.datatype_1.setChecked(True)
        elif self.system["datatype"] == "npy":
            self.datatype_2.setChecked(True)
        elif self.system["datatype"] == "hdf5":
            self.datatype_3.setChecked(True)
        elif self.system["datatype"] == "parquet":
            self.datatype_4.setChecked(True)

        self.datatype_1.toggled.connect(self.datatype_image);
        self.datatype_2.toggled.connect(self.datatype_npy);
        self.datatype_3.toggled.connect(self.datatype_hdf5);
        self.datatype_4.toggled.connect(self.datatype_parquet);

        layout_groupbox.addStretch(1)


    def createLayout_group_labeltype(self):
        self.group_labeltype = QGroupBox("Label Type", self)
        self.group_labeltype.resize(450, 70);
        self.group_labeltype.move(20, 110);
        layout_groupbox = QHBoxLayout(self.group_labeltype)

        self.labeltype_1 = QRadioButton("Single Label Prediction", self)
        layout_groupbox.addWidget(self.labeltype_1)

        self.labeltype_2 = QRadioButton("Multi Label Prediction", self)
        layout_groupbox.addWidget(self.labeltype_2)

        if self.system["labeltype"] == "single":
            self.labeltype_1.setChecked(True)
        elif self.system["labeltype"] == "multi":
            self.labeltype_2.setChecked(True)

        self.labeltype_1.toggled.connect(self.labeltype_single);
        self.labeltype_2.toggled.connect(self.labeltype_multi);

        layout_groupbox.addStretch(1)


    def datatype_image(self):
        self.group_labeltype.show();
        if self.system["labeltype"] == "single":
            self.labeltype_single();
        else:
            self.labeltype_multi();
        self.system["datatype"] = "image";
        

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

    def datatype_npy(self):
        self.group_labeltype.hide();
        self.te1.setText("Dataloader for Numpy files unimplemented");
        self.system["datatype"] = "npy";
        for x in self.image_single_label:
            x.hide();

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

    def datatype_hdf5(self):
        self.group_labeltype.hide();
        self.te1.setText("Dataloader for Hdf5 files unimplemented");
        self.system["datatype"] = "hdf5";
        for x in self.image_single_label:
            x.hide();

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

    def datatype_parquet(self):
        self.group_labeltype.hide();
        self.te1.setText("Dataloader for Parquet files unimplemented");
        self.system["datatype"] = "parquet";
        for x in self.image_single_label:
            x.hide();

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)



    def labeltype_single(self):
        self.te1.setText("Upload single image and test or upload a dataset full of images.")
        self.system["labeltype"] = "single";
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        for x in self.image_single_label:
            x.show();

    def labeltype_multi(self):
        self.te1.setText("Dataloader for Multi label classification unimplemented");
        self.system["labeltype"] = "multi";

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        for x in self.image_single_label:
            x.hide();


    
    def select_img_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd(), 
                                                    "All Files (*)", 
                                                    options=options)
        self.system["img_file"] = fileName;
        self.tb4.setText(fileName);
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_img_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", os.getcwd())
        self.system["img_folder"] = folderName;
        self.tb4.setText(folderName);
        self.img_list = sorted(os.listdir(folderName))
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def use_folder(self):
        if(self.cb1.currentText() == "Image Folder"):
            self.l2.hide();
            self.b2.hide();
            self.l3.show();
            self.b3.show();
            self.tb4.setText(self.system["img_folder"]);
            self.system["test_data"] = "Image Folder";
        else:
            self.l2.show();
            self.b2.show();
            self.l3.hide();
            self.b3.hide();
            self.tb4.setText(self.system["img_file"]);
            self.system["test_data"] = "Single Image";

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

    

    def predict(self):
        self.img_list = sorted(os.listdir(self.system["img_folder"]))
        for i in range(len(self.img_list)):
            self.img_list[i] = self.system["img_folder"] + "/" + self.img_list[i];
        self.te1.setText("");
        self.predictions = "";
        if self.system["datatype"] == "image" and self.system["labeltype"] == "single":
            os.system("cp cfg/classification/quick/infer_cls_image_single.py .");
            os.system("cp cfg/classification/quick/infer_cls_image_single.sh .");
        self.process.start('bash', ['infer_cls_image_single.sh'])
        self.append("Process PID: " + str(self.process.pid()) + "\n");

    def stop(self):
        self.process.kill();
        self.append("Prediction Stopped\n")


    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')
        if("Completed" in text):
            if(self.cb1.currentText() == "Image Folder"):
                self.b6.setEnabled(True)
                self.index = 0;
                with open('output.json') as json_file:
                    self.data = json.load(json_file)
                self.tb7_1.setText(str(self.data[str(self.index)]["img_name"]));
                self.tb7_2.setText(str(self.data[str(self.index)]["predicted_class"]));
                self.tb7_3.setText(str(self.data[str(self.index)]["score"]));
                QMessageBox.about(self, "Prediction Status", "Completed");
                pixmap = QPixmap(self.img_list[self.index])
                pixmap = pixmap.scaledToWidth(400)
                pixmap = pixmap.scaledToHeight(300)
                self.l6.setPixmap(pixmap)
            else:
                QMessageBox.about(self, "Prediction Status", "Completed");
                pixmap = QPixmap(self.system["img_file"])
                pixmap = pixmap.scaledToWidth(400)
                pixmap = pixmap.scaledToHeight(300)
                self.l6.setPixmap(pixmap)
                with open('output.json') as json_file:
                    data = json.load(json_file)
                self.tb7_1.setText(str(data["img_name"].split("/")[-1]));
                self.tb7_2.setText(str(data["predicted_class"]));
                self.tb7_3.setText(str(data["score"]));

        self.append(text)

    def finished(self):
        pass;
        #self.tb7.setText(self.predictions);


    def stderrReady(self):
        text = str(self.process.readAllStandardError().data(), encoding='utf-8')
        self.append(text)

    def append(self, text):
        cursor = self.te1.textCursor()  
        self.te1.ensureCursorVisible() 
        cursor.movePosition(cursor.End)
        cursor.insertText(text)




    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_exp.emit();


'''
app = QApplication(sys.argv)
screen = WindowClassificationInferQuick()
screen.show()
sys.exit(app.exec_())
'''