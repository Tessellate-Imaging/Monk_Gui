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



class WindowClassificationValidateUpdate(QtWidgets.QWidget):

    backward_exp = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Validate'.format(self.system["experiment"])
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
        self.b1.move(700,550)
        self.b1.clicked.connect(self.backward)

        # Quit
        self.bclose = QPushButton('Quit', self)
        self.bclose.move(800,550)
        self.bclose.clicked.connect(self.close)

        self.createLayout_group_datatype();
        self.createLayout_group_labeltype();
        self.createLayout_group_structuretype();


        self.te1 = QTextBrowser(self);
        self.te1.move(570, 20);
        self.te1.setFixedSize(300, 300);

        self.image_single_label = [];
        self.image_single_csv = [];
        self.image_single_foldered = [];

        self.isf_l1 = QLabel(self);
        self.isf_l1.setText("1. Dataset:");
        self.isf_l1.move(20, 300);
        self.image_single_foldered.append(self.isf_l1);

        self.isf_b1 = QPushButton('Select Folder', self)
        self.isf_b1.move(150,300)
        self.isf_b1.clicked.connect(self.select_val_dataset)
        self.image_single_foldered.append(self.isf_b1);

        self.isf_tb1 = QTextEdit(self)
        self.isf_tb1.move(250, 300)
        self.isf_tb1.resize(300, 50)
        self.isf_tb1.setText(self.system["evaluate"]["dir"]);
        self.isf_tb1.setReadOnly(True)
        self.image_single_foldered.append(self.isf_tb1);


        self.isc_l1 = QLabel(self);
        self.isc_l1.setText("1. Dataset:");
        self.isc_l1.move(20, 300);
        self.image_single_csv.append(self.isc_l1);

        self.isc_b1 = QPushButton('Select Folder', self)
        self.isc_b1.move(150,300)
        self.isc_b1.clicked.connect(self.select_val_dataset)
        self.image_single_csv.append(self.isc_b1);

        self.isc_tb1 = QTextEdit(self)
        self.isc_tb1.move(250, 300)
        self.isc_tb1.resize(300, 50)
        self.isc_tb1.setText(self.system["evaluate"]["cdir"]);
        self.isc_tb1.setReadOnly(True)
        self.image_single_csv.append(self.isc_tb1);


        self.isc_l3 = QLabel(self);
        self.isc_l3.setText("2. Labels:");
        self.isc_l3.move(20, 400);
        self.image_single_csv.append(self.isc_l3);

        self.isc_b3 = QPushButton('Select File', self)
        self.isc_b3.move(150, 400)
        self.isc_b3.clicked.connect(self.select_val_csv)
        self.image_single_csv.append(self.isc_b3);

        self.isc_tb3 = QTextEdit(self)
        self.isc_tb3.move(250, 400)
        self.isc_tb3.resize(300, 50)
        self.isc_tb3.setText(self.system["evaluate"]["csv"]);
        self.isc_tb3.setReadOnly(True)
        self.image_single_csv.append(self.isc_tb3);



        self.b4 = QPushButton('Predict', self)
        self.b4.move(480, 200)
        self.b4.clicked.connect(self.predict);
        self.image_single_label.append(self.b4)


        self.l7 = QLabel(self);
        self.l7.setText("Results:");
        self.l7.move(500, 470);
        self.image_single_label.append(self.l7)


        self.tb7 = QTextEdit(self)
        self.tb7.move(570, 340)
        self.tb7.resize(300, 200)
        self.tb7.setReadOnly(True)
        self.image_single_label.append(self.tb7)



        if self.system["datatype"] == "image":
            self.datatype_image();
        elif self.system["datatype"] == "npy":
            self.datatype_npy();
        elif self.system["datatype"] == "hdf5":
            self.datatype_hdf5();
        elif self.system["datatype"] == "parquet":
            self.datatype_parquet();


        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.readyReadStandardError.connect(self.stderrReady)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.finished.connect(self.finished)
        self.predictions = "";
    



    def select_val_dataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", os.getcwd())
        if self.system["evaluate"]["structuretype"] == "foldered":
            self.isf_b1.setText("Selected");
            self.isf_tb1.setText(folderName);
            self.system["evaluate"]["dir"] = folderName;
        else:
            self.isc_b1.setText("Selected");
            self.isc_tb1.setText(folderName);
            self.system["evaluate"]["cdir"] = folderName;

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_val_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd(), 
                                                    "Label Files (*.csv);;All Files (*)", options=options)

        self.isc_b4.setText("Selected");
        self.isc_tb4.setText(fileName);
        self.system["evaluate"]["csv"] = fileName;

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)




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


    def createLayout_group_structuretype(self):
        self.group_structuretype = QGroupBox("Folder Structure Type", self)
        self.group_structuretype.resize(450, 70);
        self.group_structuretype.move(20, 190);
        layout_groupbox = QHBoxLayout(self.group_structuretype)

        self.structuretype_1 = QRadioButton("Foldered Dataset", self)
        layout_groupbox.addWidget(self.structuretype_1)

        self.structuretype_2 = QRadioButton("CSV Dataset", self)
        layout_groupbox.addWidget(self.structuretype_2)

        if self.system["evaluate"]["structuretype"] == "foldered":
            self.structuretype_1.setChecked(True)
        elif self.system["evaluate"]["structuretype"] == "csv":
            self.structuretype_2.setChecked(True)

        self.structuretype_1.toggled.connect(self.structuretype_foldered);
        self.structuretype_2.toggled.connect(self.structuretype_csv);

        layout_groupbox.addStretch(1);


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


    
    def labeltype_single(self):
        self.group_structuretype.show();
        if(self.system["evaluate"]["structuretype"] == "foldered"):
            self.structuretype_foldered();
        else:
            self.structuretype_csv();
        self.system["labeltype"] = "single";

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

    def labeltype_multi(self):
        for x in self.image_single_foldered:
            x.hide();
        for x in self.image_single_csv:
            x.hide();
        self.group_structuretype.hide();
        self.te1.setText("Dataloader for Multi label classification unimplemented");
        self.system["labeltype"] = "multi";

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def structuretype_foldered(self):
        for x in self.image_single_foldered:
            x.show();
        for x in self.image_single_csv:
            x.hide();
        self.system["evaluate"]["structuretype"] = "foldered";
        self.te1.setText(self.writeup_image_single_foldered());
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

    def structuretype_csv(self):
        for x in self.image_single_foldered:
            x.hide();
        for x in self.image_single_csv:
            x.show();

        self.system["evaluate"]["structuretype"] = "csv";
        self.te1.setText(self.writeup_image_single_csv());
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    

    def predict(self):
        self.tb7.setText("Running Evaluation");
        self.te1.setText("");
        if self.system["datatype"] == "image" and self.system["labeltype"] == "single":
            os.system("cp cfg/classification/update/evaluate_cls_image_single.py .");
            os.system("cp cfg/classification/update/evaluate_cls_image_single.sh .");
        self.process.start('bash', ['evaluate_cls_image_single.sh'])
        self.append("Process PID: " + str(self.process.pid()) + "\n");

    def stop(self):
        self.process.kill();
        self.append("Prediction Stopped\n")


    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')
        if("Completed" in text):
            f = open("results.txt", 'r');
            r = f.read();
            f.close();
            self.tb7.setText(r);

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



    def writeup_image_single_foldered(self):
        wr = "";
        wr += "Foldered Dataset\n"
        wr += "Dataset Structure Example\n\n";
        wr += "Parent Folder\n";
        wr += "       |\n";
        wr += "       |\n";
        wr += "       |---Validation Images (Optional)\n";
        wr += "             |\n"
        wr += "             |--Cats\n";
        wr += "                 |\n";
        wr += "                 |--img1.jpg\n";
        wr += "                 |--img2.jpg\n";
        wr += "                 |--...(so on)\n";
        wr += "             |--Dogs\n";
        wr += "                 |\n";
        wr += "                 |--img1.jpg\n";
        wr += "                 |--img2.jpg\n";
        wr += "                 |--...(so on)\n";
        wr += "             |--...(More classes and so on)\n";

        return wr;


    def writeup_image_single_csv(self):
        wr = "";
        wr += "CSV Dataset\n"
        wr += "Dataset Structure Example\n\n";
        wr += "Parent Folder\n";
        wr += "       |\n";
        wr += "       |\n";
        wr += "       |---ValImages\n";
        wr += "                 |--img1.jpg\n";
        wr += "                 |--img2.jpg\n";
        wr += "                 |--...(so on)\n\n";
        wr += "       |---val_labels.csv\n\n\n";
        wr += "Annotation Format\n";
        wr += "       | Id         | Labels  |\n";
        wr += "       | img1.jpg   | label1  |\n";
        wr += "       | img2.jpg   | label2  |\n";

        return wr;


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_exp.emit();


'''
app = QApplication(sys.argv)
screen = WindowClassificationValidateUpdate()
screen.show()
sys.exit(app.exec_())
'''