import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainQuickDataParam(QtWidgets.QWidget):

    forward_model_param = QtCore.pyqtSignal();
    backward_experiment_run_mode = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Dataset Params'.format(self.system["experiment"])
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

        # Backward
        self.b2 = QPushButton('Next', self)
        self.b2.move(700,550)
        self.b2.clicked.connect(self.forward)

        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(800, 550)
        self.b3.clicked.connect(self.close)


        self.createLayout_group_datatype();
        self.createLayout_group_labeltype();
        self.createLayout_group_structuretype();

        self.tb1 = QTextEdit(self)
        self.tb1.move(580, 20)
        self.tb1.resize(300, 450)
        self.tb1.setReadOnly(True)




        self.image_single_foldered = [];
        self.image_single_csv = [];

        self.isf_l1 = QLabel(self);
        self.isf_l1.setText("1. Train Dataset:");
        self.isf_l1.move(20, 270);
        self.image_single_foldered.append(self.isf_l1);

        self.isf_b1 = QPushButton('Select Folder', self)
        self.isf_b1.move(150,270)
        self.isf_b1.clicked.connect(self.select_train_dataset)
        self.image_single_foldered.append(self.isf_b1);

        self.isf_tb1 = QTextEdit(self)
        self.isf_tb1.move(250, 270)
        self.isf_tb1.resize(300, 30)
        self.isf_tb1.setText(self.system["traindata"]["dir"]);
        self.isf_tb1.setReadOnly(True)
        self.image_single_foldered.append(self.isf_tb1);



        self.isf_l2 = QLabel(self);
        self.isf_l2.setText("2. Val Dataset:");
        self.isf_l2.move(20, 320);
        self.image_single_foldered.append(self.isf_l2);


        
        self.isf_cb2 = QComboBox(self);
        self.project_list = ["yes", "no"]
        self.isf_cb2.addItems(self.project_list);
        index = self.isf_cb2.findText(self.system["val"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.isf_cb2.setCurrentIndex(index)
        self.isf_cb2.move(150, 320);
        self.isf_cb2.activated.connect(self.use_val);
        self.image_single_foldered.append(self.isf_cb2);


        self.isf_l2_1 = QLabel(self);
        self.isf_l2_1.setText(" (Optional)");
        self.isf_l2_1.move(20, 340);
        self.image_single_foldered.append(self.isf_l2_1);

        self.isf_b2 = QPushButton('Select Folder', self)
        self.isf_b2.move(230, 320)
        self.isf_b2.clicked.connect(self.select_val_dataset)
        self.image_single_foldered.append(self.isf_b2);
        if(self.isf_cb2.currentText() == "no"):
            self.isf_b2.setEnabled(False);

        self.isf_tb2 = QTextEdit(self)
        self.isf_tb2.move(230, 360)
        self.isf_tb2.resize(300, 50)
        self.isf_tb2.setText(self.system["valdata"]["dir"]);
        self.isf_tb2.setReadOnly(True)
        self.image_single_foldered.append(self.isf_tb2);
        if(self.isf_cb2.currentText() == "no"):
            self.isf_tb2.setEnabled(False);


        
        self.isc_l1 = QLabel(self);
        self.isc_l1.setText("1. Train Dataset:");
        self.isc_l1.move(20, 270);
        self.image_single_csv.append(self.isc_l1);

        self.isc_b1 = QPushButton('Select Folder', self)
        self.isc_b1.move(150,270)
        self.isc_b1.clicked.connect(self.select_train_dataset)
        self.image_single_csv.append(self.isc_b1);

        self.isc_tb1 = QTextEdit(self)
        self.isc_tb1.move(250, 270)
        self.isc_tb1.resize(300, 30)
        self.isc_tb1.setText(self.system["traindata"]["cdir"]);
        self.isc_tb1.setReadOnly(True)
        self.image_single_csv.append(self.isc_tb1);


        self.isc_l3 = QLabel(self);
        self.isc_l3.setText("2. Train Labels:");
        self.isc_l3.move(20, 320);
        self.image_single_csv.append(self.isc_l3);

        self.isc_b3 = QPushButton('Select File', self)
        self.isc_b3.move(150, 320)
        self.isc_b3.clicked.connect(self.select_train_csv)
        self.image_single_csv.append(self.isc_b3);

        self.isc_tb3 = QTextEdit(self)
        self.isc_tb3.move(250, 320)
        self.isc_tb3.resize(300, 30)
        self.isc_tb3.setText(self.system["traindata"]["csv"]);
        self.isc_tb3.setReadOnly(True)
        self.image_single_csv.append(self.isc_tb3);


        self.isc_l2 = QLabel(self);
        self.isc_l2.setText("3. Val Dataset:");
        self.isc_l2.move(20, 390);
        self.image_single_csv.append(self.isc_l2);


        
        self.isc_cb2 = QComboBox(self);
        self.project_list = ["yes", "no"]
        self.isc_cb2.addItems(self.project_list);
        index = self.isc_cb2.findText(self.system["val"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.isc_cb2.setCurrentIndex(index)
        self.isc_cb2.move(150, 390);
        self.isc_cb2.activated.connect(self.use_val);
        self.image_single_csv.append(self.isc_cb2);


        self.isc_l2_1 = QLabel(self);
        self.isc_l2_1.setText(" (Optional)");
        self.isc_l2_1.move(20, 410);
        self.image_single_csv.append(self.isc_l2_1);

        self.isc_b2 = QPushButton('Select Folder', self)
        self.isc_b2.move(230, 390)
        self.isc_b2.clicked.connect(self.select_val_dataset)
        self.image_single_csv.append(self.isc_b2);
        if(self.isc_cb2.currentText() == "no"):
            self.isc_b2.setEnabled(False);

        self.isc_tb2 = QTextEdit(self)
        self.isc_tb2.move(230, 430)
        self.isc_tb2.resize(300, 50)
        self.isc_tb2.setText(self.system["valdata"]["dir"]);
        self.isc_tb2.setReadOnly(True)
        self.image_single_csv.append(self.isc_tb2);
        if(self.isc_cb2.currentText() == "no"):
            self.isc_tb2.setEnabled(False);


        self.isc_l4 = QLabel(self);
        self.isc_l4.setText("4. Val Labels:");
        self.isc_l4.move(20, 510);
        self.image_single_csv.append(self.isc_l4);

        self.isc_l4_1 = QLabel(self);
        self.isc_l4_1.setText(" (Optional)");
        self.isc_l4_1.move(20, 530);
        self.image_single_csv.append(self.isc_l4_1);

        self.isc_b4 = QPushButton('Select File', self)
        self.isc_b4.move(150, 510)
        self.isc_b4.clicked.connect(self.select_val_csv)
        self.image_single_csv.append(self.isc_b4);
        if(self.isc_cb2.currentText() == "no"):
            self.isc_b4.setEnabled(False);

        self.isc_tb4 = QTextEdit(self)
        self.isc_tb4.move(250, 510)
        self.isc_tb4.resize(300, 50)
        self.isc_tb4.setText(self.system["valdata"]["csv"]);
        self.isc_tb4.setReadOnly(True)
        self.image_single_csv.append(self.isc_tb4);
        if(self.isc_cb2.currentText() == "no"):
            self.isc_tb4.setEnabled(False);







        if self.system["datatype"] == "image":
            self.datatype_image();
        elif self.system["datatype"] == "npy":
            self.datatype_npy();
        elif self.system["datatype"] == "hdf5":
            self.datatype_hdf5();
        elif self.system["datatype"] == "parquet":
            self.datatype_parquet();

        

    def use_val(self):
        if self.system["structuretype"] == "foldered":
            if(self.isf_cb2.currentText() == "no"):
                self.system["val"] = "no";
                self.isf_b2.setEnabled(False);
                self.isf_tb2.setEnabled(False);
            else:
                self.system["val"] = "yes";
                self.isf_b2.setEnabled(True);
                self.isf_tb2.setEnabled(True);
        else:
            if(self.isc_cb2.currentText() == "no"):
                self.system["val"] = "no";
                self.isc_b2.setEnabled(False);
                self.isc_tb2.setEnabled(False);
                self.isc_b4.setEnabled(False);
                self.isc_tb4.setEnabled(False);
            else:
                self.system["val"] = "yes";
                self.isc_b2.setEnabled(True);
                self.isc_tb2.setEnabled(True);
                self.isc_b4.setEnabled(True);
                self.isc_tb4.setEnabled(True);

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)




    def select_train_dataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", os.getcwd())
        if self.system["structuretype"] == "foldered":
            self.isf_b1.setText("Selected");
            self.isf_tb1.setText(folderName);
            self.system["traindata"]["dir"] = folderName;
        else:
            self.isc_b1.setText("Selected");
            self.isc_tb1.setText(folderName);
            self.system["traindata"]["cdir"] = folderName;
            print("In here");

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_val_dataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", os.getcwd())
        if self.system["structuretype"] == "foldered":
            self.isf_b2.setText("Selected");
            self.isf_tb2.setText(folderName);
            self.system["valdata"]["dir"] = folderName;
        else:
            self.isc_b2.setText("Selected");
            self.isc_tb2.setText(folderName);
            self.system["valdata"]["cdir"] = folderName;

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        

    def select_train_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd(), 
                                                    "Label Files (*.csv);;All Files (*)", options=options)

        self.isc_b3.setText("Selected");
        self.isc_tb3.setText(fileName);
        self.system["traindata"]["csv"] = fileName;

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_val_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd(), 
                                                    "Label Files (*.csv);;All Files (*)", options=options)

        self.isc_b4.setText("Selected");
        self.isc_tb4.setText(fileName);
        self.system["valdata"]["csv"] = fileName;

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)






    def createLayout_group_datatype(self):
        self.group_datatype = QGroupBox("Data Type", self)
        self.group_datatype.resize(550, 70);
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
        self.group_labeltype.resize(550, 70);
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
        self.group_structuretype.resize(550, 70);
        self.group_structuretype.move(20, 190);
        layout_groupbox = QHBoxLayout(self.group_structuretype)

        self.structuretype_1 = QRadioButton("Foldered Dataset", self)
        layout_groupbox.addWidget(self.structuretype_1)

        self.structuretype_2 = QRadioButton("CSV Dataset", self)
        layout_groupbox.addWidget(self.structuretype_2)

        if self.system["structuretype"] == "foldered":
            self.structuretype_1.setChecked(True)
        elif self.system["structuretype"] == "csv":
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
        for x in self.image_single_foldered:
            x.hide();
        for x in self.image_single_csv:
            x.hide();
        self.group_labeltype.hide();
        self.group_structuretype.hide();
        self.tb1.setText("Dataloader for Numpy files unimplemented");
        self.system["datatype"] = "npy";

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

    def datatype_hdf5(self):
        for x in self.image_single_foldered:
            x.hide();
        for x in self.image_single_csv:
            x.hide();
        self.group_labeltype.hide();
        self.group_structuretype.hide();
        self.tb1.setText("Dataloader for Hdf5 files unimplemented");
        self.system["datatype"] = "hdf5";

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

    def datatype_parquet(self):
        for x in self.image_single_foldered:
            x.hide();
        for x in self.image_single_csv:
            x.hide();
        self.group_labeltype.hide();
        self.group_structuretype.hide();
        self.tb1.setText("Dataloader for Parquet files unimplemented");
        self.system["datatype"] = "parquet";

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)






    def labeltype_single(self):
        self.group_structuretype.show();
        if(self.system["structuretype"] == "foldered"):
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
        self.tb1.setText("Dataloader for Multi label classification unimplemented");
        self.system["labeltype"] = "multi";

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)



    def structuretype_foldered(self):
        for x in self.image_single_foldered:
            x.show();
        for x in self.image_single_csv:
            x.hide();
        self.system["structuretype"] = "foldered";
        self.tb1.setText(self.writeup_image_single_foldered());
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

    def structuretype_csv(self):
        for x in self.image_single_foldered:
            x.hide();
        for x in self.image_single_csv:
            x.show();

        self.system["structuretype"] = "csv";
        self.tb1.setText(self.writeup_image_single_csv());
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)




    def writeup_image_single_foldered(self):
        wr = "";
        wr += "Foldered Dataset\n"
        wr += "Dataset Structure Example\n\n";
        wr += "Parent Folder\n";
        wr += "       |\n";
        wr += "       |\n";
        wr += "       |---TrainImages\n";
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
        wr += "       |---TrainImages\n";
        wr += "                 |--img1.jpg\n";
        wr += "                 |--img2.jpg\n";
        wr += "                 |--...(so on)\n";
        wr += "       |---ValImages\n";
        wr += "                 |--img1.jpg\n";
        wr += "                 |--img2.jpg\n";
        wr += "                 |--...(so on)\n\n";
        wr += "       |---train_labels.csv\n";
        wr += "       |---val_labels.csv\n\n\n";
        wr += "Annotation Format\n";
        wr += "       | Id         | Labels  |\n";
        wr += "       | img1.jpg   | label1  |\n";
        wr += "       | img2.jpg   | label2  |\n";

        return wr;





    def forward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_model_param.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_experiment_run_mode.emit();



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainQuickDataParam()
screen.show()
sys.exit(app.exec_())
'''