import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowObj7Yolov3DataParam(QtWidgets.QWidget):

    backward_7_yolov3 = QtCore.pyqtSignal();
    forward_valdata_param = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Yolo V3 - Training Data Param'
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 600
        self.cfg_setup();
        self.initUI();


    def cfg_setup(self):
        if(os.path.isfile("obj_7_yolov3.json")):
            with open('obj_7_yolov3.json') as json_file:
                self.system = json.load(json_file)
        else:
            self.system = {};
            self.system["anno_type"] = "monk"
            self.system["val_anno_type"] = "monk"

            self.system["monk_root_dir"] = "Monk_Object_Detection/example_notebooks/sample_dataset/ship/";
            self.system["monk_img_dir"] = "Images";
            self.system["monk_anno_file"] = "train_labels.csv";

            self.system["voc_root_dir"] = "Monk_Object_Detection/example_notebooks/sample_dataset/ship/";
            self.system["voc_img_dir"] = "Images";
            self.system["voc_anno_dir"] = "";

            self.system["coco_root_dir"] = "Monk_Object_Detection/example_notebooks/sample_dataset/";
            self.system["coco_coco_dir"] = "ship";
            self.system["coco_img_dir"] = "";
            self.system["coco_set_dir"] = "Images"

            self.system["yolo_root_dir"] = "Monk_Object_Detection/example_notebooks/sample_dataset/ship/";
            self.system["yolo_img_dir"] = "Images";
            self.system["yolo_anno_dir"] = "";
            self.system["yolo_classes_file"] = "classes.txt";


            self.system["val_data"] = "no";

            self.system["val_monk_root_dir"] = "Monk_Object_Detection/example_notebooks/sample_dataset/ship/";
            self.system["val_monk_img_dir"] = "Images";
            self.system["val_monk_anno_file"] = "train_labels.csv";

            self.system["val_voc_root_dir"] = "Monk_Object_Detection/example_notebooks/sample_dataset/ship/";
            self.system["val_voc_img_dir"] = "Images";
            self.system["val_voc_anno_dir"] = "";

            self.system["val_coco_root_dir"] = "Monk_Object_Detection/example_notebooks/sample_dataset/";
            self.system["val_coco_coco_dir"] = "ship";
            self.system["val_coco_img_dir"] = "";
            self.system["val_coco_set_dir"] = "Images";

            self.system["val_yolo_root_dir"] = "Monk_Object_Detection/example_notebooks/sample_dataset/ship/";
            self.system["val_yolo_img_dir"] = "Images";
            self.system["val_yolo_anno_dir"] = "";
            self.system["val_yolo_classes_file"] = "classes.txt";




            self.system["batch_size"] = "4";
            

            self.system["model"] = "yolov3";
            self.system["devices"] = "0";
            self.system["use_gpu"] = "yes"

            self.system["lr"] = "0.00579";
            self.system["optimizer"] = "sgd";
            self.system["multi_scale"] = "no";
            self.system["evolve"] = "no";
            self.system["mixed_precision"] = "no";
            self.system["img_size"] = "416";
            self.system["cache_images"] = "no";

            self.system["num_generations"] = "2";
            self.system["epochs"] = "2";


            with open('obj_7_yolov3.json', 'w') as outfile:
                json.dump(self.system, outfile)



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
        self.b2.clicked.connect(self.forward);

        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(800,550)
        self.b3.clicked.connect(self.close)

        
        self.tb1 = QTextEdit(self)
        self.tb1.move(20, 20)
        self.tb1.resize(400, 450)
        self.tb1.setText(self.monk_format());
        self.tb1.setReadOnly(True)
        

        self.r1 = QRadioButton("Monk format", self)
        if self.system["anno_type"] == "monk":
            self.r1.setChecked(True)
        self.r1.move(430,20)
        self.r1.toggled.connect(self.monk);

        self.r2 = QRadioButton("VOC format", self)
        if self.system["anno_type"] == "voc":
            self.r2.setChecked(True)
        self.r2.move(560,20)
        self.r2.toggled.connect(self.voc);

        self.r3 = QRadioButton("COCO format", self)
        if self.system["anno_type"] == "coco":
            self.r3.setChecked(True)
        self.r3.move(670,20)
        self.r3.toggled.connect(self.coco);

        self.r4 = QRadioButton("Yolo format", self)
        if self.system["anno_type"] == "yolo":
            self.r4.setChecked(True)
        self.r4.move(790,20)
        self.r4.toggled.connect(self.yolo);


        
        self.m = [];
        self.m_l1 = QLabel(self);
        self.m_l1.setText("1. root:");
        self.m_l1.move(450, 70);
        self.m.append(self.m_l1);

        self.m_b1 = QPushButton('Select Folder', self)
        self.m_b1.move(550,70)
        self.m_b1.clicked.connect(self.select_root_dataset)
        self.m.append(self.m_b1);

        self.m_tb1 = QTextEdit(self)
        self.m_tb1.move(450, 100)
        self.m_tb1.resize(300, 50)
        self.m_tb1.setText(self.system["monk_root_dir"]);
        self.m_tb1.setReadOnly(True)
        self.m.append(self.m_tb1);


        self.m_l2 = QLabel(self);
        self.m_l2.setText("2. img_dir:");
        self.m_l2.move(450, 180);
        self.m.append(self.m_l2);

        self.m_b2 = QPushButton('Select Folder', self)
        self.m_b2.move(550,180)
        self.m_b2.clicked.connect(self.select_img_dir)
        self.m.append(self.m_b2);

        self.m_tb2 = QTextEdit(self)
        self.m_tb2.move(450, 210)
        self.m_tb2.resize(300, 50)
        self.m_tb2.setText(self.system["monk_img_dir"]);
        self.m_tb2.setReadOnly(True)
        self.m.append(self.m_tb2);

        self.m_l3 = QLabel(self);
        self.m_l3.setText("3. anno_file:");
        self.m_l3.move(450, 280);
        self.m.append(self.m_l3);

        self.m_b3 = QPushButton('Select File', self)
        self.m_b3.move(550, 280)
        self.m_b3.clicked.connect(self.select_anno_file)
        self.m.append(self.m_b3);

        self.m_tb3 = QTextEdit(self)
        self.m_tb3.move(450, 310)
        self.m_tb3.resize(300, 50)
        self.m_tb3.setText(self.system["monk_anno_file"]);
        self.m_tb3.setReadOnly(True)
        self.m.append(self.m_tb3);

        self.m_l4 = QLabel(self);
        self.m_l4.setText("4. batch_size:");
        self.m_l4.move(450, 380);
        self.m.append(self.m_l4);

        self.m_e4 = QLineEdit(self)
        self.m_e4.move(550, 380);
        self.m_e4.setText(self.system["batch_size"]);
        self.m_e4.resize(200, 25);
        self.m.append(self.m_e4);
        



        
        self.v = [];
        self.v_l1 = QLabel(self);
        self.v_l1.setText("1. root:");
        self.v_l1.move(450, 70);
        self.v.append(self.v_l1);


        self.v_b1 = QPushButton('Select Folder', self)
        self.v_b1.move(550,70)
        self.v_b1.clicked.connect(self.select_root_dataset)
        self.v.append(self.v_b1);

        self.v_tb1 = QTextEdit(self)
        self.v_tb1.move(450, 100)
        self.v_tb1.resize(300, 50)
        self.v_tb1.setText(self.system["voc_root_dir"]);
        self.v_tb1.setReadOnly(True)
        self.v.append(self.v_tb1);

        self.v_l2 = QLabel(self);
        self.v_l2.setText("2. img_dir:");
        self.v_l2.move(450, 180);
        self.v.append(self.v_l2);

        self.v_b2 = QPushButton('Select Folder', self)
        self.v_b2.move(550,180)
        self.v_b2.clicked.connect(self.select_img_dir)
        self.v.append(self.v_b2);

        self.v_tb2 = QTextEdit(self)
        self.v_tb2.move(450, 210)
        self.v_tb2.resize(300, 50)
        self.v_tb2.setText(self.system["voc_img_dir"]);
        self.v_tb2.setReadOnly(True)
        self.v.append(self.v_tb2);

        self.v_l3 = QLabel(self);
        self.v_l3.setText("3. anno_dir:");
        self.v_l3.move(450, 280);
        self.v.append(self.v_l3);

        self.v_b3 = QPushButton('Select Folder', self)
        self.v_b3.move(550, 280)
        self.v_b3.clicked.connect(self.select_anno_dir)
        self.v.append(self.v_b3);

        self.v_tb3 = QTextEdit(self)
        self.v_tb3.move(450, 310)
        self.v_tb3.resize(300, 50)
        self.v_tb3.setText(self.system["voc_anno_dir"]);
        self.v_tb3.setReadOnly(True)
        self.v.append(self.v_tb3);

        self.v_l4 = QLabel(self);
        self.v_l4.setText("4. batch_size:");
        self.v_l4.move(450, 380);
        self.v.append(self.v_l4);

        self.v_e4 = QLineEdit(self)
        self.v_e4.move(550, 380);
        self.v_e4.setText(self.system["batch_size"]);
        self.v_e4.resize(200, 25);
        self.v.append(self.v_e4);




        self.c = [];
        self.c_l1 = QLabel(self);
        self.c_l1.setText("1. root:");
        self.c_l1.move(450, 70);
        self.c.append(self.c_l1);

        self.c_b1 = QPushButton('Select Folder', self)
        self.c_b1.move(550,70)
        self.c_b1.clicked.connect(self.select_root_dataset)
        self.c.append(self.c_b1);

        self.c_tb1 = QTextEdit(self)
        self.c_tb1.move(450, 100)
        self.c_tb1.resize(300, 50)
        self.c_tb1.setText(self.system["coco_root_dir"]);
        self.c_tb1.setReadOnly(True)
        self.c.append(self.c_tb1);


        self.c_l2 = QLabel(self);
        self.c_l2.setText("2. coco_dir:");
        self.c_l2.move(450, 180);
        self.c.append(self.c_l2);

        self.c_b2 = QPushButton('Select Folder', self)
        self.c_b2.move(550,180)
        self.c_b2.clicked.connect(self.select_coco_dir)
        self.c.append(self.c_b2);

        self.c_b2_1 = QPushButton('Set Blank', self)
        self.c_b2_1.move(650,180)
        self.c_b2_1.clicked.connect(self.select_coco_dir_blank)
        self.c.append(self.c_b2_1);

        self.c_tb2 = QTextEdit(self)
        self.c_tb2.move(450, 210)
        self.c_tb2.resize(300, 50)
        self.c_tb2.setText(self.system["coco_coco_dir"]);
        self.c_tb2.setReadOnly(True)
        self.c.append(self.c_tb2);


        self.c_l3 = QLabel(self);
        self.c_l3.setText("3. img_dir:");
        self.c_l3.move(450, 280);
        self.c.append(self.c_l3);

        self.c_b3 = QPushButton('Select Folder', self)
        self.c_b3.move(550,280)
        self.c_b3.clicked.connect(self.select_img_dir)
        self.c.append(self.c_b3);

        self.c_b3_1 = QPushButton('Set Blank', self)
        self.c_b3_1.move(650,280)
        self.c_b3_1.clicked.connect(self.select_img_dir_blank)
        self.c.append(self.c_b3_1);

        self.c_tb3 = QTextEdit(self)
        self.c_tb3.move(450, 310)
        self.c_tb3.resize(300, 50)
        self.c_tb3.setText(self.system["coco_img_dir"]);
        self.c_tb3.setReadOnly(True)
        self.c.append(self.c_tb3);


        self.c_l5 = QLabel(self);
        self.c_l5.setText("4. set_dir:");
        self.c_l5.move(450, 380);
        self.c.append(self.c_l5);

        self.c_b5 = QPushButton('Select Folder', self)
        self.c_b5.move(550,380)
        self.c_b5.clicked.connect(self.select_set_dir)
        self.c.append(self.c_b5);

        self.c_b5_1 = QPushButton('Set Blank', self)
        self.c_b5_1.move(650,380)
        self.c_b5_1.clicked.connect(self.select_set_dir_blank)
        self.c.append(self.c_b5_1);

        self.c_tb5 = QTextEdit(self)
        self.c_tb5.move(450, 410)
        self.c_tb5.resize(300, 50)
        self.c_tb5.setText(self.system["coco_set_dir"]);
        self.c_tb5.setReadOnly(True)
        self.c.append(self.c_tb5);



        self.c_l4 = QLabel(self);
        self.c_l4.setText("5. batch_size:");
        self.c_l4.move(450, 480);
        self.c.append(self.c_l4);

        self.c_e4 = QLineEdit(self)
        self.c_e4.move(550, 480);
        self.c_e4.setText(self.system["batch_size"]);
        self.c_e4.resize(200, 25);
        self.c.append(self.c_e4);

    

        self.y = [];
        self.y_l1 = QLabel(self);
        self.y_l1.setText("1. root:");
        self.y_l1.move(450, 70);
        self.y.append(self.y_l1);


        self.y_b1 = QPushButton('Select Folder', self)
        self.y_b1.move(550,70)
        self.y_b1.clicked.connect(self.select_root_dataset)
        self.y.append(self.y_b1);

        self.y_tb1 = QTextEdit(self)
        self.y_tb1.move(450, 100)
        self.y_tb1.resize(300, 50)
        self.y_tb1.setText(self.system["yolo_root_dir"]);
        self.y_tb1.setReadOnly(True)
        self.y.append(self.y_tb1);

        self.y_l2 = QLabel(self);
        self.y_l2.setText("2. img_dir:");
        self.y_l2.move(450, 180);
        self.y.append(self.y_l2);

        self.y_b2 = QPushButton('Select Folder', self)
        self.y_b2.move(550,180)
        self.y_b2.clicked.connect(self.select_img_dir)
        self.y.append(self.y_b2);

        self.y_tb2 = QTextEdit(self)
        self.y_tb2.move(450, 210)
        self.y_tb2.resize(300, 50)
        self.y_tb2.setText(self.system["yolo_img_dir"]);
        self.y_tb2.setReadOnly(True)
        self.y.append(self.y_tb2);

        self.y_l3 = QLabel(self);
        self.y_l3.setText("3. anno_dir:");
        self.y_l3.move(450, 280);
        self.y.append(self.y_l3);

        self.y_b3 = QPushButton('Select Folder', self)
        self.y_b3.move(550, 280)
        self.y_b3.clicked.connect(self.select_yolo_anno_dir)
        self.y.append(self.y_b3);

        self.y_tb3 = QTextEdit(self)
        self.y_tb3.move(450, 310)
        self.y_tb3.resize(300, 50)
        self.y_tb3.setText(self.system["yolo_anno_dir"]);
        self.y_tb3.setReadOnly(True)
        self.y.append(self.y_tb3);


        self.y_l5 = QLabel(self);
        self.y_l5.setText("4. Classes file:");
        self.y_l5.move(450, 380);
        self.y.append(self.y_l5);


        self.y_b5 = QPushButton('Select File', self)
        self.y_b5.move(550, 380)
        self.y_b5.clicked.connect(self.select_yolo_classes_file)
        self.y.append(self.y_b5);

        self.y_tb5 = QTextEdit(self)
        self.y_tb5.move(450, 410)
        self.y_tb5.resize(300, 50)
        self.y_tb5.setText(self.system["yolo_classes_file"]);
        self.y_tb5.setReadOnly(True)
        self.y.append(self.y_tb5);


        self.y_l4 = QLabel(self);
        self.y_l4.setText("5. batch_size:");
        self.y_l4.move(450, 480);
        self.y.append(self.y_l4);

        self.y_e4 = QLineEdit(self)
        self.y_e4.move(550, 480);
        self.y_e4.setText(self.system["batch_size"]);
        self.y_e4.resize(200, 25);
        self.y.append(self.y_e4);





        if self.system["anno_type"] == "monk":
            self.monk();
        elif self.system["anno_type"] == "voc":
            self.voc();
        elif self.system["anno_type"] == "coco":
            self.coco();
        else:
            self.yolo();
        


        

    def monk(self):
        self.tb1.setText(self.monk_format());
        for x in self.m:
            x.show();
        for x in self.v:
            x.hide();
        for x in self.c:
            x.hide();
        for x in self.y:
            x.hide();


    def voc(self):
        self.tb1.setText(self.voc_format());
        for x in self.m:
            x.hide();
        for x in self.v:
            x.show();
        for x in self.c:
            x.hide();
        for x in self.y:
            x.hide();


    def coco(self):
        self.tb1.setText(self.coco_format());
        for x in self.m:
            x.hide();
        for x in self.v:
            x.hide();
        for x in self.c:
            x.show();
        for x in self.y:
            x.hide();


    def yolo(self):
        self.tb1.setText(self.yolo_format());
        for x in self.m:
            x.hide();
        for x in self.v:
            x.hide();
        for x in self.c:
            x.hide();
        for x in self.y:
            x.show();




    def select_root_dataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", os.getcwd())
        if self.r1.isChecked():
            self.m_b1.setText("Selected");
            self.m_tb1.setText(folderName);
            self.system["monk_root_dir"] = folderName;
        if self.r2.isChecked():
            self.v_b1.setText("Selected");
            self.v_tb1.setText(folderName);
            self.system["voc_root_dir"] = folderName;
        if self.r3.isChecked():
            self.c_b1.setText("Selected");
            self.c_tb1.setText(folderName);
            self.system["coco_root_dir"] = folderName;
        if self.r4.isChecked():
            self.y_b1.setText("Selected");
            self.y_tb1.setText(folderName);
            self.system["yolo_root_dir"] = folderName;


        


    def select_coco_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", self.system["coco_root_dir"])
        folderName = folderName.split("/")[-1];
        self.c_b2.setText("Selected");
        self.c_tb2.setText(folderName);
        self.system["coco_coco_dir"] = folderName;


    def select_coco_dir_blank(self):
        self.c_tb2.setText("");
        self.system["coco_coco_dir"] = "";


    def select_img_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        if self.r1.isChecked():
            folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", self.system["monk_root_dir"])
        if self.r2.isChecked():
            folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", self.system["voc_root_dir"])
        if self.r3.isChecked():
            folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", 
                self.system["coco_root_dir"] + "/" + self.system["coco_coco_dir"])
        if self.r4.isChecked():
            folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", self.system["yolo_root_dir"])

        folderName = folderName.split("/")[-1];
        if self.r1.isChecked():
            self.m_b2.setText("Selected");
            self.m_tb2.setText(folderName);
            self.system["monk_img_dir"] = folderName;
        if self.r2.isChecked():
            self.v_b2.setText("Selected");
            self.v_tb2.setText(folderName);
            self.system["voc_img_dir"] = folderName;
        if self.r3.isChecked():
            self.c_b3.setText("Selected");
            self.c_tb3.setText(folderName);
            self.system["coco_img_dir"] = folderName;
        if self.r4.isChecked():
            self.y_b3.setText("Selected");
            self.y_tb3.setText(folderName);
            self.system["yolo_img_dir"] = folderName;

    def select_img_dir_blank(self):
        self.c_tb3.setText("");
        self.system["coco_img_dir"] = "";

        

    def select_anno_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", self.system["monk_root_dir"], 
                                                    "Monk Project Files (*.csv);;All Files (*)", options=options)
        if fileName:
            fileName = fileName.split("/")[-1];
            self.system["monk_anno_file"] = fileName;
            self.m_b3.setText("Selected");
            self.m_tb3.setText(fileName);


    def select_anno_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", self.system["voc_root_dir"])
        folderName = folderName.split("/")[-1];
        self.v_b3.setText("Selected");
        self.v_tb3.setText(folderName);
        self.system["voc_anno_dir"] = folderName;


    def select_yolo_anno_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", self.system["yolo_root_dir"])
        folderName = folderName.split("/")[-1];
        self.y_b3.setText("Selected");
        self.y_tb3.setText(folderName);
        self.system["yolo_anno_dir"] = folderName;


    def select_set_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", 
            self.system["coco_root_dir"] + "/" + self.system["coco_coco_dir"] + "/" + self.system["coco_img_dir"])
        folderName = folderName.split("/")[-1];
        self.c_b5.setText("Selected");
        self.c_tb5.setText(folderName);
        self.system["coco_set_dir"] = folderName;

    def select_set_dir_blank(self):
        self.c_tb5_1.setText("");
        self.system["coco_set_dir"] = "";


    def select_yolo_classes_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", os.getcwd(), 
                                                    "Text files (*.txt);;All Files (*)", 
                                                    options=options)
        fileName = fileName.split("/")[-1];
        self.system["yolo_classes_file"] = fileName;
        self.y_b5.setText("Selected");
        self.y_tb5.setText(fileName);

        with open('obj_7_yolov3_infer.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def forward(self):
        if self.r1.isChecked():
            self.system["anno_type"] = "monk";
            self.system["batch_size"] = self.m_e4.text();
        if self.r2.isChecked():
            self.system["anno_type"] = "voc";
            self.system["batch_size"] = self.v_e4.text();
        if self.r3.isChecked():
            self.system["anno_type"] = "coco";
            self.system["batch_size"] = self.c_e4.text();
        if self.r4.isChecked():
            self.system["anno_type"] = "yolo";
            self.system["batch_size"] = self.y_e4.text();

        with open('obj_7_yolov3.json', 'w') as outfile:
            json.dump(self.system, outfile)


        self.forward_valdata_param.emit();



    def backward(self):
        if self.r1.isChecked():
            self.system["anno_type"] = "monk";
            self.system["batch_size"] = self.m_e4.text();
        if self.r2.isChecked():
            self.system["anno_type"] = "voc";
            self.system["batch_size"] = self.v_e4.text();
        if self.r3.isChecked():
            self.system["anno_type"] = "coco";
            self.system["batch_size"] = self.c_e4.text();
        if self.r4.isChecked():
            self.system["anno_type"] = "yolo";
            self.system["batch_size"] = self.y_e4.text();

        with open('obj_7_yolov3.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.backward_7_yolov3.emit();


    def monk_format(self):
        wr = "";
        wr += "Monk Type Data Format\n"
        wr += "Dataset Directory Structure\n\n";
        wr += "Parent_Directory (root)\n";
        wr += "      |\n";
        wr += "      |-----------Images (img_dir)\n";
        wr += "      |              |\n";
        wr += "      |              |------------------img1.jpg\n";
        wr += "      |              |------------------img2.jpg\n";
        wr += "      |              |------------------.........(and so on)\n";
        wr += "      |\n";
        wr += "      |\n";
        wr += "      |-----------train_labels.csv (anno_file)\n\n";
        wr += "Annotation file format\n";
        wr += "       | Id         | Labels                                 |\n";
        wr += "       | img1.jpg   | x1 y1 x2 y2 label1 x1 y1 x2 y2 label2  |\n";
        wr += "    Labels: xmin ymin xmax ymax label\n";
        wr += "    xmin, ymin - top left corner of bounding box\n";
        wr += "    xmax, ymax - bottom right corner of bounding box\n";


        return wr;


    def voc_format(self):
        wr = "";
        wr += "VOC Type Data Format\n"
        wr += "Dataset Directory Structure\n\n";
        wr += "Parent_Directory (root)\n";
        wr += "      |\n";
        wr += "      |-----------Images (img_dir)\n";
        wr += "      |              |\n";
        wr += "      |              |------------------img1.jpg\n";
        wr += "      |              |------------------img2.jpg\n";
        wr += "      |              |------------------.........(and so on)\n";
        wr += "      |\n";
        wr += "      |\n";
        wr += "      |-----------Annotations (anno_dir)\n";
        wr += "      |              |\n";
        wr += "      |              |------------------img1.xml\n";
        wr += "      |              |------------------img2.xml\n";
        wr += "      |              |------------------.........(and so on)\n";

        return wr;


    def coco_format(self):
        wr = "";
        wr += "COCO Type Data Format\n"
        wr += "Dataset Directory Structure\n\n";
        wr += "Parent_Directory (root_dir)\n";
        wr += "      |\n";
        wr += "      |--------ship (coco_dir)\n";
        wr += "      |         |\n";
        wr += "      |         |---Images (set_dir)\n";
        wr += "      |         |----|\n";
        wr += "      |              |-------------------img1.jpg\n";
        wr += "      |              |-------------------img2.jpg\n";
        wr += "      |              |------------------.........(and so on)\n";
        wr += "      |\n";
        wr += "      |\n";
        wr += "      |         |---annotations\n";
        wr += "      |         |----|\n";
        wr += "      |              |--------------------instances_Images.json\n";
        wr += "      |              |--------------------classes.txt\n"
        wr += "\n";
        wr += "\n";
        wr += "    instances_Images.json -> In proper COCO format\n";
        wr += "    Note: Annotation file name too coincides against the set_dir\n";
        wr += "    classes.txt -> A list of classes in alphabetical order\n";

        return wr;


    def yolo_format(self):
        wr = "";
        wr += "YOLO Type Data Format\n";
        wr += "Dataset Directory Structure\n\n";
        wr += "Parent_Directory (root_dir)\n";
        wr += "      |\n";
        wr += "      |-----------Images (img_dir)\n";
        wr += "      |              |\n";
        wr += "      |              |------------------img1.jpg\n";
        wr += "      |              |------------------img2.jpg\n";
        wr += "      |              |------------------.........(and so on)\n";
        wr += "      |\n";
        wr += "      |\n";
        wr += "      |-----------Annotations (anno_dir)\n";
        wr += "      |              |\n";
        wr += "      |              |------------------img1.txt\n";
        wr += "      |              |------------------img2.txt\n";
        wr += "      |              |------------------.........(and so on)\n";
        wr += "\n\n";
        wr += "Classes file\n";
        wr += "-List of classes in every new line.\n";
        wr += "-The order corresponds to the IDs in annotation files\n";
        wr += "-Example.\n";
        wr += "     class1               (------------------------------> if will be 0)\n";
        wr += "     class2               (------------------------------> if will be 1)\n";
        wr += "     class3               (------------------------------> if will be 2)\n";
        wr += "     class4               (------------------------------> if will be 3)\n";
        wr += "\n\n";
        wr += "Annotation file format\n";
        wr += "-CLASS_ID BOX_X_CENTER BOX_Y_CENTER WIDTH BOX_WIDTH BOX_HEIGHT\n";
        wr += "-(All the coordinates should be normalized)\n";
        wr += "-(X coordinates divided by width of image, Y coordinates divided by height of image)\n";
        wr += "-Ex. (One line per bounding box of object in image)\n";
        wr += "    class_id x1 y1 w h\n";
        wr += "    class_id x1 y1 w h\n";
        wr += "    ..... (and so on)\n";

        return wr;


'''
app = QApplication(sys.argv)
screen = WindowObj7Yolov3DataParam()
screen.show()
sys.exit(app.exec_())
'''