import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateTransformParam(QtWidgets.QWidget):

    forward_model_param = QtCore.pyqtSignal();
    backward_data_param = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Update Transform Params'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 600
        self.transform_ui_mxnet = [];
        self.transform_ui_keras = [];
        self.transform_ui_pytorch = [];
        self.current_transform = {};
        self.current_transform["name"] = "";
        self.current_transform["params"] = {};
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
        self.b3.move(800,550)
        self.b3.clicked.connect(self.close)


        self.cb1 = QComboBox(self);
        self.cb1.move(20, 20);
        self.cb1.activated.connect(self.select_transform);

        self.cb2 = QComboBox(self);
        self.cb2.move(20, 20);
        self.cb2.activated.connect(self.select_transform);

        self.cb3 = QComboBox(self);
        self.cb3.move(20, 20);
        self.cb3.activated.connect(self.select_transform);



        self.mxnet_transforms_list = ["select", "apply_random_resized_crop", "apply_center_crop", "apply_color_jitter", "apply_random_horizontal_flip",
                                    "apply_random_vertical_flip", "apply_random_lighting", "apply_resize", "apply_normalize"];
        self.keras_transforms_list = ["select", "apply_color_jitter", "apply_random_affine", "apply_random_horizontal_flip", 
                                    "apply_random_vertical_flip", "apply_random_rotation", "apply_mean_subtraction", 
                                    "apply_normalize"];
        self.pytorch_transforms_list =  ["select", "apply_center_crop", "apply_color_jitter", "apply_random_affine", "apply_random_crop", 
                                    "apply_random_horizontal_flip", "apply_random_perspective", "apply_random_resized_crop", 
                                    "apply_random_rotation", "apply_random_vertical_flip",
                                    "apply_resize", "apply_normalize"];
        

        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.cb1.addItems(self.mxnet_transforms_list);
            self.cb1.show();
            self.cb2.hide();
            self.cb3.hide();
        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.cb2.addItems(self.keras_transforms_list);
            self.cb2.show();
            self.cb1.hide();
            self.cb3.hide();
        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.cb3.addItems(self.pytorch_transforms_list);
            self.cb3.show();
            self.cb1.hide();
            self.cb2.hide();


        tmp = [];
        self.mx_tf1_l1 = QLabel(self);
        self.mx_tf1_l1.setText("1. Crop Size:");
        self.mx_tf1_l1.move(20, 100);
        tmp.append(self.mx_tf1_l1);

        self.mx_tf1_e1 = QLineEdit(self)
        self.mx_tf1_e1.move(150, 100);
        self.mx_tf1_e1.setText("224");
        tmp.append(self.mx_tf1_e1);



        
        self.mx_tf1_l2 = QLabel(self);
        self.mx_tf1_l2.setText("2. Scale limits");
        self.mx_tf1_l2.move(20, 150);
        tmp.append(self.mx_tf1_l2);

        self.mx_tf1_e2_1 = QLineEdit(self)
        self.mx_tf1_e2_1.move(150, 150);
        self.mx_tf1_e2_1.setText("0.08");
        tmp.append(self.mx_tf1_e2_1);

        self.mx_tf1_e2_2 = QLineEdit(self)
        self.mx_tf1_e2_2.move(300, 150);
        self.mx_tf1_e2_2.setText("1.0");
        tmp.append(self.mx_tf1_e2_2);



        
        self.mx_tf1_l3 = QLabel(self);
        self.mx_tf1_l3.setText("3. Aspect ratio limits");
        self.mx_tf1_l3.move(20, 200);
        tmp.append(self.mx_tf1_l3);

        self.mx_tf1_e3_1 = QLineEdit(self)
        self.mx_tf1_e3_1.move(180, 200);
        self.mx_tf1_e3_1.setText("0.75");
        tmp.append(self.mx_tf1_e3_1);

        self.mx_tf1_e3_2 = QLineEdit(self)
        self.mx_tf1_e3_2.move(330, 200);
        self.mx_tf1_e3_2.setText("1.33");
        tmp.append(self.mx_tf1_e3_2);



        
        self.mx_tf1_l4 = QLabel(self);
        self.mx_tf1_l4.setText("4. Apply at");
        self.mx_tf1_l4.move(20, 250);
        tmp.append(self.mx_tf1_l4);

        self.mx_tf1_cbox1 = QCheckBox("Training", self)
        self.mx_tf1_cbox1.setChecked(True)
        self.mx_tf1_cbox1.move(110, 250);
        tmp.append(self.mx_tf1_cbox1);

        self.mx_tf1_cbox2 = QCheckBox("Validation", self)
        self.mx_tf1_cbox2.setChecked(True)
        self.mx_tf1_cbox2.move(210, 250);
        tmp.append(self.mx_tf1_cbox2);

        self.mx_tf1_cbox3 = QCheckBox("Testing", self)
        self.mx_tf1_cbox3.setChecked(False)
        self.mx_tf1_cbox3.move(310, 250);
        tmp.append(self.mx_tf1_cbox3);

        self.transform_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_tf2_l1 = QLabel(self);
        self.mx_tf2_l1.setText("1. Crop Size:");
        self.mx_tf2_l1.move(20, 100);
        tmp.append(self.mx_tf2_l1);

        self.mx_tf2_e1 = QLineEdit(self)
        self.mx_tf2_e1.move(150, 100);
        self.mx_tf2_e1.setText("224");
        tmp.append(self.mx_tf2_e1);


        self.mx_tf2_l2 = QLabel(self);
        self.mx_tf2_l2.setText("2. Apply at");
        self.mx_tf2_l2.move(20, 150);
        tmp.append(self.mx_tf2_l2);

        self.mx_tf2_cbox1 = QCheckBox("Training", self)
        self.mx_tf2_cbox1.setChecked(True)
        self.mx_tf2_cbox1.move(110, 150);
        tmp.append(self.mx_tf2_cbox1);

        self.mx_tf2_cbox2 = QCheckBox("Validation", self)
        self.mx_tf2_cbox2.setChecked(True)
        self.mx_tf2_cbox2.move(210, 150);
        tmp.append(self.mx_tf2_cbox2);

        self.mx_tf2_cbox3 = QCheckBox("Testing", self)
        self.mx_tf2_cbox3.setChecked(False)
        self.mx_tf2_cbox3.move(310, 150);
        tmp.append(self.mx_tf2_cbox3);

        self.transform_ui_mxnet.append(tmp)







        tmp = [];
        self.mx_tf3_l1 = QLabel(self);
        self.mx_tf3_l1.setText("1. Brightness (0-1):");
        self.mx_tf3_l1.move(20, 100);
        tmp.append(self.mx_tf3_l1);

        self.mx_tf3_e1 = QLineEdit(self)
        self.mx_tf3_e1.move(150, 100);
        self.mx_tf3_e1.setText("0.0");
        tmp.append(self.mx_tf3_e1);


        self.mx_tf3_l2 = QLabel(self);
        self.mx_tf3_l2.setText("2. Contrast (0-1):");
        self.mx_tf3_l2.move(20, 150);
        tmp.append(self.mx_tf3_l2);

        self.mx_tf3_e2 = QLineEdit(self)
        self.mx_tf3_e2.move(150, 150);
        self.mx_tf3_e2.setText("0.0");
        tmp.append(self.mx_tf3_e2);


        self.mx_tf3_l3 = QLabel(self);
        self.mx_tf3_l3.setText("3. Saturation (0-1):");
        self.mx_tf3_l3.move(20, 200);
        tmp.append(self.mx_tf3_l3);

        self.mx_tf3_e3 = QLineEdit(self)
        self.mx_tf3_e3.move(150, 200);
        self.mx_tf3_e3.setText("0.0");
        tmp.append(self.mx_tf3_e3);


        self.mx_tf3_l4 = QLabel(self);
        self.mx_tf3_l4.setText("4. Hue (0-1):");
        self.mx_tf3_l4.move(20, 250);
        tmp.append(self.mx_tf3_l4);

        self.mx_tf3_e4 = QLineEdit(self)
        self.mx_tf3_e4.move(150, 250);
        self.mx_tf3_e4.setText("0.0");
        tmp.append(self.mx_tf3_e4);


        self.mx_tf3_l5 = QLabel(self);
        self.mx_tf3_l5.setText("5. Apply at");
        self.mx_tf3_l5.move(20, 300);
        tmp.append(self.mx_tf3_l5);

        self.mx_tf3_cbox1 = QCheckBox("Training", self)
        self.mx_tf3_cbox1.setChecked(True)
        self.mx_tf3_cbox1.move(110, 300);
        tmp.append(self.mx_tf3_cbox1);

        self.mx_tf3_cbox2 = QCheckBox("Validation", self)
        self.mx_tf3_cbox2.setChecked(True)
        self.mx_tf3_cbox2.move(210, 300);
        tmp.append(self.mx_tf3_cbox2);

        self.mx_tf3_cbox3 = QCheckBox("Testing", self)
        self.mx_tf3_cbox3.setChecked(False)
        self.mx_tf3_cbox3.move(310, 300);
        tmp.append(self.mx_tf3_cbox3);

        self.transform_ui_mxnet.append(tmp)

        



        tmp = [];
        self.mx_tf4_l1 = QLabel(self);
        self.mx_tf4_l1.setText("1. Flip probability (0-1):");
        self.mx_tf4_l1.move(20, 100);
        tmp.append(self.mx_tf4_l1);

        self.mx_tf4_e1 = QLineEdit(self)
        self.mx_tf4_e1.move(180, 100);
        self.mx_tf4_e1.setText("0.5");
        tmp.append(self.mx_tf4_e1);



        self.mx_tf4_l2 = QLabel(self);
        self.mx_tf4_l2.setText("2. Apply at");
        self.mx_tf4_l2.move(20, 150);
        tmp.append(self.mx_tf4_l2);

        self.mx_tf4_cbox1 = QCheckBox("Training", self)
        self.mx_tf4_cbox1.setChecked(True)
        self.mx_tf4_cbox1.move(110, 150);
        tmp.append(self.mx_tf4_cbox1);

        self.mx_tf4_cbox2 = QCheckBox("Validation", self)
        self.mx_tf4_cbox2.setChecked(True)
        self.mx_tf4_cbox2.move(210, 150);
        tmp.append(self.mx_tf4_cbox2);

        self.mx_tf4_cbox3 = QCheckBox("Testing", self)
        self.mx_tf4_cbox3.setChecked(False)
        self.mx_tf4_cbox3.move(310, 150);
        tmp.append(self.mx_tf4_cbox3);

        self.transform_ui_mxnet.append(tmp)






        tmp = [];
        self.mx_tf5_l1 = QLabel(self);
        self.mx_tf5_l1.setText("1. Flip probability (0-1):");
        self.mx_tf5_l1.move(20, 100);
        tmp.append(self.mx_tf5_l1);

        self.mx_tf5_e1 = QLineEdit(self)
        self.mx_tf5_e1.move(180, 100);
        self.mx_tf5_e1.setText("0.5");
        tmp.append(self.mx_tf5_e1);



        self.mx_tf5_l2 = QLabel(self);
        self.mx_tf5_l2.setText("2. Apply at");
        self.mx_tf5_l2.move(20, 150);
        tmp.append(self.mx_tf5_l2);

        self.mx_tf5_cbox1 = QCheckBox("Training", self)
        self.mx_tf5_cbox1.setChecked(True)
        self.mx_tf5_cbox1.move(110, 150);
        tmp.append(self.mx_tf5_cbox1);

        self.mx_tf5_cbox2 = QCheckBox("Validation", self)
        self.mx_tf5_cbox2.setChecked(True)
        self.mx_tf5_cbox2.move(210, 150);
        tmp.append(self.mx_tf5_cbox2);

        self.mx_tf5_cbox3 = QCheckBox("Testing", self)
        self.mx_tf5_cbox3.setChecked(False)
        self.mx_tf5_cbox3.move(310, 150);
        tmp.append(self.mx_tf5_cbox3);

        self.transform_ui_mxnet.append(tmp);





        tmp = [];
        self.mx_tf6_l1 = QLabel(self);
        self.mx_tf6_l1.setText("1. Alpha:");
        self.mx_tf6_l1.move(20, 100);
        tmp.append(self.mx_tf6_l1);

        self.mx_tf6_e1 = QLineEdit(self)
        self.mx_tf6_e1.move(120, 100);
        self.mx_tf6_e1.setText("1.0");
        tmp.append(self.mx_tf6_e1);



        self.mx_tf6_l2 = QLabel(self);
        self.mx_tf6_l2.setText("2. Apply at");
        self.mx_tf6_l2.move(20, 150);
        tmp.append(self.mx_tf6_l2);

        self.mx_tf6_cbox1 = QCheckBox("Training", self)
        self.mx_tf6_cbox1.setChecked(True)
        self.mx_tf6_cbox1.move(110, 150);
        tmp.append(self.mx_tf6_cbox1);

        self.mx_tf6_cbox2 = QCheckBox("Validation", self)
        self.mx_tf6_cbox2.setChecked(True)
        self.mx_tf6_cbox2.move(210, 150);
        tmp.append(self.mx_tf6_cbox2);

        self.mx_tf6_cbox3 = QCheckBox("Testing", self)
        self.mx_tf6_cbox3.setChecked(False)
        self.mx_tf6_cbox3.move(310, 150);
        tmp.append(self.mx_tf6_cbox3);

        self.transform_ui_mxnet.append(tmp);



        tmp = [];
        self.mx_tf7_l1 = QLabel(self);
        self.mx_tf7_l1.setText("1. New size:");
        self.mx_tf7_l1.move(20, 100);
        tmp.append(self.mx_tf7_l1);

        self.mx_tf7_e1 = QLineEdit(self)
        self.mx_tf7_e1.move(120, 100);
        self.mx_tf7_e1.setText("224");
        tmp.append(self.mx_tf7_e1);



        self.mx_tf7_l2 = QLabel(self);
        self.mx_tf7_l2.setText("2. Apply at");
        self.mx_tf7_l2.move(20, 150);
        tmp.append(self.mx_tf7_l2);

        self.mx_tf7_cbox1 = QCheckBox("Training", self)
        self.mx_tf7_cbox1.setChecked(True)
        self.mx_tf7_cbox1.move(110, 150);
        tmp.append(self.mx_tf7_cbox1);

        self.mx_tf7_cbox2 = QCheckBox("Validation", self)
        self.mx_tf7_cbox2.setChecked(True)
        self.mx_tf7_cbox2.move(210, 150);
        tmp.append(self.mx_tf7_cbox2);

        self.mx_tf7_cbox3 = QCheckBox("Testing", self)
        self.mx_tf7_cbox3.setChecked(False)
        self.mx_tf7_cbox3.move(310, 150);
        tmp.append(self.mx_tf7_cbox3);

        self.transform_ui_mxnet.append(tmp);





        tmp = [];
        self.mx_tf8_l1 = QLabel(self);
        self.mx_tf8_l1.setText("1. Mean:");
        self.mx_tf8_l1.move(20, 100);
        tmp.append(self.mx_tf8_l1);

        self.mx_tf8_e1_1 = QLineEdit(self)
        self.mx_tf8_e1_1.move(120, 100);
        self.mx_tf8_e1_1.setText("0.485");
        self.mx_tf8_e1_1.resize(70, 25);
        tmp.append(self.mx_tf8_e1_1);

        self.mx_tf8_e1_2 = QLineEdit(self)
        self.mx_tf8_e1_2.move(220, 100);
        self.mx_tf8_e1_2.setText("0.456");
        self.mx_tf8_e1_2.resize(70, 25);
        tmp.append(self.mx_tf8_e1_2);

        self.mx_tf8_e1_3 = QLineEdit(self)
        self.mx_tf8_e1_3.move(320, 100);
        self.mx_tf8_e1_3.setText("0.406");
        self.mx_tf8_e1_3.resize(70, 25);
        tmp.append(self.mx_tf8_e1_3);




        self.mx_tf8_l2 = QLabel(self);
        self.mx_tf8_l2.setText("2. Standard deviation:");
        self.mx_tf8_l2.move(20, 150);
        tmp.append(self.mx_tf8_l2);

        self.mx_tf8_e2_1 = QLineEdit(self)
        self.mx_tf8_e2_1.move(180, 150);
        self.mx_tf8_e2_1.setText("0.229");
        self.mx_tf8_e2_1.resize(70, 25);
        tmp.append(self.mx_tf8_e2_1);

        self.mx_tf8_e2_2 = QLineEdit(self)
        self.mx_tf8_e2_2.move(280, 150);
        self.mx_tf8_e2_2.setText("0.224");
        self.mx_tf8_e2_2.resize(70, 25);
        tmp.append(self.mx_tf8_e2_2);

        self.mx_tf8_e2_3 = QLineEdit(self)
        self.mx_tf8_e2_3.move(380, 150);
        self.mx_tf8_e2_3.setText("0.225");
        self.mx_tf8_e2_3.resize(70, 25);
        tmp.append(self.mx_tf8_e2_3);




        self.mx_tf8_l3 = QLabel(self);
        self.mx_tf8_l3.setText("3. Apply at");
        self.mx_tf8_l3.move(20, 200);
        tmp.append(self.mx_tf8_l3);

        self.mx_tf8_cbox1 = QCheckBox("Training", self)
        self.mx_tf8_cbox1.setChecked(True)
        self.mx_tf8_cbox1.move(110, 200);
        tmp.append(self.mx_tf8_cbox1);

        self.mx_tf8_cbox2 = QCheckBox("Validation", self)
        self.mx_tf8_cbox2.setChecked(True)
        self.mx_tf8_cbox2.move(210, 200);
        tmp.append(self.mx_tf8_cbox2);

        self.mx_tf8_cbox3 = QCheckBox("Testing", self)
        self.mx_tf8_cbox3.setChecked(False)
        self.mx_tf8_cbox3.move(310, 200);
        tmp.append(self.mx_tf8_cbox3);

        self.transform_ui_mxnet.append(tmp);






        tmp = [];
        self.py_tf1_l1 = QLabel(self);
        self.py_tf1_l1.setText("1. Crop Size:");
        self.py_tf1_l1.move(20, 100);
        tmp.append(self.py_tf1_l1);

        self.py_tf1_e1 = QLineEdit(self)
        self.py_tf1_e1.move(150, 100);
        self.py_tf1_e1.setText("224");
        tmp.append(self.py_tf1_e1);


        self.py_tf1_l2 = QLabel(self);
        self.py_tf1_l2.setText("2. Apply at");
        self.py_tf1_l2.move(20, 150);
        tmp.append(self.py_tf1_l2);

        self.py_tf1_cbox1 = QCheckBox("Training", self)
        self.py_tf1_cbox1.setChecked(True)
        self.py_tf1_cbox1.move(110, 150);
        tmp.append(self.py_tf1_cbox1);

        self.py_tf1_cbox2 = QCheckBox("Validation", self)
        self.py_tf1_cbox2.setChecked(True)
        self.py_tf1_cbox2.move(210, 150);
        tmp.append(self.py_tf1_cbox2);

        self.py_tf1_cbox3 = QCheckBox("Testing", self)
        self.py_tf1_cbox3.setChecked(False)
        self.py_tf1_cbox3.move(310, 150);
        tmp.append(self.py_tf1_cbox3);

        self.transform_ui_pytorch.append(tmp)





        tmp = [];
        self.py_tf2_l1 = QLabel(self);
        self.py_tf2_l1.setText("1. Brightness (0-1):");
        self.py_tf2_l1.move(20, 100);
        tmp.append(self.py_tf2_l1);

        self.py_tf2_e1 = QLineEdit(self)
        self.py_tf2_e1.move(150, 100);
        self.py_tf2_e1.setText("0.0");
        tmp.append(self.py_tf2_e1);


        self.py_tf2_l2 = QLabel(self);
        self.py_tf2_l2.setText("2. Contrast (0-1):");
        self.py_tf2_l2.move(20, 150);
        tmp.append(self.py_tf2_l2);

        self.py_tf2_e2 = QLineEdit(self)
        self.py_tf2_e2.move(150, 150);
        self.py_tf2_e2.setText("0.0");
        tmp.append(self.py_tf2_e2);


        self.py_tf2_l3 = QLabel(self);
        self.py_tf2_l3.setText("3. Saturation (0-1):");
        self.py_tf2_l3.move(20, 200);
        tmp.append(self.py_tf2_l3);

        self.py_tf2_e3 = QLineEdit(self)
        self.py_tf2_e3.move(150, 200);
        self.py_tf2_e3.setText("0.0");
        tmp.append(self.py_tf2_e3);


        self.py_tf2_l4 = QLabel(self);
        self.py_tf2_l4.setText("4. Hue (0-1):");
        self.py_tf2_l4.move(20, 250);
        tmp.append(self.py_tf2_l4);

        self.py_tf2_e4 = QLineEdit(self)
        self.py_tf2_e4.move(150, 250);
        self.py_tf2_e4.setText("0.0");
        tmp.append(self.py_tf2_e4);


        self.py_tf2_l5 = QLabel(self);
        self.py_tf2_l5.setText("5. Apply at");
        self.py_tf2_l5.move(20, 300);
        tmp.append(self.py_tf2_l5);

        self.py_tf2_cbox1 = QCheckBox("Training", self)
        self.py_tf2_cbox1.setChecked(True)
        self.py_tf2_cbox1.move(110, 300);
        tmp.append(self.py_tf2_cbox1);

        self.py_tf2_cbox2 = QCheckBox("Validation", self)
        self.py_tf2_cbox2.setChecked(True)
        self.py_tf2_cbox2.move(210, 300);
        tmp.append(self.py_tf2_cbox2);

        self.py_tf2_cbox3 = QCheckBox("Testing", self)
        self.py_tf2_cbox3.setChecked(False)
        self.py_tf2_cbox3.move(310, 300);
        tmp.append(self.py_tf2_cbox3);

        self.transform_ui_pytorch.append(tmp)





        tmp = [];
        self.py_tf3_l1 = QLabel(self);
        self.py_tf3_l1.setText("1. Rotation (angle):");
        self.py_tf3_l1.move(20, 100);
        tmp.append(self.py_tf3_l1);

        self.py_tf3_e1 = QLineEdit(self)
        self.py_tf3_e1.move(150, 100);
        self.py_tf3_e1.setText("0.0");
        self.py_tf3_e1.resize(50, 25);
        tmp.append(self.py_tf3_e1);


        self.py_tf3_l2 = QLabel(self);
        self.py_tf3_l2.setText("2. Translation (ratio):");
        self.py_tf3_l2.move(20, 150);
        tmp.append(self.py_tf3_l2);

        self.py_tf3_e2_1 = QLineEdit(self)
        self.py_tf3_e2_1.move(200, 150);
        self.py_tf3_e2_1.setText("None");
        self.py_tf3_e2_1.resize(50, 25);
        tmp.append(self.py_tf3_e2_1);

        self.py_tf3_e2_2 = QLineEdit(self)
        self.py_tf3_e2_2.move(300, 150);
        self.py_tf3_e2_2.setText("None");
        self.py_tf3_e2_2.resize(50, 25);
        tmp.append(self.py_tf3_e2_2);


        self.py_tf3_l3 = QLabel(self);
        self.py_tf3_l3.setText("3. Scale (ratio):");
        self.py_tf3_l3.move(20, 200);
        tmp.append(self.py_tf3_l3);

        self.py_tf3_e3_1 = QLineEdit(self)
        self.py_tf3_e3_1.move(200, 200);
        self.py_tf3_e3_1.setText("None");
        self.py_tf3_e3_1.resize(50, 25);
        tmp.append(self.py_tf3_e3_1);

        self.py_tf3_e3_2 = QLineEdit(self)
        self.py_tf3_e3_2.move(300, 200);
        self.py_tf3_e3_2.setText("None");
        self.py_tf3_e3_2.resize(50, 25);
        tmp.append(self.py_tf3_e3_2);


        self.py_tf3_l4 = QLabel(self);
        self.py_tf3_l4.setText("4. Sheer (ratio):");
        self.py_tf3_l4.move(20, 250);
        tmp.append(self.py_tf3_l4);

        self.py_tf3_e4_1 = QLineEdit(self)
        self.py_tf3_e4_1.move(200, 250);
        self.py_tf3_e4_1.setText("None");
        self.py_tf3_e4_1.resize(50, 25);
        tmp.append(self.py_tf3_e4_1);

        self.py_tf3_e4_2 = QLineEdit(self)
        self.py_tf3_e4_2.move(300, 250);
        self.py_tf3_e4_2.setText("None");
        self.py_tf3_e4_2.resize(50, 25);
        tmp.append(self.py_tf3_e4_2);


        self.py_tf3_l5 = QLabel(self);
        self.py_tf3_l5.setText("5. Apply at");
        self.py_tf3_l5.move(20, 300);
        tmp.append(self.py_tf3_l5);

        self.py_tf3_cbox1 = QCheckBox("Training", self)
        self.py_tf3_cbox1.setChecked(True)
        self.py_tf3_cbox1.move(110, 300);
        tmp.append(self.py_tf3_cbox1);

        self.py_tf3_cbox2 = QCheckBox("Validation", self)
        self.py_tf3_cbox2.setChecked(True)
        self.py_tf3_cbox2.move(210, 300);
        tmp.append(self.py_tf3_cbox2);

        self.py_tf3_cbox3 = QCheckBox("Testing", self)
        self.py_tf3_cbox3.setChecked(False)
        self.py_tf3_cbox3.move(310, 300);
        tmp.append(self.py_tf3_cbox3);

        self.transform_ui_pytorch.append(tmp)




        tmp = [];
        self.py_tf4_l1 = QLabel(self);
        self.py_tf4_l1.setText("1. Crop Size:");
        self.py_tf4_l1.move(20, 100);
        tmp.append(self.py_tf4_l1);

        self.py_tf4_e1 = QLineEdit(self)
        self.py_tf4_e1.move(150, 100);
        self.py_tf4_e1.setText("224");
        tmp.append(self.py_tf4_e1);


        self.py_tf4_l2 = QLabel(self);
        self.py_tf4_l2.setText("2. Apply at");
        self.py_tf4_l2.move(20, 150);
        tmp.append(self.py_tf4_l2);

        self.py_tf4_cbox1 = QCheckBox("Training", self)
        self.py_tf4_cbox1.setChecked(True)
        self.py_tf4_cbox1.move(110, 150);
        tmp.append(self.py_tf4_cbox1);

        self.py_tf4_cbox2 = QCheckBox("Validation", self)
        self.py_tf4_cbox2.setChecked(True)
        self.py_tf4_cbox2.move(210, 150);
        tmp.append(self.py_tf4_cbox2);

        self.py_tf4_cbox3 = QCheckBox("Testing", self)
        self.py_tf4_cbox3.setChecked(False)
        self.py_tf4_cbox3.move(310, 150);
        tmp.append(self.py_tf4_cbox3);

        self.transform_ui_pytorch.append(tmp)




        tmp = [];
        self.py_tf5_l1 = QLabel(self);
        self.py_tf5_l1.setText("1. Probability (0-1):");
        self.py_tf5_l1.move(20, 100);
        tmp.append(self.py_tf5_l1);

        self.py_tf5_e1 = QLineEdit(self)
        self.py_tf5_e1.move(200, 100);
        self.py_tf5_e1.setText("0.5");
        tmp.append(self.py_tf5_e1);


        self.py_tf5_l2 = QLabel(self);
        self.py_tf5_l2.setText("2. Apply at");
        self.py_tf5_l2.move(20, 150);
        tmp.append(self.py_tf5_l2);

        self.py_tf5_cbox1 = QCheckBox("Training", self)
        self.py_tf5_cbox1.setChecked(True)
        self.py_tf5_cbox1.move(110, 150);
        tmp.append(self.py_tf5_cbox1);

        self.py_tf5_cbox2 = QCheckBox("Validation", self)
        self.py_tf5_cbox2.setChecked(True)
        self.py_tf5_cbox2.move(210, 150);
        tmp.append(self.py_tf5_cbox2);

        self.py_tf5_cbox3 = QCheckBox("Testing", self)
        self.py_tf5_cbox3.setChecked(False)
        self.py_tf5_cbox3.move(310, 150);
        tmp.append(self.py_tf5_cbox3);

        self.transform_ui_pytorch.append(tmp)





        tmp = [];
        self.py_tf6_l1 = QLabel(self);
        self.py_tf6_l1.setText("1. Distrotion (0-1):");
        self.py_tf6_l1.move(20, 100);
        tmp.append(self.py_tf6_l1);

        self.py_tf6_e1 = QLineEdit(self)
        self.py_tf6_e1.move(200, 100);
        self.py_tf6_e1.setText("0.5");
        tmp.append(self.py_tf6_e1);


        self.py_tf6_l2 = QLabel(self);
        self.py_tf6_l2.setText("2. Probability (0-1):");
        self.py_tf6_l2.move(20, 150);
        tmp.append(self.py_tf6_l2);

        self.py_tf6_e2 = QLineEdit(self)
        self.py_tf6_e2.move(200, 150);
        self.py_tf6_e2.setText("0.5");
        tmp.append(self.py_tf6_e2);


        self.py_tf6_l3 = QLabel(self);
        self.py_tf6_l3.setText("3. Apply at");
        self.py_tf6_l3.move(20, 200);
        tmp.append(self.py_tf6_l3);

        self.py_tf6_cbox1 = QCheckBox("Training", self)
        self.py_tf6_cbox1.setChecked(True)
        self.py_tf6_cbox1.move(110, 200);
        tmp.append(self.py_tf6_cbox1);

        self.py_tf6_cbox2 = QCheckBox("Validation", self)
        self.py_tf6_cbox2.setChecked(True)
        self.py_tf6_cbox2.move(210, 200);
        tmp.append(self.py_tf6_cbox2);

        self.py_tf6_cbox3 = QCheckBox("Testing", self)
        self.py_tf6_cbox3.setChecked(False)
        self.py_tf6_cbox3.move(310, 200);
        tmp.append(self.py_tf6_cbox3);

        self.transform_ui_pytorch.append(tmp)





        tmp = [];
        self.py_tf7_l1 = QLabel(self);
        self.py_tf7_l1.setText("1. Crop Size:");
        self.py_tf7_l1.move(20, 100);
        tmp.append(self.py_tf7_l1);

        self.py_tf7_e1 = QLineEdit(self)
        self.py_tf7_e1.move(150, 100);
        self.py_tf7_e1.setText("224");
        tmp.append(self.py_tf7_e1);


        self.py_tf7_l2 = QLabel(self);
        self.py_tf7_l2.setText("2. Scale:");
        self.py_tf7_l2.move(20, 150);
        tmp.append(self.py_tf7_l2);

        self.py_tf7_e2_1 = QLineEdit(self)
        self.py_tf7_e2_1.move(120, 150);
        self.py_tf7_e2_1.setText("0.08");
        self.py_tf7_e2_1.resize(50, 25);
        tmp.append(self.py_tf7_e2_1);

        self.py_tf7_e2_2 = QLineEdit(self)
        self.py_tf7_e2_2.move(220, 150);
        self.py_tf7_e2_2.setText("1.0");
        self.py_tf7_e2_2.resize(50, 25);
        tmp.append(self.py_tf7_e2_2);


        self.py_tf7_l3 = QLabel(self);
        self.py_tf7_l3.setText("3. Ratio:");
        self.py_tf7_l3.move(20, 200);
        tmp.append(self.py_tf7_l3);

        self.py_tf7_e3_1 = QLineEdit(self)
        self.py_tf7_e3_1.move(120, 200);
        self.py_tf7_e3_1.setText("0.75");
        self.py_tf7_e3_1.resize(50, 25);
        tmp.append(self.py_tf7_e3_1);

        self.py_tf7_e3_2 = QLineEdit(self)
        self.py_tf7_e3_2.move(220, 200);
        self.py_tf7_e3_2.setText("1.33");
        self.py_tf7_e3_2.resize(50, 25);
        tmp.append(self.py_tf7_e3_2);



        self.py_tf7_l4 = QLabel(self);
        self.py_tf7_l4.setText("4. Apply at");
        self.py_tf7_l4.move(20, 300);
        tmp.append(self.py_tf7_l4);

        self.py_tf7_cbox1 = QCheckBox("Training", self)
        self.py_tf7_cbox1.setChecked(True)
        self.py_tf7_cbox1.move(110, 300);
        tmp.append(self.py_tf7_cbox1);

        self.py_tf7_cbox2 = QCheckBox("Validation", self)
        self.py_tf7_cbox2.setChecked(True)
        self.py_tf7_cbox2.move(210, 300);
        tmp.append(self.py_tf7_cbox2);

        self.py_tf7_cbox3 = QCheckBox("Testing", self)
        self.py_tf7_cbox3.setChecked(False)
        self.py_tf7_cbox3.move(310, 300);
        tmp.append(self.py_tf7_cbox3);

        self.transform_ui_pytorch.append(tmp)







        tmp = [];
        self.py_tf8_l1 = QLabel(self);
        self.py_tf8_l1.setText("1. Degrees:");
        self.py_tf8_l1.move(20, 100);
        tmp.append(self.py_tf8_l1);

        self.py_tf8_e1 = QLineEdit(self)
        self.py_tf8_e1.move(200, 100);
        self.py_tf8_e1.setText("10");
        tmp.append(self.py_tf8_e1);


        self.py_tf8_l2 = QLabel(self);
        self.py_tf8_l2.setText("2. Apply at");
        self.py_tf8_l2.move(20, 200);
        tmp.append(self.py_tf8_l2);

        self.py_tf8_cbox1 = QCheckBox("Training", self)
        self.py_tf8_cbox1.setChecked(True)
        self.py_tf8_cbox1.move(110, 200);
        tmp.append(self.py_tf8_cbox1);

        self.py_tf8_cbox2 = QCheckBox("Validation", self)
        self.py_tf8_cbox2.setChecked(True)
        self.py_tf8_cbox2.move(210, 200);
        tmp.append(self.py_tf8_cbox2);

        self.py_tf8_cbox3 = QCheckBox("Testing", self)
        self.py_tf8_cbox3.setChecked(False)
        self.py_tf8_cbox3.move(310, 200);
        tmp.append(self.py_tf8_cbox3);

        self.transform_ui_pytorch.append(tmp)






        tmp = [];
        self.py_tf9_l1 = QLabel(self);
        self.py_tf9_l1.setText("1. Probability (0-1):");
        self.py_tf9_l1.move(20, 100);
        tmp.append(self.py_tf9_l1);

        self.py_tf9_e1 = QLineEdit(self)
        self.py_tf9_e1.move(200, 100);
        self.py_tf9_e1.setText("0.5");
        tmp.append(self.py_tf9_e1);


        self.py_tf9_l2 = QLabel(self);
        self.py_tf9_l2.setText("2. Apply at");
        self.py_tf9_l2.move(20, 200);
        tmp.append(self.py_tf9_l2);

        self.py_tf9_cbox1 = QCheckBox("Training", self)
        self.py_tf9_cbox1.setChecked(True)
        self.py_tf9_cbox1.move(110, 200);
        tmp.append(self.py_tf9_cbox1);

        self.py_tf9_cbox2 = QCheckBox("Validation", self)
        self.py_tf9_cbox2.setChecked(True)
        self.py_tf9_cbox2.move(210, 200);
        tmp.append(self.py_tf9_cbox2);

        self.py_tf9_cbox3 = QCheckBox("Testing", self)
        self.py_tf9_cbox3.setChecked(False)
        self.py_tf9_cbox3.move(310, 200);
        tmp.append(self.py_tf9_cbox3);

        self.transform_ui_pytorch.append(tmp)





        tmp = [];
        self.py_tf10_l1 = QLabel(self);
        self.py_tf10_l1.setText("1. New size:");
        self.py_tf10_l1.move(20, 100);
        tmp.append(self.py_tf10_l1);

        self.py_tf10_e1 = QLineEdit(self)
        self.py_tf10_e1.move(200, 100);
        self.py_tf10_e1.setText("224");
        tmp.append(self.py_tf10_e1);


        self.py_tf10_l2 = QLabel(self);
        self.py_tf10_l2.setText("2. Apply at");
        self.py_tf10_l2.move(20, 200);
        tmp.append(self.py_tf10_l2);

        self.py_tf10_cbox1 = QCheckBox("Training", self)
        self.py_tf10_cbox1.setChecked(True)
        self.py_tf10_cbox1.move(110, 200);
        tmp.append(self.py_tf10_cbox1);

        self.py_tf10_cbox2 = QCheckBox("Validation", self)
        self.py_tf10_cbox2.setChecked(True)
        self.py_tf10_cbox2.move(210, 200);
        tmp.append(self.py_tf10_cbox2);

        self.py_tf10_cbox3 = QCheckBox("Testing", self)
        self.py_tf10_cbox3.setChecked(False)
        self.py_tf10_cbox3.move(310, 200);
        tmp.append(self.py_tf10_cbox3);

        self.transform_ui_pytorch.append(tmp)





        tmp = [];
        self.py_tf11_l1 = QLabel(self);
        self.py_tf11_l1.setText("1. Mean:");
        self.py_tf11_l1.move(20, 100);
        tmp.append(self.py_tf11_l1);

        self.py_tf11_e1_1 = QLineEdit(self)
        self.py_tf11_e1_1.move(120, 100);
        self.py_tf11_e1_1.setText("0.485");
        self.py_tf11_e1_1.resize(70, 25);
        tmp.append(self.py_tf11_e1_1);

        self.py_tf11_e1_2 = QLineEdit(self)
        self.py_tf11_e1_2.move(220, 100);
        self.py_tf11_e1_2.setText("0.456");
        self.py_tf11_e1_2.resize(70, 25);
        tmp.append(self.py_tf11_e1_2);

        self.py_tf11_e1_3 = QLineEdit(self)
        self.py_tf11_e1_3.move(320, 100);
        self.py_tf11_e1_3.setText("0.406");
        self.py_tf11_e1_3.resize(70, 25);
        tmp.append(self.py_tf11_e1_3);




        self.py_tf11_l2 = QLabel(self);
        self.py_tf11_l2.setText("2. Standard deviation:");
        self.py_tf11_l2.move(20, 150);
        tmp.append(self.py_tf11_l2);

        self.py_tf11_e2_1 = QLineEdit(self)
        self.py_tf11_e2_1.move(180, 150);
        self.py_tf11_e2_1.setText("0.229");
        self.py_tf11_e2_1.resize(70, 25);
        tmp.append(self.py_tf11_e2_1);

        self.py_tf11_e2_2 = QLineEdit(self)
        self.py_tf11_e2_2.move(280, 150);
        self.py_tf11_e2_2.setText("0.224");
        self.py_tf11_e2_2.resize(70, 25);
        tmp.append(self.py_tf11_e2_2);

        self.py_tf11_e2_3 = QLineEdit(self)
        self.py_tf11_e2_3.move(380, 150);
        self.py_tf11_e2_3.setText("0.225");
        self.py_tf11_e2_3.resize(70, 25);
        tmp.append(self.py_tf11_e2_3);




        self.py_tf11_l3 = QLabel(self);
        self.py_tf11_l3.setText("3. Apply at");
        self.py_tf11_l3.move(20, 200);
        tmp.append(self.py_tf11_l3);

        self.py_tf11_cbox1 = QCheckBox("Training", self)
        self.py_tf11_cbox1.setChecked(True)
        self.py_tf11_cbox1.move(110, 200);
        tmp.append(self.py_tf11_cbox1);

        self.py_tf11_cbox2 = QCheckBox("Validation", self)
        self.py_tf11_cbox2.setChecked(True)
        self.py_tf11_cbox2.move(210, 200);
        tmp.append(self.py_tf11_cbox2);

        self.py_tf11_cbox3 = QCheckBox("Testing", self)
        self.py_tf11_cbox3.setChecked(False)
        self.py_tf11_cbox3.move(310, 200);
        tmp.append(self.py_tf11_cbox3);

        self.transform_ui_pytorch.append(tmp);





        tmp = [];
        self.ke_tf1_l1 = QLabel(self);
        self.ke_tf1_l1.setText("1. Brightness (0-1):");
        self.ke_tf1_l1.move(20, 100);
        tmp.append(self.ke_tf1_l1);

        self.ke_tf1_e1 = QLineEdit(self)
        self.ke_tf1_e1.move(150, 100);
        self.ke_tf1_e1.setText("0.0");
        tmp.append(self.ke_tf1_e1);


        self.ke_tf1_l2 = QLabel(self);
        self.ke_tf1_l2.setText("2. Contrast (0-1):");
        self.ke_tf1_l2.move(20, 150);
        tmp.append(self.ke_tf1_l2);

        self.ke_tf1_e2 = QLineEdit(self)
        self.ke_tf1_e2.move(150, 150);
        self.ke_tf1_e2.setText("0.0");
        tmp.append(self.ke_tf1_e2);


        self.ke_tf1_l3 = QLabel(self);
        self.ke_tf1_l3.setText("3. Saturation (0-1):");
        self.ke_tf1_l3.move(20, 200);
        tmp.append(self.ke_tf1_l3);

        self.ke_tf1_e3 = QLineEdit(self)
        self.ke_tf1_e3.move(150, 200);
        self.ke_tf1_e3.setText("0.0");
        tmp.append(self.ke_tf1_e3);


        self.ke_tf1_l4 = QLabel(self);
        self.ke_tf1_l4.setText("4. Hue (0-1):");
        self.ke_tf1_l4.move(20, 250);
        tmp.append(self.ke_tf1_l4);

        self.ke_tf1_e4 = QLineEdit(self)
        self.ke_tf1_e4.move(150, 250);
        self.ke_tf1_e4.setText("0.0");
        tmp.append(self.ke_tf1_e4);


        self.ke_tf1_l5 = QLabel(self);
        self.ke_tf1_l5.setText("5. Apply at");
        self.ke_tf1_l5.move(20, 300);
        tmp.append(self.ke_tf1_l5);

        self.ke_tf1_cbox1 = QCheckBox("Training", self)
        self.ke_tf1_cbox1.setChecked(True)
        self.ke_tf1_cbox1.move(110, 300);
        tmp.append(self.ke_tf1_cbox1);

        self.ke_tf1_cbox2 = QCheckBox("Validation", self)
        self.ke_tf1_cbox2.setChecked(True)
        self.ke_tf1_cbox2.move(210, 300);
        tmp.append(self.ke_tf1_cbox2);

        self.ke_tf1_cbox3 = QCheckBox("Testing", self)
        self.ke_tf1_cbox3.setChecked(False)
        self.ke_tf1_cbox3.move(310, 300);
        tmp.append(self.ke_tf1_cbox3);

        self.transform_ui_keras.append(tmp)






        tmp = [];
        self.ke_tf2_l1 = QLabel(self);
        self.ke_tf2_l1.setText("1. Crop Size:");
        self.ke_tf2_l1.move(20, 100);
        tmp.append(self.ke_tf2_l1);

        self.ke_tf2_e1 = QLineEdit(self)
        self.ke_tf2_e1.move(150, 100);
        self.ke_tf2_e1.setText("224");
        tmp.append(self.ke_tf2_e1);


        self.ke_tf2_l2 = QLabel(self);
        self.ke_tf2_l2.setText("2. Scale:");
        self.ke_tf2_l2.move(20, 150);
        tmp.append(self.ke_tf2_l2);

        self.ke_tf2_e2_1 = QLineEdit(self)
        self.ke_tf2_e2_1.move(120, 150);
        self.ke_tf2_e2_1.setText("0.08");
        self.ke_tf2_e2_1.resize(50, 25);
        tmp.append(self.ke_tf2_e2_1);

        self.ke_tf2_e2_2 = QLineEdit(self)
        self.ke_tf2_e2_2.move(220, 150);
        self.ke_tf2_e2_2.setText("1.0");
        self.ke_tf2_e2_2.resize(50, 25);
        tmp.append(self.ke_tf2_e2_2);


        self.ke_tf2_l3 = QLabel(self);
        self.ke_tf2_l3.setText("3. Ratio:");
        self.ke_tf2_l3.move(20, 200);
        tmp.append(self.ke_tf2_l3);

        self.ke_tf2_e3_1 = QLineEdit(self)
        self.ke_tf2_e3_1.move(120, 200);
        self.ke_tf2_e3_1.setText("0.75");
        self.ke_tf2_e3_1.resize(50, 25);
        tmp.append(self.ke_tf2_e3_1);

        self.ke_tf2_e3_2 = QLineEdit(self)
        self.ke_tf2_e3_2.move(220, 200);
        self.ke_tf2_e3_2.setText("1.33");
        self.ke_tf2_e3_2.resize(50, 25);
        tmp.append(self.ke_tf2_e3_2);



        self.ke_tf2_l4 = QLabel(self);
        self.ke_tf2_l4.setText("4. Apply at");
        self.ke_tf2_l4.move(20, 300);
        tmp.append(self.ke_tf2_l4);

        self.ke_tf2_cbox1 = QCheckBox("Training", self)
        self.ke_tf2_cbox1.setChecked(True)
        self.ke_tf2_cbox1.move(110, 300);
        tmp.append(self.ke_tf2_cbox1);

        self.ke_tf2_cbox2 = QCheckBox("Validation", self)
        self.ke_tf2_cbox2.setChecked(True)
        self.ke_tf2_cbox2.move(210, 300);
        tmp.append(self.ke_tf2_cbox2);

        self.ke_tf2_cbox3 = QCheckBox("Testing", self)
        self.ke_tf2_cbox3.setChecked(False)
        self.ke_tf2_cbox3.move(310, 300);
        tmp.append(self.ke_tf2_cbox3);

        self.transform_ui_keras.append(tmp)







        tmp = [];
        self.ke_tf3_l1 = QLabel(self);
        self.ke_tf3_l1.setText("1. Probability (0-1):");
        self.ke_tf3_l1.move(20, 100);
        tmp.append(self.ke_tf3_l1);

        self.ke_tf3_e1 = QLineEdit(self)
        self.ke_tf3_e1.move(150, 100);
        self.ke_tf3_e1.setText("0.5");
        tmp.append(self.ke_tf3_e1);



        self.ke_tf3_l2 = QLabel(self);
        self.ke_tf3_l2.setText("2. Apply at");
        self.ke_tf3_l2.move(20, 150);
        tmp.append(self.ke_tf3_l2);

        self.ke_tf3_cbox1 = QCheckBox("Training", self)
        self.ke_tf3_cbox1.setChecked(True)
        self.ke_tf3_cbox1.move(110, 150);
        tmp.append(self.ke_tf3_cbox1);

        self.ke_tf3_cbox2 = QCheckBox("Validation", self)
        self.ke_tf3_cbox2.setChecked(True)
        self.ke_tf3_cbox2.move(210, 150);
        tmp.append(self.ke_tf3_cbox2);

        self.ke_tf3_cbox3 = QCheckBox("Testing", self)
        self.ke_tf3_cbox3.setChecked(False)
        self.ke_tf3_cbox3.move(310, 150);
        tmp.append(self.ke_tf3_cbox3);

        self.transform_ui_keras.append(tmp)






        tmp = [];
        self.ke_tf4_l1 = QLabel(self);
        self.ke_tf4_l1.setText("1. Probability (0-1):");
        self.ke_tf4_l1.move(20, 100);
        tmp.append(self.ke_tf4_l1);

        self.ke_tf4_e1 = QLineEdit(self)
        self.ke_tf4_e1.move(150, 100);
        self.ke_tf4_e1.setText("0.5");
        tmp.append(self.ke_tf4_e1);



        self.ke_tf4_l2 = QLabel(self);
        self.ke_tf4_l2.setText("2. Apply at");
        self.ke_tf4_l2.move(20, 150);
        tmp.append(self.ke_tf4_l2);

        self.ke_tf4_cbox1 = QCheckBox("Training", self)
        self.ke_tf4_cbox1.setChecked(True)
        self.ke_tf4_cbox1.move(110, 150);
        tmp.append(self.ke_tf4_cbox1);

        self.ke_tf4_cbox2 = QCheckBox("Validation", self)
        self.ke_tf4_cbox2.setChecked(True)
        self.ke_tf4_cbox2.move(210, 150);
        tmp.append(self.ke_tf4_cbox2);

        self.ke_tf4_cbox3 = QCheckBox("Testing", self)
        self.ke_tf4_cbox3.setChecked(False)
        self.ke_tf4_cbox3.move(310, 150);
        tmp.append(self.ke_tf4_cbox3);

        self.transform_ui_keras.append(tmp)





        tmp = [];
        self.ke_tf5_l1 = QLabel(self);
        self.ke_tf5_l1.setText("1. Degrees:");
        self.ke_tf5_l1.move(20, 100);
        tmp.append(self.ke_tf5_l1);

        self.ke_tf5_e1 = QLineEdit(self)
        self.ke_tf5_e1.move(150, 100);
        self.ke_tf5_e1.setText("0.5");
        tmp.append(self.ke_tf5_e1);



        self.ke_tf5_l2 = QLabel(self);
        self.ke_tf5_l2.setText("2. Apply at");
        self.ke_tf5_l2.move(20, 150);
        tmp.append(self.ke_tf5_l2);

        self.ke_tf5_cbox1 = QCheckBox("Training", self)
        self.ke_tf5_cbox1.setChecked(True)
        self.ke_tf5_cbox1.move(110, 150);
        tmp.append(self.ke_tf5_cbox1);

        self.ke_tf5_cbox2 = QCheckBox("Validation", self)
        self.ke_tf5_cbox2.setChecked(True)
        self.ke_tf5_cbox2.move(210, 150);
        tmp.append(self.ke_tf5_cbox2);

        self.ke_tf5_cbox3 = QCheckBox("Testing", self)
        self.ke_tf5_cbox3.setChecked(False)
        self.ke_tf5_cbox3.move(310, 150);
        tmp.append(self.ke_tf5_cbox3);

        self.transform_ui_keras.append(tmp)





        tmp = [];
        self.ke_tf6_l1 = QLabel(self);
        self.ke_tf6_l1.setText("1. Mean:");
        self.ke_tf6_l1.move(20, 100);
        tmp.append(self.ke_tf6_l1);

        self.ke_tf6_e1_1 = QLineEdit(self)
        self.ke_tf6_e1_1.move(120, 100);
        self.ke_tf6_e1_1.setText("0.485");
        self.ke_tf6_e1_1.resize(70, 25);
        tmp.append(self.ke_tf6_e1_1);

        self.ke_tf6_e1_2 = QLineEdit(self)
        self.ke_tf6_e1_2.move(220, 100);
        self.ke_tf6_e1_2.setText("0.456");
        self.ke_tf6_e1_2.resize(70, 25);
        tmp.append(self.ke_tf6_e1_2);

        self.ke_tf6_e1_3 = QLineEdit(self)
        self.ke_tf6_e1_3.move(320, 100);
        self.ke_tf6_e1_3.setText("0.406");
        self.ke_tf6_e1_3.resize(70, 25);
        tmp.append(self.ke_tf6_e1_3);


        self.ke_tf6_l2 = QLabel(self);
        self.ke_tf6_l2.setText("2. Apply at");
        self.ke_tf6_l2.move(20, 150);
        tmp.append(self.ke_tf6_l2);

        self.ke_tf6_cbox1 = QCheckBox("Training", self)
        self.ke_tf6_cbox1.setChecked(True)
        self.ke_tf6_cbox1.move(110, 150);
        tmp.append(self.ke_tf6_cbox1);

        self.ke_tf6_cbox2 = QCheckBox("Validation", self)
        self.ke_tf6_cbox2.setChecked(True)
        self.ke_tf6_cbox2.move(210, 150);
        tmp.append(self.ke_tf6_cbox2);

        self.ke_tf6_cbox3 = QCheckBox("Testing", self)
        self.ke_tf6_cbox3.setChecked(False)
        self.ke_tf6_cbox3.move(310, 150);
        tmp.append(self.ke_tf6_cbox3);

        self.transform_ui_keras.append(tmp);




        tmp = [];
        self.ke_tf7_l1 = QLabel(self);
        self.ke_tf7_l1.setText("1. Mean:");
        self.ke_tf7_l1.move(20, 100);
        tmp.append(self.ke_tf7_l1);

        self.ke_tf7_e1_1 = QLineEdit(self)
        self.ke_tf7_e1_1.move(120, 100);
        self.ke_tf7_e1_1.setText("0.485");
        self.ke_tf7_e1_1.resize(70, 25);
        tmp.append(self.ke_tf7_e1_1);

        self.ke_tf7_e1_2 = QLineEdit(self)
        self.ke_tf7_e1_2.move(220, 100);
        self.ke_tf7_e1_2.setText("0.456");
        self.ke_tf7_e1_2.resize(70, 25);
        tmp.append(self.ke_tf7_e1_2);

        self.ke_tf7_e1_3 = QLineEdit(self)
        self.ke_tf7_e1_3.move(320, 100);
        self.ke_tf7_e1_3.setText("0.406");
        self.ke_tf7_e1_3.resize(70, 25);
        tmp.append(self.ke_tf7_e1_3);


        self.ke_tf7_l2 = QLabel(self);
        self.ke_tf7_l2.setText("2. Standard deviation:");
        self.ke_tf7_l2.move(20, 150);
        tmp.append(self.ke_tf7_l2);

        self.ke_tf7_e2_1 = QLineEdit(self)
        self.ke_tf7_e2_1.move(180, 150);
        self.ke_tf7_e2_1.setText("0.229");
        self.ke_tf7_e2_1.resize(70, 25);
        tmp.append(self.ke_tf7_e2_1);

        self.ke_tf7_e2_2 = QLineEdit(self)
        self.ke_tf7_e2_2.move(280, 150);
        self.ke_tf7_e2_2.setText("0.224");
        self.ke_tf7_e2_2.resize(70, 25);
        tmp.append(self.ke_tf7_e2_2);

        self.ke_tf7_e2_3 = QLineEdit(self)
        self.ke_tf7_e2_3.move(380, 150);
        self.ke_tf7_e2_3.setText("0.225");
        self.ke_tf7_e2_3.resize(70, 25);
        tmp.append(self.ke_tf7_e2_3);


        self.ke_tf7_l2 = QLabel(self);
        self.ke_tf7_l2.setText("3. Apply at");
        self.ke_tf7_l2.move(20, 200);
        tmp.append(self.ke_tf7_l2);

        self.ke_tf7_cbox1 = QCheckBox("Training", self)
        self.ke_tf7_cbox1.setChecked(True)
        self.ke_tf7_cbox1.move(110, 200);
        tmp.append(self.ke_tf7_cbox1);

        self.ke_tf7_cbox2 = QCheckBox("Validation", self)
        self.ke_tf7_cbox2.setChecked(True)
        self.ke_tf7_cbox2.move(210, 200);
        tmp.append(self.ke_tf7_cbox2);

        self.ke_tf7_cbox3 = QCheckBox("Testing", self)
        self.ke_tf7_cbox3.setChecked(False)
        self.ke_tf7_cbox3.move(310, 200);
        tmp.append(self.ke_tf7_cbox3);

        self.transform_ui_keras.append(tmp);





















        self.select_transform();

        self.tb1 = QTextEdit(self)
        self.tb1.move(550, 20)
        self.tb1.resize(300, 500)
        wr = "";
        for i in range(len(self.system["update"]["transforms"]["value"])):
            tmp = json.dumps(self.system["update"]["transforms"]["value"][i], indent=4)
            wr += "{}\n".format(tmp);
        self.tb1.setText(wr);


        self.b4 = QPushButton('Add Transform', self)
        self.b4.move(400,400)
        self.b4.clicked.connect(self.add_transform)

        
        self.b5 = QPushButton('Remove last transform', self)
        self.b5.move(370,450)
        self.b5.clicked.connect(self.remove_transform)

        
        self.b6 = QPushButton('Clear transforms', self)
        self.b6.move(400,500)
        self.b6.clicked.connect(self.clear_transform)

        



    def select_transform(self):
        self.current_transform = {};
        self.current_transform["name"] = "";
        self.current_transform["params"] = {};


        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.current_transform["name"] = self.cb1.currentText();
            index = self.mxnet_transforms_list.index(self.cb1.currentText());
            for i in range(len(self.transform_ui_mxnet)):
                for j in range(len(self.transform_ui_mxnet[i])):
                    if((index-1)==i):
                        self.transform_ui_mxnet[i][j].show();
                    else:
                        self.transform_ui_mxnet[i][j].hide();

            for i in range(len(self.transform_ui_keras)):
                for j in range(len(self.transform_ui_keras[i])):
                    self.transform_ui_keras[i][j].hide();
            for i in range(len(self.transform_ui_pytorch)):
                for j in range(len(self.transform_ui_pytorch[i])):
                    self.transform_ui_pytorch[i][j].hide();
            


        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.current_transform["name"] = self.cb2.currentText();
            index = self.keras_transforms_list.index(self.cb2.currentText());
            for i in range(len(self.transform_ui_keras)):
                for j in range(len(self.transform_ui_keras[i])):
                    if((index-1)==i):
                        self.transform_ui_keras[i][j].show();
                    else:
                        self.transform_ui_keras[i][j].hide();

            for i in range(len(self.transform_ui_mxnet)):
                for j in range(len(self.transform_ui_mxnet[i])):
                    self.transform_ui_mxnet[i][j].hide();
            for i in range(len(self.transform_ui_pytorch)):
                for j in range(len(self.transform_ui_pytorch[i])):
                    self.transform_ui_pytorch[i][j].hide();



        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.current_transform["name"] = self.cb3.currentText();
            index = self.pytorch_transforms_list.index(self.cb3.currentText());
            for i in range(len(self.transform_ui_pytorch)):
                for j in range(len(self.transform_ui_pytorch[i])):
                    if((index-1)==i):
                        self.transform_ui_pytorch[i][j].show();
                    else:
                        self.transform_ui_pytorch[i][j].hide();

            for i in range(len(self.transform_ui_keras)):
                for j in range(len(self.transform_ui_keras[i])):
                    self.transform_ui_keras[i][j].hide();
            for i in range(len(self.transform_ui_mxnet)):
                for j in range(len(self.transform_ui_mxnet[i])):
                    self.transform_ui_mxnet[i][j].hide();
        


    def add_transform(self):
        self.system["update"]["transforms"]["active"] = True;
        if(self.system["backend"] == "Mxnet-1.5.1"):
            if(self.current_transform["name"] == self.mxnet_transforms_list[1]):
                self.current_transform["params"]["input_size"] = self.mx_tf1_e1.text();
                self.current_transform["params"]["scale"] = [self.mx_tf1_e2_1.text(), self.mx_tf1_e2_2.text()];
                self.current_transform["params"]["ratio"] = [self.mx_tf1_e3_1.text(), self.mx_tf1_e3_2.text()];
                if(self.mx_tf1_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.mx_tf1_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.mx_tf1_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.mxnet_transforms_list[2]):
                self.current_transform["params"]["input_size"] = self.mx_tf2_e1.text();
                if(self.mx_tf2_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.mx_tf2_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.mx_tf2_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.mxnet_transforms_list[3]):
                self.current_transform["params"]["brightness"] = self.mx_tf3_e1.text();
                self.current_transform["params"]["contrast"] = self.mx_tf3_e2.text();
                self.current_transform["params"]["saturation"] = self.mx_tf3_e3.text();
                self.current_transform["params"]["hue"] = self.mx_tf3_e4.text();
                if(self.mx_tf3_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.mx_tf3_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.mx_tf3_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.mxnet_transforms_list[4]):
                self.current_transform["params"]["probability"] = self.mx_tf4_e1.text();
                if(self.mx_tf4_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.mx_tf4_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.mx_tf4_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.mxnet_transforms_list[5]):
                self.current_transform["params"]["probability"] = self.mx_tf5_e1.text();
                if(self.mx_tf5_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.mx_tf5_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.mx_tf5_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.mxnet_transforms_list[6]):
                self.current_transform["params"]["alpha"] = self.mx_tf6_e1.text();
                if(self.mx_tf6_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.mx_tf6_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.mx_tf6_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.mxnet_transforms_list[7]):
                self.current_transform["params"]["input_size"] = self.mx_tf7_e1.text();
                if(self.mx_tf7_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.mx_tf7_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.mx_tf7_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.mxnet_transforms_list[8]):

                self.current_transform["params"]["mean"] = [self.mx_tf8_e1_1.text(), self.mx_tf8_e1_2.text(), self.mx_tf8_e1_3.text()];
                self.current_transform["params"]["std"] = [self.mx_tf8_e2_1.text(), self.mx_tf8_e2_2.text(), self.mx_tf8_e2_3.text()];
                if(self.mx_tf8_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.mx_tf8_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.mx_tf8_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);


        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            if(self.current_transform["name"] == self.keras_transforms_list[1]):

                self.current_transform["params"]["brightness"] = self.ke_tf1_e1.text();
                self.current_transform["params"]["contrast"] = self.ke_tf1_e2.text();
                self.current_transform["params"]["saturation"] = self.ke_tf1_e3.text();
                self.current_transform["params"]["hue"] = self.ke_tf1_e4.text();
                if(self.ke_tf1_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.ke_tf1_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.ke_tf1_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.keras_transforms_list[2]):
                self.current_transform["params"]["input_size"] = self.ke_tf2_e1.text();
                self.current_transform["params"]["scale"] = [self.ke_tf2_e2_1.text(), self.ke_tf2_e2_2.text()];
                self.current_transform["params"]["ratio"] = [self.ke_tf2_e3_1.text(), self.ke_tf2_e3_2.text()];
                if(self.ke_tf2_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.ke_tf2_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.ke_tf2_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.keras_transforms_list[3]):
                self.current_transform["params"]["probability"] = self.ke_tf3_e1.text();
                if(self.ke_tf3_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.ke_tf3_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.ke_tf3_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.keras_transforms_list[4]):
                self.current_transform["params"]["probability"] = self.ke_tf4_e1.text();
                if(self.ke_tf4_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.ke_tf4_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.ke_tf4_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.keras_transforms_list[5]):
                self.current_transform["params"]["degrees"] = self.ke_tf5_e1.text();
                if(self.ke_tf5_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.ke_tf5_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.ke_tf5_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.keras_transforms_list[6]):
                self.current_transform["params"]["mean"] = [self.ke_tf6_e1_1.text(), self.ke_tf6_e1_2.text(), self.ke_tf6_e1_3.text()];
                if(self.ke_tf6_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.ke_tf6_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.ke_tf6_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.keras_transforms_list[7]):
                self.current_transform["params"]["mean"] = [self.ke_tf7_e1_1.text(), self.ke_tf7_e1_2.text(), self.ke_tf7_e1_3.text()];
                self.current_transform["params"]["std"] = [self.ke_tf7_e2_1.text(), self.ke_tf7_e2_2.text(), self.ke_tf7_e2_3.text()];
                if(self.ke_tf7_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.ke_tf7_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.ke_tf7_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);



        elif(self.system["backend"] == "Pytorch-1.3.1"):

            if(self.current_transform["name"] == self.pytorch_transforms_list[1]):

                self.current_transform["params"]["input_size"] = self.py_tf1_e1.text();
                if(self.py_tf1_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf1_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf1_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.pytorch_transforms_list[2]):

                self.current_transform["params"]["brightness"] = self.py_tf2_e1.text();
                self.current_transform["params"]["contrast"] = self.py_tf2_e2.text();
                self.current_transform["params"]["saturation"] = self.py_tf2_e3.text();
                self.current_transform["params"]["hue"] = self.py_tf2_e4.text();
                if(self.py_tf2_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf2_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf2_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.pytorch_transforms_list[3]):

                self.current_transform["params"]["degrees"] = self.py_tf3_e1.text();
                self.current_transform["params"]["translate"] = [self.py_tf3_e2_1.text(), self.py_tf3_e2_2.text()];
                self.current_transform["params"]["scale"] = [self.py_tf3_e3_1.text(), self.py_tf3_e3_2.text()];
                self.current_transform["params"]["sheer"] = [self.py_tf3_e4_1.text(), self.py_tf3_e4_2.text()];
                if(self.py_tf3_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf3_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf3_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.pytorch_transforms_list[4]):

                self.current_transform["params"]["input_size"] = self.py_tf4_e1.text();
                if(self.py_tf4_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf4_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf4_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.pytorch_transforms_list[5]):

                self.current_transform["params"]["probability"] = self.py_tf5_e1.text();
                if(self.py_tf5_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf5_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf5_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.pytorch_transforms_list[6]):

                self.current_transform["params"]["distortion_scale"] = self.py_tf6_e1.text();
                self.current_transform["params"]["probability"] = self.py_tf6_e2.text();
                if(self.py_tf6_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf6_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf6_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.pytorch_transforms_list[7]):

                self.current_transform["params"]["input_size"] = self.py_tf7_e1.text();
                self.current_transform["params"]["scale"] = [self.py_tf7_e2_1.text(), self.py_tf7_e2_2.text()];
                self.current_transform["params"]["ratio"] = [self.py_tf7_e3_1.text(), self.py_tf7_e3_2.text()];
                if(self.py_tf7_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf7_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf7_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.pytorch_transforms_list[8]):

                self.current_transform["params"]["degrees"] = self.py_tf8_e1.text();
                if(self.py_tf8_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf8_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf8_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.pytorch_transforms_list[9]):

                self.current_transform["params"]["probability"] = self.py_tf9_e1.text();
                if(self.py_tf9_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf9_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf9_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.pytorch_transforms_list[10]):

                self.current_transform["params"]["input_size"] = self.py_tf10_e1.text();
                if(self.py_tf10_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf10_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf10_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);

            elif(self.current_transform["name"] == self.pytorch_transforms_list[11]):

                self.current_transform["params"]["mean"] = [self.py_tf11_e1_1.text(), self.py_tf11_e1_2.text(), self.py_tf11_e1_3.text()];
                self.current_transform["params"]["std"] = [self.py_tf11_e2_1.text(), self.py_tf11_e2_2.text(), self.py_tf11_e2_3.text()];
                if(self.py_tf11_cbox1.isChecked()):
                    self.current_transform["params"]["train"] = "True";
                else:
                    self.current_transform["params"]["train"] = "False";
                if(self.py_tf11_cbox2.isChecked()):
                    self.current_transform["params"]["val"] = "True";
                else:
                    self.current_transform["params"]["val"] = "False";
                if(self.py_tf11_cbox3.isChecked()):
                    self.current_transform["params"]["test"] = "True";
                else:
                    self.current_transform["params"]["test"] = "False";
                self.system["update"]["transforms"]["value"].append(self.current_transform);








        wr = "";
        for i in range(len(self.system["update"]["transforms"]["value"])):
            tmp = json.dumps(self.system["update"]["transforms"]["value"][i], indent=4)
            wr += "{}\n".format(tmp);
        self.tb1.setText(wr);


    def remove_transform(self):
        if(len(self.system["update"]["transforms"]["value"]) > 0):
            del self.system["update"]["transforms"]["value"][-1]
        else:
            self.system["update"]["transforms"]["active"] = False;

        wr = "";
        for i in range(len(self.system["update"]["transforms"]["value"])):
            tmp = json.dumps(self.system["update"]["transforms"]["value"][i], indent=4)
            wr += "{}\n".format(tmp);
        self.tb1.setText(wr);

        if(len(self.system["update"]["transforms"]["value"]) == 0):
            self.system["update"]["transforms"]["active"] = False;

    def clear_transform(self):
        self.system["update"]["transforms"]["value"] = [];
        self.system["update"]["transforms"]["active"] = False;

        wr = "";
        for i in range(len(self.system["update"]["transforms"]["value"])):
            tmp = json.dumps(self.system["update"]["transforms"]["value"][i], indent=4)
            wr += "{}\n".format(tmp);
        self.tb1.setText(wr);



    def forward(self):        
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_model_param.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_data_param.emit();



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateTransformParam()
screen.show()
sys.exit(app.exec_())
'''