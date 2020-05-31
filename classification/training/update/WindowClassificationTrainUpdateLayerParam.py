import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateLayerParam(QtWidgets.QWidget):

    forward_train_param = QtCore.pyqtSignal();
    backward_model_param = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Update Model Layer Params'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 600
        self.layer_ui_mxnet = [];
        self.layer_ui_keras = [];
        self.layer_ui_pytorch = [];
        self.current_layer = {};
        self.current_layer["name"] = "";
        self.current_layer["params"] = {};
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



        self.cb1 = QComboBox(self);
        self.cb1.move(20, 20);
        self.cb1.activated.connect(self.select_layer);

        self.cb2 = QComboBox(self);
        self.cb2.move(20, 20);
        self.cb2.activated.connect(self.select_layer);

        self.cb3 = QComboBox(self);
        self.cb3.move(20, 20);
        self.cb3.activated.connect(self.select_layer);



        self.mxnet_layers_list = ["Select Layer/Activation", "append_linear", "append_dropout", "relu", "sigmoid", "tanh", 
                                        "softplus", "softsign", "elu", "leaky_relu",
                                        "prelu", "selu", "swish"];
        self.keras_layers_list = ["Select Layer/Activation", "append_linear", "append_dropout", "relu", "elu", "leaky_relu", 
                                        "prelu", "threshold", "softmax", "selu", "softplus", "softsign", "tanh", "sigmoid"];
        self.pytorch_layers_list =  ["Select Layer/Activation", "append_linear", "append_dropout", "relu", "sigmoid", "tanh", 
                                            "softplus", "softsign",  "elu", "leaky_relu", "prelu", "selu", "hardshrink", "hardtanh", 
                                            "logsigmoid", "relu6", "rrelu", "celu", "softshrink", "tanhshrink", "threshold", 
                                            "softmin", "softmax", "logsoftmax"];
        

        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.cb1.addItems(self.mxnet_layers_list);
            self.cb1.show();
            self.cb2.hide();
            self.cb3.hide();
        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.cb2.addItems(self.keras_layers_list);
            self.cb2.show();
            self.cb1.hide();
            self.cb3.hide();
        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.cb3.addItems(self.pytorch_layers_list);
            self.cb3.show();
            self.cb1.hide();
            self.cb2.hide();



        tmp = [];
        self.mx_la1_l1 = QLabel(self);
        self.mx_la1_l1.setText("1. Num neurons:");
        self.mx_la1_l1.move(20, 100);
        tmp.append(self.mx_la1_l1);

        self.mx_la1_e1 = QLineEdit(self)
        self.mx_la1_e1.move(150, 100);
        self.mx_la1_e1.setText("1024");
        tmp.append(self.mx_la1_e1);

        self.mx_la1_l2 = QLabel(self);
        self.mx_la1_l2.setText("2. Set as final Layer:");
        self.mx_la1_l2.move(20, 150);
        tmp.append(self.mx_la1_l2);

        self.mx_la1_cb2 = QComboBox(self);
        self.mx_la1_cb2.move(200, 150);
        self.mx_la1_cb2.activated.connect(self.set_final_layer);
        self.mx_la1_cb2.addItems(["No", "Yes"]);
        tmp.append(self.mx_la1_cb2);

        self.layer_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_la2_l1 = QLabel(self);
        self.mx_la2_l1.setText("1. Probability (0-1):");
        self.mx_la2_l1.move(20, 100);
        tmp.append(self.mx_la2_l1);

        self.mx_la2_e1 = QLineEdit(self)
        self.mx_la2_e1.move(180, 100);
        self.mx_la2_e1.setText("0.2");
        tmp.append(self.mx_la2_e1);

        self.mx_la2_l2 = QLabel(self);
        self.mx_la2_l2.setText("2. Set as final Layer:");
        self.mx_la2_l2.move(20, 150);
        tmp.append(self.mx_la2_l2);

        self.mx_la2_cb2 = QComboBox(self);
        self.mx_la2_cb2.move(200, 150);
        self.mx_la2_cb2.activated.connect(self.set_final_layer);
        self.mx_la2_cb2.addItems(["No", "Yes"]);
        tmp.append(self.mx_la2_cb2);

        self.layer_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_la3_l1 = QLabel(self);
        self.mx_la3_l1.setText("1. Set as final layer:");
        self.mx_la3_l1.move(20, 100);
        tmp.append(self.mx_la3_l1);

        self.mx_la3_cb1 = QComboBox(self);
        self.mx_la3_cb1.move(200, 100);
        self.mx_la3_cb1.activated.connect(self.set_final_layer);
        self.mx_la3_cb1.addItems(["No", "Yes"]);
        tmp.append(self.mx_la3_cb1);

        self.layer_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_la4_l1 = QLabel(self);
        self.mx_la4_l1.setText("1. Set as final layer:");
        self.mx_la4_l1.move(20, 100);
        tmp.append(self.mx_la4_l1);

        self.mx_la4_cb1 = QComboBox(self);
        self.mx_la4_cb1.move(200, 100);
        self.mx_la4_cb1.activated.connect(self.set_final_layer);
        self.mx_la4_cb1.addItems(["No", "Yes"]);
        tmp.append(self.mx_la4_cb1);

        self.layer_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_la5_l1 = QLabel(self);
        self.mx_la5_l1.setText("1. Set as final layer:");
        self.mx_la5_l1.move(20, 100);
        tmp.append(self.mx_la5_l1);

        self.mx_la5_cb1 = QComboBox(self);
        self.mx_la5_cb1.move(200, 100);
        self.mx_la5_cb1.activated.connect(self.set_final_layer);
        self.mx_la5_cb1.addItems(["No", "Yes"]);
        tmp.append(self.mx_la5_cb1);

        self.layer_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_la6_l1 = QLabel(self);
        self.mx_la6_l1.setText("1. Beta:");
        self.mx_la6_l1.move(20, 100);
        tmp.append(self.mx_la6_l1);

        self.mx_la6_e1 = QLineEdit(self)
        self.mx_la6_e1.move(180, 100);
        self.mx_la6_e1.setText("1.0");
        tmp.append(self.mx_la6_e1);


        self.mx_la6_l2 = QLabel(self);
        self.mx_la6_l2.setText("2. Threshold:");
        self.mx_la6_l2.move(20, 150);
        tmp.append(self.mx_la6_l2);

        self.mx_la6_e2 = QLineEdit(self)
        self.mx_la6_e2.move(180, 150);
        self.mx_la6_e2.setText("20");
        tmp.append(self.mx_la6_e2);


        self.mx_la6_l3 = QLabel(self);
        self.mx_la6_l3.setText("3. Set as final Layer:");
        self.mx_la6_l3.move(20, 200);
        tmp.append(self.mx_la6_l3);

        self.mx_la6_cb3 = QComboBox(self);
        self.mx_la6_cb3.move(200, 200);
        self.mx_la6_cb3.activated.connect(self.set_final_layer);
        self.mx_la6_cb3.addItems(["No", "Yes"]);
        tmp.append(self.mx_la6_cb3);

        self.layer_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_la7_l1 = QLabel(self);
        self.mx_la7_l1.setText("1. Set as final layer:");
        self.mx_la7_l1.move(20, 100);
        tmp.append(self.mx_la7_l1);

        self.mx_la7_cb1 = QComboBox(self);
        self.mx_la7_cb1.move(200, 100);
        self.mx_la7_cb1.activated.connect(self.set_final_layer);
        self.mx_la7_cb1.addItems(["No", "Yes"]);
        tmp.append(self.mx_la7_cb1);

        self.layer_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_la8_l1 = QLabel(self);
        self.mx_la8_l1.setText("1. Alpha:");
        self.mx_la8_l1.move(20, 100);
        tmp.append(self.mx_la8_l1);

        self.mx_la8_e1 = QLineEdit(self)
        self.mx_la8_e1.move(150, 100);
        self.mx_la8_e1.setText("1.0");
        tmp.append(self.mx_la8_e1);

        self.mx_la8_l2 = QLabel(self);
        self.mx_la8_l2.setText("2. Set as final Layer:");
        self.mx_la8_l2.move(20, 150);
        tmp.append(self.mx_la8_l2);

        self.mx_la8_cb2 = QComboBox(self);
        self.mx_la8_cb2.move(200, 150);
        self.mx_la8_cb2.activated.connect(self.set_final_layer);
        self.mx_la8_cb2.addItems(["No", "Yes"]);
        tmp.append(self.mx_la8_cb2);

        self.layer_ui_mxnet.append(tmp)
        





        tmp = [];
        self.mx_la9_l1 = QLabel(self);
        self.mx_la9_l1.setText("1. Negative slope:");
        self.mx_la9_l1.move(20, 100);
        tmp.append(self.mx_la9_l1);

        self.mx_la9_e1 = QLineEdit(self)
        self.mx_la9_e1.move(150, 100);
        self.mx_la9_e1.setText("0.01");
        tmp.append(self.mx_la9_e1);

        self.mx_la9_l2 = QLabel(self);
        self.mx_la9_l2.setText("2. Set as final Layer:");
        self.mx_la9_l2.move(20, 150);
        tmp.append(self.mx_la9_l2);

        self.mx_la9_cb2 = QComboBox(self);
        self.mx_la9_cb2.move(200, 150);
        self.mx_la9_cb2.activated.connect(self.set_final_layer);
        self.mx_la9_cb2.addItems(["No", "Yes"]);
        tmp.append(self.mx_la9_cb2);

        self.layer_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_la10_l1 = QLabel(self);
        self.mx_la10_l1.setText("1. Initializing param:");
        self.mx_la10_l1.move(20, 100);
        tmp.append(self.mx_la10_l1);

        self.mx_la10_e1 = QLineEdit(self)
        self.mx_la10_e1.move(200, 100);
        self.mx_la10_e1.setText("0.25");
        tmp.append(self.mx_la10_e1);

        self.mx_la10_l2 = QLabel(self);
        self.mx_la10_l2.setText("2. Set as final Layer:");
        self.mx_la10_l2.move(20, 150);
        tmp.append(self.mx_la10_l2);

        self.mx_la10_cb2 = QComboBox(self);
        self.mx_la10_cb2.move(200, 150);
        self.mx_la10_cb2.activated.connect(self.set_final_layer);
        self.mx_la10_cb2.addItems(["No", "Yes"]);
        tmp.append(self.mx_la10_cb2);

        self.layer_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_la11_l1 = QLabel(self);
        self.mx_la11_l1.setText("1. Set as final layer:");
        self.mx_la11_l1.move(20, 100);
        tmp.append(self.mx_la11_l1);

        self.mx_la11_cb1 = QComboBox(self);
        self.mx_la11_cb1.move(200, 100);
        self.mx_la11_cb1.activated.connect(self.set_final_layer);
        self.mx_la11_cb1.addItems(["No", "Yes"]);
        tmp.append(self.mx_la11_cb1);

        self.layer_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_la12_l1 = QLabel(self);
        self.mx_la12_l1.setText("1. Beta:");
        self.mx_la12_l1.move(20, 100);
        tmp.append(self.mx_la12_l1);

        self.mx_la12_e1 = QLineEdit(self)
        self.mx_la12_e1.move(150, 100);
        self.mx_la12_e1.setText("1.0");
        tmp.append(self.mx_la12_e1);

        self.mx_la12_l2 = QLabel(self);
        self.mx_la12_l2.setText("2. Set as final Layer:");
        self.mx_la12_l2.move(20, 150);
        tmp.append(self.mx_la12_l2);

        self.mx_la12_cb2 = QComboBox(self);
        self.mx_la12_cb2.move(200, 150);
        self.mx_la12_cb2.activated.connect(self.set_final_layer);
        self.mx_la12_cb2.addItems(["No", "Yes"]);
        tmp.append(self.mx_la12_cb2);

        self.layer_ui_mxnet.append(tmp)






        tmp = [];
        self.ke_la1_l1 = QLabel(self);
        self.ke_la1_l1.setText("1. Num neurons:");
        self.ke_la1_l1.move(20, 100);
        tmp.append(self.ke_la1_l1);

        self.ke_la1_e1 = QLineEdit(self)
        self.ke_la1_e1.move(150, 100);
        self.ke_la1_e1.setText("1024");
        tmp.append(self.ke_la1_e1);

        self.ke_la1_l2 = QLabel(self);
        self.ke_la1_l2.setText("2. Set as final Layer:");
        self.ke_la1_l2.move(20, 150);
        tmp.append(self.ke_la1_l2);

        self.ke_la1_cb2 = QComboBox(self);
        self.ke_la1_cb2.move(200, 150);
        self.ke_la1_cb2.activated.connect(self.set_final_layer);
        self.ke_la1_cb2.addItems(["No", "Yes"]);
        tmp.append(self.ke_la1_cb2);

        self.layer_ui_keras.append(tmp)





        tmp = [];
        self.ke_la2_l1 = QLabel(self);
        self.ke_la2_l1.setText("1. Probability (0-1):");
        self.ke_la2_l1.move(20, 100);
        tmp.append(self.ke_la2_l1);

        self.ke_la2_e1 = QLineEdit(self)
        self.ke_la2_e1.move(180, 100);
        self.ke_la2_e1.setText("0.2");
        tmp.append(self.ke_la2_e1);

        self.ke_la2_l2 = QLabel(self);
        self.ke_la2_l2.setText("2. Set as final Layer:");
        self.ke_la2_l2.move(20, 150);
        tmp.append(self.ke_la2_l2);

        self.ke_la2_cb2 = QComboBox(self);
        self.ke_la2_cb2.move(200, 150);
        self.ke_la2_cb2.activated.connect(self.set_final_layer);
        self.ke_la2_cb2.addItems(["No", "Yes"]);
        tmp.append(self.ke_la2_cb2);

        self.layer_ui_keras.append(tmp)





        tmp = [];
        self.ke_la3_l1 = QLabel(self);
        self.ke_la3_l1.setText("1. Set as final layer:");
        self.ke_la3_l1.move(20, 100);
        tmp.append(self.ke_la3_l1);

        self.ke_la3_cb1 = QComboBox(self);
        self.ke_la3_cb1.move(200, 100);
        self.ke_la3_cb1.activated.connect(self.set_final_layer);
        self.ke_la3_cb1.addItems(["No", "Yes"]);
        tmp.append(self.ke_la3_cb1);

        self.layer_ui_keras.append(tmp)




        tmp = [];
        self.ke_la4_l1 = QLabel(self);
        self.ke_la4_l1.setText("1. Alpha:");
        self.ke_la4_l1.move(20, 100);
        tmp.append(self.ke_la4_l1);

        self.ke_la4_e1 = QLineEdit(self)
        self.ke_la4_e1.move(180, 100);
        self.ke_la4_e1.setText("1.0");
        tmp.append(self.ke_la4_e1);

        self.ke_la4_l2 = QLabel(self);
        self.ke_la4_l2.setText("2. Set as final Layer:");
        self.ke_la4_l2.move(20, 150);
        tmp.append(self.ke_la4_l2);

        self.ke_la4_cb2 = QComboBox(self);
        self.ke_la4_cb2.move(200, 150);
        self.ke_la4_cb2.activated.connect(self.set_final_layer);
        self.ke_la4_cb2.addItems(["No", "Yes"]);
        tmp.append(self.ke_la4_cb2);

        self.layer_ui_keras.append(tmp)




        tmp = [];
        self.ke_la5_l1 = QLabel(self);
        self.ke_la5_l1.setText("1. Negative Slope:");
        self.ke_la5_l1.move(20, 100);
        tmp.append(self.ke_la5_l1);

        self.ke_la5_e1 = QLineEdit(self)
        self.ke_la5_e1.move(180, 100);
        self.ke_la5_e1.setText("1.0");
        tmp.append(self.ke_la5_e1);

        self.ke_la5_l2 = QLabel(self);
        self.ke_la5_l2.setText("2. Set as final Layer:");
        self.ke_la5_l2.move(20, 150);
        tmp.append(self.ke_la5_l2);

        self.ke_la5_cb2 = QComboBox(self);
        self.ke_la5_cb2.move(200, 150);
        self.ke_la5_cb2.activated.connect(self.set_final_layer);
        self.ke_la5_cb2.addItems(["No", "Yes"]);
        tmp.append(self.ke_la5_cb2);

        self.layer_ui_keras.append(tmp)




        tmp = [];
        self.ke_la6_l1 = QLabel(self);
        self.ke_la6_l1.setText("1. Initialing param:");
        self.ke_la6_l1.move(20, 100);
        tmp.append(self.ke_la6_l1);

        self.ke_la6_e1 = QLineEdit(self)
        self.ke_la6_e1.move(180, 100);
        self.ke_la6_e1.setText("1.0");
        tmp.append(self.ke_la6_e1);

        self.ke_la6_l2 = QLabel(self);
        self.ke_la6_l2.setText("2. Set as final Layer:");
        self.ke_la6_l2.move(20, 150);
        tmp.append(self.ke_la6_l2);

        self.ke_la6_cb2 = QComboBox(self);
        self.ke_la6_cb2.move(200, 150);
        self.ke_la6_cb2.activated.connect(self.set_final_layer);
        self.ke_la6_cb2.addItems(["No", "Yes"]);
        tmp.append(self.ke_la6_cb2);

        self.layer_ui_keras.append(tmp)




        tmp = [];
        self.ke_la7_l1 = QLabel(self);
        self.ke_la7_l1.setText("1. Threshold:");
        self.ke_la7_l1.move(20, 100);
        tmp.append(self.ke_la7_l1);

        self.ke_la7_e1 = QLineEdit(self)
        self.ke_la7_e1.move(180, 100);
        self.ke_la7_e1.setText("None");
        tmp.append(self.ke_la7_e1);


        self.ke_la7_l2 = QLabel(self);
        self.ke_la7_l2.setText("2. Resplacement value:");
        self.ke_la7_l2.move(20, 150);
        tmp.append(self.ke_la7_l2);

        self.ke_la7_e2 = QLineEdit(self)
        self.ke_la7_e2.move(180, 150);
        self.ke_la7_e2.setText("None");
        tmp.append(self.ke_la7_e2);


        self.ke_la7_l3 = QLabel(self);
        self.ke_la7_l3.setText("3. Set as final Layer:");
        self.ke_la7_l3.move(20, 200);
        tmp.append(self.ke_la7_l3);

        self.ke_la7_cb3 = QComboBox(self);
        self.ke_la7_cb3.move(200, 200);
        self.ke_la7_cb3.activated.connect(self.set_final_layer);
        self.ke_la7_cb3.addItems(["No", "Yes"]);
        tmp.append(self.ke_la7_cb3);

        self.layer_ui_keras.append(tmp)




        tmp = [];
        self.ke_la8_l1 = QLabel(self);
        self.ke_la8_l1.setText("1. Set as final layer:");
        self.ke_la8_l1.move(20, 100);
        tmp.append(self.ke_la8_l1);

        self.ke_la8_cb1 = QComboBox(self);
        self.ke_la8_cb1.move(200, 100);
        self.ke_la8_cb1.activated.connect(self.set_final_layer);
        self.ke_la8_cb1.addItems(["No", "Yes"]);
        tmp.append(self.ke_la8_cb1);

        self.layer_ui_keras.append(tmp)




        tmp = [];
        self.ke_la9_l1 = QLabel(self);
        self.ke_la9_l1.setText("1. Set as final layer:");
        self.ke_la9_l1.move(20, 100);
        tmp.append(self.ke_la9_l1);

        self.ke_la9_cb1 = QComboBox(self);
        self.ke_la9_cb1.move(200, 100);
        self.ke_la9_cb1.activated.connect(self.set_final_layer);
        self.ke_la9_cb1.addItems(["No", "Yes"]);
        tmp.append(self.ke_la9_cb1);

        self.layer_ui_keras.append(tmp)





        tmp = [];
        self.ke_la10_l1 = QLabel(self);
        self.ke_la10_l1.setText("1. Beta:");
        self.ke_la10_l1.move(20, 100);
        tmp.append(self.ke_la10_l1);

        self.ke_la10_e1 = QLineEdit(self)
        self.ke_la10_e1.move(180, 100);
        self.ke_la10_e1.setText("1.0");
        tmp.append(self.ke_la10_e1);


        self.ke_la10_l2 = QLabel(self);
        self.ke_la10_l2.setText("2. Threshold:");
        self.ke_la10_l2.move(20, 150);
        tmp.append(self.ke_la10_l2);

        self.ke_la10_e2 = QLineEdit(self)
        self.ke_la10_e2.move(180, 150);
        self.ke_la10_e2.setText("20.0");
        tmp.append(self.ke_la10_e2);


        self.ke_la10_l3 = QLabel(self);
        self.ke_la10_l3.setText("3. Set as final Layer:");
        self.ke_la10_l3.move(20, 200);
        tmp.append(self.ke_la10_l3);

        self.ke_la10_cb3 = QComboBox(self);
        self.ke_la10_cb3.move(200, 200);
        self.ke_la10_cb3.activated.connect(self.set_final_layer);
        self.ke_la10_cb3.addItems(["No", "Yes"]);
        tmp.append(self.ke_la10_cb3);

        self.layer_ui_keras.append(tmp)





        tmp = [];
        self.ke_la11_l1 = QLabel(self);
        self.ke_la11_l1.setText("1. Set as final layer:");
        self.ke_la11_l1.move(20, 100);
        tmp.append(self.ke_la11_l1);

        self.ke_la11_cb1 = QComboBox(self);
        self.ke_la11_cb1.move(200, 100);
        self.ke_la11_cb1.activated.connect(self.set_final_layer);
        self.ke_la11_cb1.addItems(["No", "Yes"]);
        tmp.append(self.ke_la11_cb1);

        self.layer_ui_keras.append(tmp)





        tmp = [];
        self.ke_la12_l1 = QLabel(self);
        self.ke_la12_l1.setText("1. Set as final layer:");
        self.ke_la12_l1.move(20, 100);
        tmp.append(self.ke_la12_l1);

        self.ke_la12_cb1 = QComboBox(self);
        self.ke_la12_cb1.move(200, 100);
        self.ke_la12_cb1.activated.connect(self.set_final_layer);
        self.ke_la12_cb1.addItems(["No", "Yes"]);
        tmp.append(self.ke_la12_cb1);

        self.layer_ui_keras.append(tmp)




        tmp = [];
        self.ke_la13_l1 = QLabel(self);
        self.ke_la13_l1.setText("1. Set as final layer:");
        self.ke_la13_l1.move(20, 100);
        tmp.append(self.ke_la13_l1);

        self.ke_la13_cb1 = QComboBox(self);
        self.ke_la13_cb1.move(200, 100);
        self.ke_la13_cb1.activated.connect(self.set_final_layer);
        self.ke_la13_cb1.addItems(["No", "Yes"]);
        tmp.append(self.ke_la13_cb1);

        self.layer_ui_keras.append(tmp)













        tmp = [];
        self.py_la1_l1 = QLabel(self);
        self.py_la1_l1.setText("1. Num neurons:");
        self.py_la1_l1.move(20, 100);
        tmp.append(self.py_la1_l1);

        self.py_la1_e1 = QLineEdit(self)
        self.py_la1_e1.move(150, 100);
        self.py_la1_e1.setText("1024");
        tmp.append(self.py_la1_e1);

        self.py_la1_l2 = QLabel(self);
        self.py_la1_l2.setText("2. Set as final Layer:");
        self.py_la1_l2.move(20, 150);
        tmp.append(self.py_la1_l2);

        self.py_la1_cb2 = QComboBox(self);
        self.py_la1_cb2.move(200, 150);
        self.py_la1_cb2.activated.connect(self.set_final_layer);
        self.py_la1_cb2.addItems(["No", "Yes"]);
        tmp.append(self.py_la1_cb2);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la2_l1 = QLabel(self);
        self.py_la2_l1.setText("1. Probability (0-1):");
        self.py_la2_l1.move(20, 100);
        tmp.append(self.py_la2_l1);

        self.py_la2_e1 = QLineEdit(self)
        self.py_la2_e1.move(180, 100);
        self.py_la2_e1.setText("0.2");
        tmp.append(self.py_la2_e1);

        self.py_la2_l2 = QLabel(self);
        self.py_la2_l2.setText("2. Set as final Layer:");
        self.py_la2_l2.move(20, 150);
        tmp.append(self.py_la2_l2);

        self.py_la2_cb2 = QComboBox(self);
        self.py_la2_cb2.move(200, 150);
        self.py_la2_cb2.activated.connect(self.set_final_layer);
        self.py_la2_cb2.addItems(["No", "Yes"]);
        tmp.append(self.py_la2_cb2);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la3_l1 = QLabel(self);
        self.py_la3_l1.setText("1. Set as final layer:");
        self.py_la3_l1.move(20, 100);
        tmp.append(self.py_la3_l1);

        self.py_la3_cb1 = QComboBox(self);
        self.py_la3_cb1.move(200, 100);
        self.py_la3_cb1.activated.connect(self.set_final_layer);
        self.py_la3_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la3_cb1);

        self.layer_ui_pytorch.append(tmp)




        tmp = [];
        self.py_la4_l1 = QLabel(self);
        self.py_la4_l1.setText("1. Set as final layer:");
        self.py_la4_l1.move(20, 100);
        tmp.append(self.py_la4_l1);

        self.py_la4_cb1 = QComboBox(self);
        self.py_la4_cb1.move(200, 100);
        self.py_la4_cb1.activated.connect(self.set_final_layer);
        self.py_la4_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la4_cb1);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la5_l1 = QLabel(self);
        self.py_la5_l1.setText("1. Set as final layer:");
        self.py_la5_l1.move(20, 100);
        tmp.append(self.py_la5_l1);

        self.py_la5_cb1 = QComboBox(self);
        self.py_la5_cb1.move(200, 100);
        self.py_la5_cb1.activated.connect(self.set_final_layer);
        self.py_la5_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la5_cb1);

        self.layer_ui_pytorch.append(tmp)




        tmp = [];
        self.py_la6_l1 = QLabel(self);
        self.py_la6_l1.setText("1. Beta:");
        self.py_la6_l1.move(20, 100);
        tmp.append(self.py_la6_l1);

        self.py_la6_e1 = QLineEdit(self)
        self.py_la6_e1.move(180, 100);
        self.py_la6_e1.setText("1.0");
        tmp.append(self.py_la6_e1);


        self.py_la6_l2 = QLabel(self);
        self.py_la6_l2.setText("2. Threshold:");
        self.py_la6_l2.move(20, 150);
        tmp.append(self.py_la6_l2);

        self.py_la6_e2 = QLineEdit(self)
        self.py_la6_e2.move(180, 150);
        self.py_la6_e2.setText("20.0");
        tmp.append(self.py_la6_e2);


        self.py_la6_l3 = QLabel(self);
        self.py_la6_l3.setText("3. Set as final Layer:");
        self.py_la6_l3.move(20, 200);
        tmp.append(self.py_la6_l3);

        self.py_la6_cb3 = QComboBox(self);
        self.py_la6_cb3.move(200, 200);
        self.py_la6_cb3.activated.connect(self.set_final_layer);
        self.py_la6_cb3.addItems(["No", "Yes"]);
        tmp.append(self.py_la6_cb3);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la7_l1 = QLabel(self);
        self.py_la7_l1.setText("1. Set as final layer:");
        self.py_la7_l1.move(20, 100);
        tmp.append(self.py_la7_l1);

        self.py_la7_cb1 = QComboBox(self);
        self.py_la7_cb1.move(200, 100);
        self.py_la7_cb1.activated.connect(self.set_final_layer);
        self.py_la7_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la7_cb1);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la8_l1 = QLabel(self);
        self.py_la8_l1.setText("1. Alpha:");
        self.py_la8_l1.move(20, 100);
        tmp.append(self.py_la8_l1);

        self.py_la8_e1 = QLineEdit(self)
        self.py_la8_e1.move(180, 100);
        self.py_la8_e1.setText("1.0");
        tmp.append(self.py_la8_e1);

        self.py_la8_l2 = QLabel(self);
        self.py_la8_l2.setText("2. Set as final Layer:");
        self.py_la8_l2.move(20, 150);
        tmp.append(self.py_la8_l2);

        self.py_la8_cb2 = QComboBox(self);
        self.py_la8_cb2.move(200, 150);
        self.py_la8_cb2.activated.connect(self.set_final_layer);
        self.py_la8_cb2.addItems(["No", "Yes"]);
        tmp.append(self.py_la8_cb2);

        self.layer_ui_pytorch.append(tmp)




        tmp = [];
        self.py_la9_l1 = QLabel(self);
        self.py_la9_l1.setText("1. Negative Slope:");
        self.py_la9_l1.move(20, 100);
        tmp.append(self.py_la9_l1);

        self.py_la9_e1 = QLineEdit(self)
        self.py_la9_e1.move(180, 100);
        self.py_la9_e1.setText("0.01");
        tmp.append(self.py_la9_e1);

        self.py_la9_l2 = QLabel(self);
        self.py_la9_l2.setText("2. Set as final Layer:");
        self.py_la9_l2.move(20, 150);
        tmp.append(self.py_la9_l2);

        self.py_la9_cb2 = QComboBox(self);
        self.py_la9_cb2.move(200, 150);
        self.py_la9_cb2.activated.connect(self.set_final_layer);
        self.py_la9_cb2.addItems(["No", "Yes"]);
        tmp.append(self.py_la9_cb2);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la10_l1 = QLabel(self);
        self.py_la10_l1.setText("1. Initializing param:");
        self.py_la10_l1.move(20, 100);
        tmp.append(self.py_la10_l1);

        self.py_la10_e1 = QLineEdit(self)
        self.py_la10_e1.move(180, 100);
        self.py_la10_e1.setText("0.01");
        tmp.append(self.py_la10_e1);

        self.py_la10_l2 = QLabel(self);
        self.py_la10_l2.setText("2. Set as final Layer:");
        self.py_la10_l2.move(20, 150);
        tmp.append(self.py_la10_l2);

        self.py_la10_cb2 = QComboBox(self);
        self.py_la10_cb2.move(200, 150);
        self.py_la10_cb2.activated.connect(self.set_final_layer);
        self.py_la10_cb2.addItems(["No", "Yes"]);
        tmp.append(self.py_la10_cb2);

        self.layer_ui_pytorch.append(tmp)




        tmp = [];
        self.py_la11_l1 = QLabel(self);
        self.py_la11_l1.setText("1. Set as final layer:");
        self.py_la11_l1.move(20, 100);
        tmp.append(self.py_la11_l1);

        self.py_la11_cb1 = QComboBox(self);
        self.py_la11_cb1.move(200, 100);
        self.py_la11_cb1.activated.connect(self.set_final_layer);
        self.py_la11_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la11_cb1);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la12_l1 = QLabel(self);
        self.py_la12_l1.setText("1. Lambd param:");
        self.py_la12_l1.move(20, 100);
        tmp.append(self.py_la12_l1);

        self.py_la12_e1 = QLineEdit(self)
        self.py_la12_e1.move(180, 100);
        self.py_la12_e1.setText("0.5");
        tmp.append(self.py_la12_e1);

        self.py_la12_l2 = QLabel(self);
        self.py_la12_l2.setText("2. Set as final Layer:");
        self.py_la12_l2.move(20, 150);
        tmp.append(self.py_la12_l2);

        self.py_la12_cb2 = QComboBox(self);
        self.py_la12_cb2.move(200, 150);
        self.py_la12_cb2.activated.connect(self.set_final_layer);
        self.py_la12_cb2.addItems(["No", "Yes"]);
        tmp.append(self.py_la12_cb2);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la13_l1 = QLabel(self);
        self.py_la13_l1.setText("1. Beta:");
        self.py_la13_l1.move(20, 100);
        tmp.append(self.py_la13_l1);

        self.py_la13_e1 = QLineEdit(self)
        self.py_la13_e1.move(180, 100);
        self.py_la13_e1.setText("-1.0");
        tmp.append(self.py_la13_e1);


        self.py_la13_l2 = QLabel(self);
        self.py_la13_l2.setText("2. Threshold:");
        self.py_la13_l2.move(20, 150);
        tmp.append(self.py_la13_l2);

        self.py_la13_e2 = QLineEdit(self)
        self.py_la13_e2.move(180, 150);
        self.py_la13_e2.setText("1.0");
        tmp.append(self.py_la13_e2);


        self.py_la13_l3 = QLabel(self);
        self.py_la13_l3.setText("3. Set as final Layer:");
        self.py_la13_l3.move(20, 200);
        tmp.append(self.py_la13_l3);

        self.py_la13_cb3 = QComboBox(self);
        self.py_la13_cb3.move(200, 200);
        self.py_la13_cb3.activated.connect(self.set_final_layer);
        self.py_la13_cb3.addItems(["No", "Yes"]);
        tmp.append(self.py_la13_cb3);

        self.layer_ui_pytorch.append(tmp)




        tmp = [];
        self.py_la14_l1 = QLabel(self);
        self.py_la14_l1.setText("1. Set as final layer:");
        self.py_la14_l1.move(20, 100);
        tmp.append(self.py_la14_l1);

        self.py_la14_cb1 = QComboBox(self);
        self.py_la14_cb1.move(200, 100);
        self.py_la14_cb1.activated.connect(self.set_final_layer);
        self.py_la14_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la14_cb1);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la15_l1 = QLabel(self);
        self.py_la15_l1.setText("1. Set as final layer:");
        self.py_la15_l1.move(20, 100);
        tmp.append(self.py_la15_l1);

        self.py_la15_cb1 = QComboBox(self);
        self.py_la15_cb1.move(200, 100);
        self.py_la15_cb1.activated.connect(self.set_final_layer);
        self.py_la15_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la15_cb1);

        self.layer_ui_pytorch.append(tmp)



        tmp = [];
        self.py_la16_l1 = QLabel(self);
        self.py_la16_l1.setText("1. Lower limit:");
        self.py_la16_l1.move(20, 100);
        tmp.append(self.py_la16_l1);

        self.py_la16_e1 = QLineEdit(self)
        self.py_la16_e1.move(180, 100);
        self.py_la16_e1.setText("0.125");
        tmp.append(self.py_la16_e1);


        self.py_la16_l2 = QLabel(self);
        self.py_la16_l2.setText("2. Upper Limit:");
        self.py_la16_l2.move(20, 150);
        tmp.append(self.py_la16_l2);

        self.py_la16_e2 = QLineEdit(self)
        self.py_la16_e2.move(180, 150);
        self.py_la16_e2.setText("1.333");
        tmp.append(self.py_la16_e2);


        self.py_la16_l3 = QLabel(self);
        self.py_la16_l3.setText("3. Set as final Layer:");
        self.py_la16_l3.move(20, 200);
        tmp.append(self.py_la16_l3);

        self.py_la16_cb3 = QComboBox(self);
        self.py_la16_cb3.move(200, 200);
        self.py_la16_cb3.activated.connect(self.set_final_layer);
        self.py_la16_cb3.addItems(["No", "Yes"]);
        tmp.append(self.py_la16_cb3);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la17_l1 = QLabel(self);
        self.py_la17_l1.setText("1. Alpha:");
        self.py_la17_l1.move(20, 100);
        tmp.append(self.py_la17_l1);

        self.py_la17_e1 = QLineEdit(self)
        self.py_la17_e1.move(180, 100);
        self.py_la17_e1.setText("1.0");
        tmp.append(self.py_la17_e1);

        self.py_la17_l2 = QLabel(self);
        self.py_la17_l2.setText("2. Set as final Layer:");
        self.py_la17_l2.move(20, 150);
        tmp.append(self.py_la17_l2);

        self.py_la17_cb2 = QComboBox(self);
        self.py_la17_cb2.move(200, 150);
        self.py_la17_cb2.activated.connect(self.set_final_layer);
        self.py_la17_cb2.addItems(["No", "Yes"]);
        tmp.append(self.py_la17_cb2);

        self.layer_ui_pytorch.append(tmp)




        tmp = [];
        self.py_la18_l1 = QLabel(self);
        self.py_la18_l1.setText("1. Lambd param:");
        self.py_la18_l1.move(20, 100);
        tmp.append(self.py_la18_l1);

        self.py_la18_e1 = QLineEdit(self)
        self.py_la18_e1.move(180, 100);
        self.py_la18_e1.setText("0.5");
        tmp.append(self.py_la18_e1);

        self.py_la18_l2 = QLabel(self);
        self.py_la18_l2.setText("2. Set as final Layer:");
        self.py_la18_l2.move(20, 150);
        tmp.append(self.py_la18_l2);

        self.py_la18_cb2 = QComboBox(self);
        self.py_la18_cb2.move(200, 150);
        self.py_la18_cb2.activated.connect(self.set_final_layer);
        self.py_la18_cb2.addItems(["No", "Yes"]);
        tmp.append(self.py_la18_cb2);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la19_l1 = QLabel(self);
        self.py_la19_l1.setText("1. Set as final layer:");
        self.py_la19_l1.move(20, 100);
        tmp.append(self.py_la19_l1);

        self.py_la19_cb1 = QComboBox(self);
        self.py_la19_cb1.move(200, 100);
        self.py_la19_cb1.activated.connect(self.set_final_layer);
        self.py_la19_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la19_cb1);

        self.layer_ui_pytorch.append(tmp)




        tmp = [];
        self.py_la20_l1 = QLabel(self);
        self.py_la20_l1.setText("1. Threshold:");
        self.py_la20_l1.move(20, 100);
        tmp.append(self.py_la20_l1);

        self.py_la20_e1 = QLineEdit(self)
        self.py_la20_e1.move(180, 100);
        self.py_la20_e1.setText("None");
        tmp.append(self.py_la20_e1);


        self.py_la20_l2 = QLabel(self);
        self.py_la20_l2.setText("2. Replacement value:");
        self.py_la20_l2.move(20, 150);
        tmp.append(self.py_la20_l2);

        self.py_la20_e2 = QLineEdit(self)
        self.py_la20_e2.move(180, 150);
        self.py_la20_e2.setText("None");
        tmp.append(self.py_la20_e2);


        self.py_la20_l3 = QLabel(self);
        self.py_la20_l3.setText("3. Set as final Layer:");
        self.py_la20_l3.move(20, 200);
        tmp.append(self.py_la20_l3);

        self.py_la20_cb3 = QComboBox(self);
        self.py_la20_cb3.move(200, 200);
        self.py_la20_cb3.activated.connect(self.set_final_layer);
        self.py_la20_cb3.addItems(["No", "Yes"]);
        tmp.append(self.py_la20_cb3);

        self.layer_ui_pytorch.append(tmp)



        tmp = [];
        self.py_la21_l1 = QLabel(self);
        self.py_la21_l1.setText("1. Set as final layer:");
        self.py_la21_l1.move(20, 100);
        tmp.append(self.py_la21_l1);

        self.py_la21_cb1 = QComboBox(self);
        self.py_la21_cb1.move(200, 100);
        self.py_la21_cb1.activated.connect(self.set_final_layer);
        self.py_la21_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la21_cb1);

        self.layer_ui_pytorch.append(tmp)



        tmp = [];
        self.py_la22_l1 = QLabel(self);
        self.py_la22_l1.setText("1. Set as final layer:");
        self.py_la22_l1.move(20, 100);
        tmp.append(self.py_la22_l1);

        self.py_la22_cb1 = QComboBox(self);
        self.py_la22_cb1.move(200, 100);
        self.py_la22_cb1.activated.connect(self.set_final_layer);
        self.py_la22_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la22_cb1);

        self.layer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_la23_l1 = QLabel(self);
        self.py_la23_l1.setText("1. Set as final layer:");
        self.py_la23_l1.move(20, 100);
        tmp.append(self.py_la23_l1);

        self.py_la23_cb1 = QComboBox(self);
        self.py_la23_cb1.move(200, 100);
        self.py_la23_cb1.activated.connect(self.set_final_layer);
        self.py_la23_cb1.addItems(["No", "Yes"]);
        tmp.append(self.py_la23_cb1);

        self.layer_ui_pytorch.append(tmp)























        self.select_layer();

        self.tb1 = QTextEdit(self)
        self.tb1.move(550, 20)
        self.tb1.resize(300, 500)
        if(self.system["update"]["layers"]["active"]):
            wr = "";
            for i in range(len(self.system["update"]["layers"]["value"])):
                tmp = json.dumps(self.system["update"]["layers"]["value"][i], indent=4)
                wr += "{}\n".format(tmp);
            self.tb1.setText(wr);
        else:
            self.tb1.setText("Using Default fintuning layers. Add new layers to replace them if required.");


        self.b4 = QPushButton('Add Layer', self)
        self.b4.move(400,400)
        self.b4.clicked.connect(self.add_layer)

        
        self.b5 = QPushButton('Remove last layer', self)
        self.b5.move(370,450)
        self.b5.clicked.connect(self.remove_layer)

        
        self.b6 = QPushButton('Clear layers', self)
        self.b6.move(400,500)
        self.b6.clicked.connect(self.clear_layer)







    def set_final_layer(self):
        if(self.system["backend"] == "Mxnet-1.5.1"):
            if(self.mx_la1_cb2.currentText() == "Yes"):
                self.mx_la1_e1.setEnabled(False)
            else:
                self.mx_la1_e1.setEnabled(True)

            if(self.mx_la2_cb2.currentText() == "Yes"):
                self.mx_la2_e1.setEnabled(False)
            else:
                self.mx_la2_e1.setEnabled(True)

            if(self.mx_la6_cb3.currentText() == "Yes"):
                self.mx_la6_e1.setEnabled(False)
                self.mx_la6_e2.setEnabled(False)
            else:
                self.mx_la6_e1.setEnabled(True)
                self.mx_la6_e2.setEnabled(True)

            if(self.mx_la8_cb2.currentText() == "Yes"):
                self.mx_la8_e1.setEnabled(False)
            else:
                self.mx_la8_e1.setEnabled(True)

            if(self.mx_la9_cb2.currentText() == "Yes"):
                self.mx_la9_e1.setEnabled(False)
            else:
                self.mx_la9_e1.setEnabled(True)

            if(self.mx_la10_cb2.currentText() == "Yes"):
                self.mx_la10_e1.setEnabled(False)
            else:
                self.mx_la10_e1.setEnabled(True)

            if(self.mx_la12_cb2.currentText() == "Yes"):
                self.mx_la12_e1.setEnabled(False)
            else:
                self.mx_la12_e1.setEnabled(True)

        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            if(self.ke_la1_cb2.currentText() == "Yes"):
                self.ke_la1_e1.setEnabled(False)
            else:
                self.ke_la1_e1.setEnabled(True)

            if(self.ke_la2_cb2.currentText() == "Yes"):
                self.ke_la2_e1.setEnabled(False)
            else:
                self.ke_la2_e1.setEnabled(True)

            if(self.ke_la4_cb2.currentText() == "Yes"):
                self.ke_la4_e1.setEnabled(False)
            else:
                self.ke_la4_e1.setEnabled(True)

            if(self.ke_la5_cb2.currentText() == "Yes"):
                self.ke_la5_e1.setEnabled(False)
            else:
                self.ke_la5_e1.setEnabled(True)

            if(self.ke_la6_cb2.currentText() == "Yes"):
                self.ke_la6_e1.setEnabled(False)
            else:
                self.ke_la6_e1.setEnabled(True)

            if(self.ke_la7_cb3.currentText() == "Yes"):
                self.ke_la7_e1.setEnabled(False)
                self.ke_la7_e2.setEnabled(False)
            else:
                self.ke_la7_e1.setEnabled(True)
                self.ke_la7_e2.setEnabled(True)

            if(self.ke_la10_cb3.currentText() == "Yes"):
                self.ke_la10_e1.setEnabled(False)
                self.ke_la10_e2.setEnabled(False)
            else:
                self.ke_la10_e1.setEnabled(True)
                self.ke_la10_e2.setEnabled(True)

        elif(self.system["backend"] == "Pytorch-1.3.1"):
            if(self.py_la1_cb2.currentText() == "Yes"):
                self.py_la1_e1.setEnabled(False)
            else:
                self.py_la1_e1.setEnabled(True)

            if(self.py_la2_cb2.currentText() == "Yes"):
                self.py_la2_e1.setEnabled(False)
            else:
                self.py_la2_e1.setEnabled(True)

            if(self.py_la6_cb3.currentText() == "Yes"):
                self.py_la6_e1.setEnabled(False)
                self.py_la6_e2.setEnabled(False)
            else:
                self.py_la6_e1.setEnabled(True)
                self.py_la6_e2.setEnabled(True)

            if(self.py_la8_cb2.currentText() == "Yes"):
                self.py_la8_e1.setEnabled(False)
            else:
                self.py_la8_e1.setEnabled(True)

            if(self.py_la9_cb2.currentText() == "Yes"):
                self.py_la9_e1.setEnabled(False)
            else:
                self.py_la9_e1.setEnabled(True)

            if(self.py_la10_cb2.currentText() == "Yes"):
                self.py_la10_e1.setEnabled(False)
            else:
                self.py_la10_e1.setEnabled(True)

            if(self.py_la12_cb2.currentText() == "Yes"):
                self.py_la12_e1.setEnabled(False)
            else:
                self.py_la12_e1.setEnabled(True)

            if(self.py_la13_cb3.currentText() == "Yes"):
                self.py_la13_e1.setEnabled(False)
                self.py_la13_e2.setEnabled(False)
            else:
                self.py_la13_e1.setEnabled(True)
                self.py_la13_e2.setEnabled(True)

            if(self.py_la16_cb3.currentText() == "Yes"):
                self.py_la16_e1.setEnabled(False)
                self.py_la16_e2.setEnabled(False)
            else:
                self.py_la16_e1.setEnabled(True)
                self.py_la16_e2.setEnabled(True)

            if(self.py_la17_cb2.currentText() == "Yes"):
                self.py_la17_e1.setEnabled(False)
            else:
                self.py_la17_e1.setEnabled(True)

            if(self.py_la18_cb2.currentText() == "Yes"):
                self.py_la18_e1.setEnabled(False)
            else:
                self.py_la18_e1.setEnabled(True)

            if(self.py_la20_cb3.currentText() == "Yes"):
                self.py_la20_e1.setEnabled(False)
                self.py_la20_e2.setEnabled(False)
            else:
                self.py_la20_e1.setEnabled(True)
                self.py_la20_e2.setEnabled(True)



    def select_layer(self):
        self.current_layer = {};
        self.current_layer["name"] = "";
        self.current_layer["params"] = {};

        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.current_layer["name"] = self.cb1.currentText();
            index = self.mxnet_layers_list.index(self.cb1.currentText());
            for i in range(len(self.layer_ui_mxnet)):
                for j in range(len(self.layer_ui_mxnet[i])):
                    if((index-1)==i):
                        self.layer_ui_mxnet[i][j].show();
                    else:
                        self.layer_ui_mxnet[i][j].hide();

            for i in range(len(self.layer_ui_keras)):
                for j in range(len(self.layer_ui_keras[i])):
                    self.layer_ui_keras[i][j].hide();
            for i in range(len(self.layer_ui_pytorch)):
                for j in range(len(self.layer_ui_pytorch[i])):
                    self.layer_ui_pytorch[i][j].hide();
            


        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.current_layer["name"] = self.cb2.currentText();
            index = self.keras_layers_list.index(self.cb2.currentText());
            for i in range(len(self.layer_ui_keras)):
                for j in range(len(self.layer_ui_keras[i])):
                    if((index-1)==i):
                        self.layer_ui_keras[i][j].show();
                    else:
                        self.layer_ui_keras[i][j].hide();

            for i in range(len(self.layer_ui_mxnet)):
                for j in range(len(self.layer_ui_mxnet[i])):
                    self.layer_ui_mxnet[i][j].hide();
            for i in range(len(self.layer_ui_pytorch)):
                for j in range(len(self.layer_ui_pytorch[i])):
                    self.layer_ui_pytorch[i][j].hide();



        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.current_layer["name"] = self.cb3.currentText();
            index = self.pytorch_layers_list.index(self.cb3.currentText());
            for i in range(len(self.layer_ui_pytorch)):
                for j in range(len(self.layer_ui_pytorch[i])):
                    if((index-1)==i):
                        self.layer_ui_pytorch[i][j].show();
                    else:
                        self.layer_ui_pytorch[i][j].hide();

            for i in range(len(self.layer_ui_keras)):
                for j in range(len(self.layer_ui_keras[i])):
                    self.layer_ui_keras[i][j].hide();
            for i in range(len(self.layer_ui_mxnet)):
                for j in range(len(self.layer_ui_mxnet[i])):
                    self.layer_ui_mxnet[i][j].hide();



    def add_layer(self):
        self.system["update"]["layers"]["active"] = True;
        if(self.system["backend"] == "Mxnet-1.5.1"):
            if(self.current_layer["name"] == self.mxnet_layers_list[1]):
                if(self.current_layer["params"]["final"] == "Yes"):
                    self.current_layer["params"]["neurons"] = "-";
                    self.current_layer["params"]["final"] = self.mx_la1_cb2.currentText();
                    self.system["update"]["layers"]["value"].append(self.current_layer);
                else:
                    self.current_layer["params"]["neurons"] = self.mx_la1_e1.text();
                    self.current_layer["params"]["final"] = self.mx_la1_cb2.currentText();
                    self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[2]):
                self.current_layer["params"]["probability"] = self.mx_la2_e1.text();
                self.current_layer["params"]["final"] = self.mx_la2_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[3]):
                self.current_layer["params"]["final"] = self.mx_la3_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[4]):
                self.current_layer["params"]["final"] = self.mx_la4_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[5]):
                self.current_layer["params"]["final"] = self.mx_la5_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[6]):
                self.current_layer["params"]["beta"] = self.mx_la6_e1.text();
                self.current_layer["params"]["threshold"] = self.mx_la6_e2.text();
                self.current_layer["params"]["final"] = self.mx_la6_cb3.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[7]):
                self.current_layer["params"]["final"] = self.mx_la7_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[8]):
                self.current_layer["params"]["alpha"] = self.mx_la8_e1.text();
                self.current_layer["params"]["final"] = self.mx_la8_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[9]):
                self.current_layer["params"]["negative_slope"] = self.mx_la9_e1.text();
                self.current_layer["params"]["final"] = self.mx_la9_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[10]):
                self.current_layer["params"]["init"] = self.mx_la10_e1.text();
                self.current_layer["params"]["final"] = self.mx_la10_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[11]):
                self.current_layer["params"]["final"] = self.mx_la11_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.mxnet_layers_list[12]):
                self.current_layer["params"]["beta"] = self.mx_la12_e1.text();
                self.current_layer["params"]["final"] = self.mx_la12_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            if(self.current_layer["name"] == self.keras_layers_list[1]):
                self.current_layer["params"]["neurons"] = self.ke_la1_e1.text();
                self.current_layer["params"]["final"] = self.ke_la1_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[2]):
                self.current_layer["params"]["probability"] = self.ke_la2_e1.text();
                self.current_layer["params"]["final"] = self.ke_la2_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[3]):
                self.current_layer["params"]["final"] = self.ke_la3_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[4]):
                self.current_layer["params"]["alpha"] = self.ke_la4_e1.text();
                self.current_layer["params"]["final"] = self.ke_la4_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[5]):
                self.current_layer["params"]["negative_slope"] = self.ke_la5_e1.text();
                self.current_layer["params"]["final"] = self.ke_la5_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[6]):
                self.current_layer["params"]["init"] = self.ke_la6_e1.text();
                self.current_layer["params"]["final"] = self.ke_la6_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[7]):
                self.current_layer["params"]["threshold"] = self.ke_la7_e1.text();
                self.current_layer["params"]["value"] = self.ke_la7_e2.text();
                self.current_layer["params"]["final"] = self.ke_la7_cb3.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[8]):
                self.current_layer["params"]["final"] = self.ke_la8_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[9]):
                self.current_layer["params"]["final"] = self.ke_la9_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[10]):
                self.current_layer["params"]["beta"] = self.ke_la10_e1.text();
                self.current_layer["params"]["threshold"] = self.ke_la10_e2.text();
                self.current_layer["params"]["final"] = self.ke_la10_cb3.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[11]):
                self.current_layer["params"]["final"] = self.ke_la11_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[12]):
                self.current_layer["params"]["final"] = self.ke_la12_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.keras_layers_list[13]):
                self.current_layer["params"]["final"] = self.ke_la13_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

        elif(self.system["backend"] == "Pytorch-1.3.1"):
            if(self.current_layer["name"] == self.pytorch_layers_list[1]):
                self.current_layer["params"]["neurons"] = self.py_la1_e1.text();
                self.current_layer["params"]["final"] = self.py_la1_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[2]):
                self.current_layer["params"]["probability"] = self.py_la2_e1.text();
                self.current_layer["params"]["final"] = self.py_la2_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[3]):
                self.current_layer["params"]["final"] = self.py_la3_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[4]):
                self.current_layer["params"]["final"] = self.py_la4_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[5]):
                self.current_layer["params"]["final"] = self.py_la5_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[6]):
                self.current_layer["params"]["beta"] = self.py_la6_e1.text();
                self.current_layer["params"]["threshold"] = self.py_la6_e2.text();
                self.current_layer["params"]["final"] = self.py_la6_cb3.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[7]):
                self.current_layer["params"]["final"] = self.py_la7_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[8]):
                self.current_layer["params"]["alpha"] = self.py_la8_e1.text();
                self.current_layer["params"]["final"] = self.py_la8_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[9]):
                self.current_layer["params"]["negative_slope"] = self.py_la9_e1.text();
                self.current_layer["params"]["final"] = self.py_la9_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[10]):
                self.current_layer["params"]["init"] = self.py_la10_e1.text();
                self.current_layer["params"]["final"] = self.py_la10_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[11]):
                self.current_layer["params"]["final"] = self.py_la11_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[12]):
                self.current_layer["params"]["lambd"] = self.py_la12_e1.text();
                self.current_layer["params"]["final"] = self.py_la12_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[13]):
                self.current_layer["params"]["min_val"] = self.py_la13_e1.text();
                self.current_layer["params"]["max_Val"] = self.py_la13_e2.text();
                self.current_layer["params"]["final"] = self.py_la13_cb3.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[14]):
                self.current_layer["params"]["final"] = self.py_la14_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[15]):
                self.current_layer["params"]["final"] = self.py_la15_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[16]):
                self.current_layer["params"]["lower"] = self.py_la16_e1.text();
                self.current_layer["params"]["upper"] = self.py_la16_e2.text();
                self.current_layer["params"]["final"] = self.py_la16_cb3.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[17]):
                self.current_layer["params"]["alpha"] = self.py_la17_e1.text();
                self.current_layer["params"]["final"] = self.py_la17_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[18]):
                self.current_layer["params"]["lambd"] = self.py_la18_e1.text();
                self.current_layer["params"]["final"] = self.py_la18_cb2.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[19]):
                self.current_layer["params"]["final"] = self.py_la19_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[20]):
                self.current_layer["params"]["threshold"] = self.py_la20_e1.text();
                self.current_layer["params"]["value"] = self.py_la20_e2.text();
                self.current_layer["params"]["final"] = self.py_la20_cb3.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[21]):
                self.current_layer["params"]["final"] = self.py_la21_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[22]):
                self.current_layer["params"]["final"] = self.py_la22_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);

            elif(self.current_layer["name"] == self.pytorch_layers_list[23]):
                self.current_layer["params"]["final"] = self.py_la23_cb1.currentText();
                self.system["update"]["layers"]["value"].append(self.current_layer);








        wr = "";
        for i in range(len(self.system["update"]["layers"]["value"])):
            tmp = json.dumps(self.system["update"]["layers"]["value"][i], indent=4)
            wr += "{}\n".format(tmp);
        self.tb1.setText(wr);


    def remove_layer(self):
        if(len(self.system["update"]["layers"]["value"]) > 0):
            del self.system["update"]["layers"]["value"][-1]
        else:
            self.system["update"]["layers"]["active"] = False;

        wr = "";
        for i in range(len(self.system["update"]["layers"]["value"])):
            tmp = json.dumps(self.system["update"]["layers"]["value"][i], indent=4)
            wr += "{}\n".format(tmp);
        self.tb1.setText(wr);

        if(len(self.system["update"]["layers"]["value"]) == 0):
            self.system["update"]["layers"]["active"] = False;


    def clear_layer(self):
        self.system["update"]["layers"]["value"] = [];
        self.system["update"]["layers"]["active"] = False;

        wr = "";
        for i in range(len(self.system["update"]["layers"]["value"])):
            tmp = json.dumps(self.system["update"]["layers"]["value"][i], indent=4)
            wr += "{}\n".format(tmp);
        self.tb1.setText(wr);




    def forward(self):        
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_train_param.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_model_param.emit();



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateLayerParam()
screen.show()
sys.exit(app.exec_())
'''