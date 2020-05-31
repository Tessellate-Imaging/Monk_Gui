import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateLossParam(QtWidgets.QWidget):

    forward_train = QtCore.pyqtSignal();
    backward_scheduler_param = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Update Loss Params'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 600
        self.loss_ui_mxnet = [];
        self.loss_ui_keras = [];
        self.loss_ui_pytorch = [];
        self.current_loss_ = {};
        self.current_loss_["name"] = "";
        self.current_loss_["params"] = {};
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
        self.cb1.activated.connect(self.select_loss);

        self.cb2 = QComboBox(self);
        self.cb2.move(20, 20);
        self.cb2.activated.connect(self.select_loss);

        self.cb3 = QComboBox(self);
        self.cb3.move(20, 20);
        self.cb3.activated.connect(self.select_loss);


        self.mxnet_losses_list = ["select", "loss_l1", "loss_l2", "loss_softmax_crossentropy", "loss_crossentropy",
                                    "loss_sigmoid_binary_crossentropy", "loss_binary_crossentropy",
                                    "loss_kldiv", "loss_poisson_nll", "loss_huber", "loss_hinge",
                                    "loss_squared_hinge"];

        self.keras_losses_list = ["select", "loss_l1", "loss_l2", "loss_crossentropy", "loss_binary_crossentropy", 
                                    "loss_kldiv", "loss_hinge", "loss_squared_hinge"];

        self.pytorch_losses_list = ["select", "loss_l1", "loss_l2", "loss_softmax_crossentropy", "loss_crossentropy",
                                    "loss_sigmoid_binary_crossentropy", "loss_binary_crossentropy",
                                    "loss_kldiv", "loss_poisson_nll", "loss_huber", "loss_hinge",
                                    "loss_squared_hinge", "loss_multimargin", "loss_squared_multimargin",
                                    "loss_multilabel_margin", "loss_multilabel_softmargin"];


        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.cb1.addItems(self.mxnet_losses_list);
            self.cb1.show();
            self.cb2.hide();
            self.cb3.hide();
        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.cb2.addItems(self.keras_losses_list);
            self.cb2.show();
            self.cb1.hide();
            self.cb3.hide();
        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.cb3.addItems(self.pytorch_losses_list);
            self.cb3.show();
            self.cb1.hide();
            self.cb2.hide();




        tmp = [];
        self.mx_lo1_l1 = QLabel(self);
        self.mx_lo1_l1.setText("1. Scalar Weight: ");
        self.mx_lo1_l1.move(20, 100);
        tmp.append(self.mx_lo1_l1);

        self.mx_lo1_e1 = QLineEdit(self)
        self.mx_lo1_e1.move(150, 100);
        self.mx_lo1_e1.setText("1.0");
        tmp.append(self.mx_lo1_e1);

        self.mx_lo1_l2 = QLabel(self);
        self.mx_lo1_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo1_l2.move(20, 150);
        tmp.append(self.mx_lo1_l2);

        self.mx_lo1_e2 = QLineEdit(self)
        self.mx_lo1_e2.move(290, 150);
        self.mx_lo1_e2.setText("0");
        tmp.append(self.mx_lo1_e2);

        self.loss_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_lo2_l1 = QLabel(self);
        self.mx_lo2_l1.setText("1. Scalar Weight: ");
        self.mx_lo2_l1.move(20, 100);
        tmp.append(self.mx_lo2_l1);

        self.mx_lo2_e1 = QLineEdit(self)
        self.mx_lo2_e1.move(150, 100);
        self.mx_lo2_e1.setText("1.0");
        tmp.append(self.mx_lo2_e1);

        self.mx_lo2_l2 = QLabel(self);
        self.mx_lo2_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo2_l2.move(20, 150);
        tmp.append(self.mx_lo2_l2);

        self.mx_lo2_e2 = QLineEdit(self)
        self.mx_lo2_e2.move(290, 150);
        self.mx_lo2_e2.setText("0");
        tmp.append(self.mx_lo2_e2);

        self.loss_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_lo3_l1 = QLabel(self);
        self.mx_lo3_l1.setText("1. Scalar Weight: ");
        self.mx_lo3_l1.move(20, 100);
        tmp.append(self.mx_lo3_l1);

        self.mx_lo3_e1 = QLineEdit(self)
        self.mx_lo3_e1.move(150, 100);
        self.mx_lo3_e1.setText("1.0");
        tmp.append(self.mx_lo3_e1);

        self.mx_lo3_l2 = QLabel(self);
        self.mx_lo3_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo3_l2.move(20, 150);
        tmp.append(self.mx_lo3_l2);

        self.mx_lo3_e2 = QLineEdit(self)
        self.mx_lo3_e2.move(290, 150);
        self.mx_lo3_e2.setText("0");
        tmp.append(self.mx_lo3_e2);

        self.loss_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_lo4_l1 = QLabel(self);
        self.mx_lo4_l1.setText("1. Scalar Weight: ");
        self.mx_lo4_l1.move(20, 100);
        tmp.append(self.mx_lo4_l1);

        self.mx_lo4_e1 = QLineEdit(self)
        self.mx_lo4_e1.move(150, 100);
        self.mx_lo4_e1.setText("1.0");
        tmp.append(self.mx_lo4_e1);

        self.mx_lo4_l2 = QLabel(self);
        self.mx_lo4_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo4_l2.move(20, 150);
        tmp.append(self.mx_lo4_l2);

        self.mx_lo4_e2 = QLineEdit(self)
        self.mx_lo4_e2.move(290, 150);
        self.mx_lo4_e2.setText("0");
        tmp.append(self.mx_lo4_e2);

        self.loss_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_lo5_l1 = QLabel(self);
        self.mx_lo5_l1.setText("1. Scalar Weight: ");
        self.mx_lo5_l1.move(20, 100);
        tmp.append(self.mx_lo5_l1);

        self.mx_lo5_e1 = QLineEdit(self)
        self.mx_lo5_e1.move(150, 100);
        self.mx_lo5_e1.setText("1.0");
        tmp.append(self.mx_lo5_e1);

        self.mx_lo5_l2 = QLabel(self);
        self.mx_lo5_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo5_l2.move(20, 150);
        tmp.append(self.mx_lo5_l2);

        self.mx_lo5_e2 = QLineEdit(self)
        self.mx_lo5_e2.move(290, 150);
        self.mx_lo5_e2.setText("0");
        tmp.append(self.mx_lo5_e2);

        self.loss_ui_mxnet.append(tmp)



        tmp = [];
        self.mx_lo6_l1 = QLabel(self);
        self.mx_lo6_l1.setText("1. Scalar Weight: ");
        self.mx_lo6_l1.move(20, 100);
        tmp.append(self.mx_lo6_l1);

        self.mx_lo6_e1 = QLineEdit(self)
        self.mx_lo6_e1.move(150, 100);
        self.mx_lo6_e1.setText("1.0");
        tmp.append(self.mx_lo6_e1);

        self.mx_lo6_l2 = QLabel(self);
        self.mx_lo6_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo6_l2.move(20, 150);
        tmp.append(self.mx_lo6_l2);

        self.mx_lo6_e2 = QLineEdit(self)
        self.mx_lo6_e2.move(290, 150);
        self.mx_lo6_e2.setText("0");
        tmp.append(self.mx_lo6_e2);

        self.loss_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_lo7_l1 = QLabel(self);
        self.mx_lo7_l1.setText("1. Scalar Weight: ");
        self.mx_lo7_l1.move(20, 100);
        tmp.append(self.mx_lo7_l1);

        self.mx_lo7_e1 = QLineEdit(self)
        self.mx_lo7_e1.move(150, 100);
        self.mx_lo7_e1.setText("1.0");
        tmp.append(self.mx_lo7_e1);

        self.mx_lo7_l2 = QLabel(self);
        self.mx_lo7_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo7_l2.move(20, 150);
        tmp.append(self.mx_lo7_l2);

        self.mx_lo7_e2 = QLineEdit(self)
        self.mx_lo7_e2.move(290, 150);
        self.mx_lo7_e2.setText("0");
        tmp.append(self.mx_lo7_e2);

        self.mx_lo7_l3 = QLabel(self);
        self.mx_lo7_l3.setText("3. Input has log pre-applied: ");
        self.mx_lo7_l3.move(20, 200);
        tmp.append(self.mx_lo7_l3);

        self.mx_lo7_cb3 = QComboBox(self);
        self.mx_lo7_cb3.move(290, 200);
        self.mx_lo7_cb3.addItems(["No", "Yes"]);
        tmp.append(self.mx_lo7_cb3);

        self.loss_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_lo8_l1 = QLabel(self);
        self.mx_lo8_l1.setText("1. Scalar Weight: ");
        self.mx_lo8_l1.move(20, 100);
        tmp.append(self.mx_lo8_l1);

        self.mx_lo8_e1 = QLineEdit(self)
        self.mx_lo8_e1.move(150, 100);
        self.mx_lo8_e1.setText("1.0");
        tmp.append(self.mx_lo8_e1);

        self.mx_lo8_l2 = QLabel(self);
        self.mx_lo8_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo8_l2.move(20, 150);
        tmp.append(self.mx_lo8_l2);

        self.mx_lo8_e2 = QLineEdit(self)
        self.mx_lo8_e2.move(290, 150);
        self.mx_lo8_e2.setText("0");
        tmp.append(self.mx_lo8_e2);

        self.mx_lo8_l3 = QLabel(self);
        self.mx_lo8_l3.setText("3. Input has log pre-applied: ");
        self.mx_lo8_l3.move(20, 200);
        tmp.append(self.mx_lo8_l3);

        self.mx_lo8_cb3 = QComboBox(self);
        self.mx_lo8_cb3.move(290, 200);
        self.mx_lo8_cb3.addItems(["No", "Yes"]);
        tmp.append(self.mx_lo8_cb3);

        self.loss_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_lo9_l1 = QLabel(self);
        self.mx_lo9_l1.setText("1. Scalar Weight: ");
        self.mx_lo9_l1.move(20, 100);
        tmp.append(self.mx_lo9_l1);

        self.mx_lo9_e1 = QLineEdit(self)
        self.mx_lo9_e1.move(150, 100);
        self.mx_lo9_e1.setText("1.0");
        tmp.append(self.mx_lo9_e1);

        self.mx_lo9_l2 = QLabel(self);
        self.mx_lo9_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo9_l2.move(20, 150);
        tmp.append(self.mx_lo9_l2);

        self.mx_lo9_e2 = QLineEdit(self)
        self.mx_lo9_e2.move(290, 150);
        self.mx_lo9_e2.setText("0");
        tmp.append(self.mx_lo9_e2);

        self.mx_lo9_l3 = QLabel(self);
        self.mx_lo9_l3.setText("3. Threshold for mean estimator: ");
        self.mx_lo9_l3.move(20, 200);
        tmp.append(self.mx_lo9_l3);

        self.mx_lo9_e3 = QLineEdit(self)
        self.mx_lo9_e3.move(290, 200);
        self.mx_lo9_e3.setText("1.0");
        tmp.append(self.mx_lo9_e3);

        self.loss_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_lo10_l1 = QLabel(self);
        self.mx_lo10_l1.setText("1. Scalar Weight: ");
        self.mx_lo10_l1.move(20, 100);
        tmp.append(self.mx_lo10_l1);

        self.mx_lo10_e1 = QLineEdit(self)
        self.mx_lo10_e1.move(150, 100);
        self.mx_lo10_e1.setText("1.0");
        tmp.append(self.mx_lo10_e1);

        self.mx_lo10_l2 = QLabel(self);
        self.mx_lo10_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo10_l2.move(20, 150);
        tmp.append(self.mx_lo10_l2);

        self.mx_lo10_e2 = QLineEdit(self)
        self.mx_lo10_e2.move(290, 150);
        self.mx_lo10_e2.setText("0");
        tmp.append(self.mx_lo10_e2);

        self.mx_lo10_l3 = QLabel(self);
        self.mx_lo10_l3.setText("3. Margin: ");
        self.mx_lo10_l3.move(20, 200);
        tmp.append(self.mx_lo10_l3);

        self.mx_lo10_e3 = QLineEdit(self)
        self.mx_lo10_e3.move(150, 200);
        self.mx_lo10_e3.setText("1.0");
        tmp.append(self.mx_lo10_e3);

        self.loss_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_lo11_l1 = QLabel(self);
        self.mx_lo11_l1.setText("1. Scalar Weight: ");
        self.mx_lo11_l1.move(20, 100);
        tmp.append(self.mx_lo11_l1);

        self.mx_lo11_e1 = QLineEdit(self)
        self.mx_lo11_e1.move(150, 100);
        self.mx_lo11_e1.setText("1.0");
        tmp.append(self.mx_lo11_e1);

        self.mx_lo11_l2 = QLabel(self);
        self.mx_lo11_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.mx_lo11_l2.move(20, 150);
        tmp.append(self.mx_lo11_l2);

        self.mx_lo11_e2 = QLineEdit(self)
        self.mx_lo11_e2.move(290, 150);
        self.mx_lo11_e2.setText("0");
        tmp.append(self.mx_lo11_e2);

        self.mx_lo11_l3 = QLabel(self);
        self.mx_lo11_l3.setText("3. Margin: ");
        self.mx_lo11_l3.move(20, 200);
        tmp.append(self.mx_lo11_l3);

        self.mx_lo11_e3 = QLineEdit(self)
        self.mx_lo11_e3.move(150, 200);
        self.mx_lo11_e3.setText("1.0");
        tmp.append(self.mx_lo11_e3);

        self.loss_ui_mxnet.append(tmp)






        tmp = [];
        self.ke_lo1_l1 = QLabel(self);
        self.ke_lo1_l1.setText("1. Scalar Weight: ");
        self.ke_lo1_l1.move(20, 100);
        tmp.append(self.ke_lo1_l1);

        self.ke_lo1_e1 = QLineEdit(self)
        self.ke_lo1_e1.move(150, 100);
        self.ke_lo1_e1.setText("1.0");
        tmp.append(self.ke_lo1_e1);

        self.ke_lo1_l2 = QLabel(self);
        self.ke_lo1_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.ke_lo1_l2.move(20, 150);
        tmp.append(self.ke_lo1_l2);

        self.ke_lo1_e2 = QLineEdit(self)
        self.ke_lo1_e2.move(290, 150);
        self.ke_lo1_e2.setText("0");
        tmp.append(self.ke_lo1_e2);

        self.loss_ui_keras.append(tmp)





        tmp = [];
        self.ke_lo2_l1 = QLabel(self);
        self.ke_lo2_l1.setText("1. Scalar Weight: ");
        self.ke_lo2_l1.move(20, 100);
        tmp.append(self.ke_lo2_l1);

        self.ke_lo2_e1 = QLineEdit(self)
        self.ke_lo2_e1.move(150, 100);
        self.ke_lo2_e1.setText("1.0");
        tmp.append(self.ke_lo2_e1);

        self.ke_lo2_l2 = QLabel(self);
        self.ke_lo2_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.ke_lo2_l2.move(20, 150);
        tmp.append(self.ke_lo2_l2);

        self.ke_lo2_e2 = QLineEdit(self)
        self.ke_lo2_e2.move(290, 150);
        self.ke_lo2_e2.setText("0");
        tmp.append(self.ke_lo2_e2);

        self.loss_ui_keras.append(tmp)




        tmp = [];
        self.ke_lo3_l1 = QLabel(self);
        self.ke_lo3_l1.setText("1. Scalar Weight: ");
        self.ke_lo3_l1.move(20, 100);
        tmp.append(self.ke_lo3_l1);

        self.ke_lo3_e1 = QLineEdit(self)
        self.ke_lo3_e1.move(150, 100);
        self.ke_lo3_e1.setText("1.0");
        tmp.append(self.ke_lo3_e1);

        self.ke_lo3_l2 = QLabel(self);
        self.ke_lo3_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.ke_lo3_l2.move(20, 150);
        tmp.append(self.ke_lo3_l2);

        self.ke_lo3_e2 = QLineEdit(self)
        self.ke_lo3_e2.move(290, 150);
        self.ke_lo3_e2.setText("0");
        tmp.append(self.ke_lo3_e2);

        self.loss_ui_keras.append(tmp)




        tmp = [];
        self.ke_lo4_l1 = QLabel(self);
        self.ke_lo4_l1.setText("1. Scalar Weight: ");
        self.ke_lo4_l1.move(20, 100);
        tmp.append(self.ke_lo4_l1);

        self.ke_lo4_e1 = QLineEdit(self)
        self.ke_lo4_e1.move(150, 100);
        self.ke_lo4_e1.setText("1.0");
        tmp.append(self.ke_lo4_e1);

        self.ke_lo4_l2 = QLabel(self);
        self.ke_lo4_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.ke_lo4_l2.move(20, 150);
        tmp.append(self.ke_lo4_l2);

        self.ke_lo4_e2 = QLineEdit(self)
        self.ke_lo4_e2.move(290, 150);
        self.ke_lo4_e2.setText("0");
        tmp.append(self.ke_lo4_e2);

        self.loss_ui_keras.append(tmp)






        tmp = [];
        self.ke_lo5_l1 = QLabel(self);
        self.ke_lo5_l1.setText("1. Scalar Weight: ");
        self.ke_lo5_l1.move(20, 100);
        tmp.append(self.ke_lo5_l1);

        self.ke_lo5_e1 = QLineEdit(self)
        self.ke_lo5_e1.move(150, 100);
        self.ke_lo5_e1.setText("1.0");
        tmp.append(self.ke_lo5_e1);

        self.ke_lo5_l2 = QLabel(self);
        self.ke_lo5_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.ke_lo5_l2.move(20, 150);
        tmp.append(self.ke_lo5_l2);

        self.ke_lo5_e2 = QLineEdit(self)
        self.ke_lo5_e2.move(290, 150);
        self.ke_lo5_e2.setText("0");
        tmp.append(self.ke_lo5_e2);

        self.ke_lo5_l3 = QLabel(self);
        self.ke_lo5_l3.setText("3. Input has log pre-applied: ");
        self.ke_lo5_l3.move(20, 200);
        tmp.append(self.ke_lo5_l3);

        self.ke_lo5_cb3 = QComboBox(self);
        self.ke_lo5_cb3.move(290, 200);
        self.ke_lo5_cb3.addItems(["No", "Yes"]);
        tmp.append(self.ke_lo5_cb3);

        self.loss_ui_keras.append(tmp)





        tmp = [];
        self.ke_lo6_l1 = QLabel(self);
        self.ke_lo6_l1.setText("1. Scalar Weight: ");
        self.ke_lo6_l1.move(20, 100);
        tmp.append(self.ke_lo6_l1);

        self.ke_lo6_e1 = QLineEdit(self)
        self.ke_lo6_e1.move(150, 100);
        self.ke_lo6_e1.setText("1.0");
        tmp.append(self.ke_lo6_e1);

        self.ke_lo6_l2 = QLabel(self);
        self.ke_lo6_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.ke_lo6_l2.move(20, 150);
        tmp.append(self.ke_lo6_l2);

        self.ke_lo6_e2 = QLineEdit(self)
        self.ke_lo6_e2.move(290, 150);
        self.ke_lo6_e2.setText("0");
        tmp.append(self.ke_lo6_e2);

        self.ke_lo6_l3 = QLabel(self);
        self.ke_lo6_l3.setText("3. Margin: ");
        self.ke_lo6_l3.move(20, 200);
        tmp.append(self.ke_lo6_l3);

        self.ke_lo6_e3 = QLineEdit(self)
        self.ke_lo6_e3.move(150, 200);
        self.ke_lo6_e3.setText("1.0");
        tmp.append(self.ke_lo6_e3);

        self.loss_ui_keras.append(tmp)




        tmp = [];
        self.ke_lo7_l1 = QLabel(self);
        self.ke_lo7_l1.setText("1. Scalar Weight: ");
        self.ke_lo7_l1.move(20, 100);
        tmp.append(self.ke_lo7_l1);

        self.ke_lo7_e1 = QLineEdit(self)
        self.ke_lo7_e1.move(150, 100);
        self.ke_lo7_e1.setText("1.0");
        tmp.append(self.ke_lo7_e1);

        self.ke_lo7_l2 = QLabel(self);
        self.ke_lo7_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.ke_lo7_l2.move(20, 150);
        tmp.append(self.ke_lo7_l2);

        self.ke_lo7_e2 = QLineEdit(self)
        self.ke_lo7_e2.move(290, 150);
        self.ke_lo7_e2.setText("0");
        tmp.append(self.ke_lo7_e2);

        self.ke_lo7_l3 = QLabel(self);
        self.ke_lo7_l3.setText("3. Margin: ");
        self.ke_lo7_l3.move(20, 200);
        tmp.append(self.ke_lo7_l3);

        self.ke_lo7_e3 = QLineEdit(self)
        self.ke_lo7_e3.move(150, 200);
        self.ke_lo7_e3.setText("1.0");
        tmp.append(self.ke_lo7_e3);

        self.loss_ui_keras.append(tmp)





        tmp = [];
        self.py_lo1_l1 = QLabel(self);
        self.py_lo1_l1.setText("1. Scalar Weight: ");
        self.py_lo1_l1.move(20, 100);
        tmp.append(self.py_lo1_l1);

        self.py_lo1_e1 = QLineEdit(self)
        self.py_lo1_e1.move(150, 100);
        self.py_lo1_e1.setText("1.0");
        tmp.append(self.py_lo1_e1);

        self.py_lo1_l2 = QLabel(self);
        self.py_lo1_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo1_l2.move(20, 150);
        tmp.append(self.py_lo1_l2);

        self.py_lo1_e2 = QLineEdit(self)
        self.py_lo1_e2.move(290, 150);
        self.py_lo1_e2.setText("0");
        tmp.append(self.py_lo1_e2);

        self.loss_ui_pytorch.append(tmp)




        tmp = [];
        self.py_lo2_l1 = QLabel(self);
        self.py_lo2_l1.setText("1. Scalar Weight: ");
        self.py_lo2_l1.move(20, 100);
        tmp.append(self.py_lo2_l1);

        self.py_lo2_e1 = QLineEdit(self)
        self.py_lo2_e1.move(150, 100);
        self.py_lo2_e1.setText("1.0");
        tmp.append(self.py_lo2_e1);

        self.py_lo2_l2 = QLabel(self);
        self.py_lo2_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo2_l2.move(20, 150);
        tmp.append(self.py_lo2_l2);

        self.py_lo2_e2 = QLineEdit(self)
        self.py_lo2_e2.move(290, 150);
        self.py_lo2_e2.setText("0");
        tmp.append(self.py_lo2_e2);

        self.loss_ui_pytorch.append(tmp)





        tmp = [];
        self.py_lo3_l1 = QLabel(self);
        self.py_lo3_l1.setText("1. Scalar Weight: ");
        self.py_lo3_l1.move(20, 100);
        tmp.append(self.py_lo3_l1);

        self.py_lo3_e1 = QLineEdit(self)
        self.py_lo3_e1.move(150, 100);
        self.py_lo3_e1.setText("1.0");
        tmp.append(self.py_lo3_e1);

        self.py_lo3_l2 = QLabel(self);
        self.py_lo3_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo3_l2.move(20, 150);
        tmp.append(self.py_lo3_l2);

        self.py_lo3_e2 = QLineEdit(self)
        self.py_lo3_e2.move(290, 150);
        self.py_lo3_e2.setText("0");
        tmp.append(self.py_lo3_e2);

        self.loss_ui_pytorch.append(tmp)




        tmp = [];
        self.py_lo4_l1 = QLabel(self);
        self.py_lo4_l1.setText("1. Scalar Weight: ");
        self.py_lo4_l1.move(20, 100);
        tmp.append(self.py_lo4_l1);

        self.py_lo4_e1 = QLineEdit(self)
        self.py_lo4_e1.move(150, 100);
        self.py_lo4_e1.setText("1.0");
        tmp.append(self.py_lo4_e1);

        self.py_lo4_l2 = QLabel(self);
        self.py_lo4_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo4_l2.move(20, 150);
        tmp.append(self.py_lo4_l2);

        self.py_lo4_e2 = QLineEdit(self)
        self.py_lo4_e2.move(290, 150);
        self.py_lo4_e2.setText("0");
        tmp.append(self.py_lo4_e2);

        self.loss_ui_pytorch.append(tmp)





        tmp = [];
        self.py_lo5_l1 = QLabel(self);
        self.py_lo5_l1.setText("1. Scalar Weight: ");
        self.py_lo5_l1.move(20, 100);
        tmp.append(self.py_lo5_l1);

        self.py_lo5_e1 = QLineEdit(self)
        self.py_lo5_e1.move(150, 100);
        self.py_lo5_e1.setText("1.0");
        tmp.append(self.py_lo5_e1);

        self.py_lo5_l2 = QLabel(self);
        self.py_lo5_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo5_l2.move(20, 150);
        tmp.append(self.py_lo5_l2);

        self.py_lo5_e2 = QLineEdit(self)
        self.py_lo5_e2.move(290, 150);
        self.py_lo5_e2.setText("0");
        tmp.append(self.py_lo5_e2);

        self.loss_ui_pytorch.append(tmp)





        tmp = [];
        self.py_lo6_l1 = QLabel(self);
        self.py_lo6_l1.setText("1. Scalar Weight: ");
        self.py_lo6_l1.move(20, 100);
        tmp.append(self.py_lo6_l1);

        self.py_lo6_e1 = QLineEdit(self)
        self.py_lo6_e1.move(150, 100);
        self.py_lo6_e1.setText("1.0");
        tmp.append(self.py_lo6_e1);

        self.py_lo6_l2 = QLabel(self);
        self.py_lo6_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo6_l2.move(20, 150);
        tmp.append(self.py_lo6_l2);

        self.py_lo6_e2 = QLineEdit(self)
        self.py_lo6_e2.move(290, 150);
        self.py_lo6_e2.setText("0");
        tmp.append(self.py_lo6_e2);

        self.loss_ui_pytorch.append(tmp)





        tmp = [];
        self.py_lo7_l1 = QLabel(self);
        self.py_lo7_l1.setText("1. Scalar Weight: ");
        self.py_lo7_l1.move(20, 100);
        tmp.append(self.py_lo7_l1);

        self.py_lo7_e1 = QLineEdit(self)
        self.py_lo7_e1.move(150, 100);
        self.py_lo7_e1.setText("1.0");
        tmp.append(self.py_lo7_e1);

        self.py_lo7_l2 = QLabel(self);
        self.py_lo7_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo7_l2.move(20, 150);
        tmp.append(self.py_lo7_l2);

        self.py_lo7_e2 = QLineEdit(self)
        self.py_lo7_e2.move(290, 150);
        self.py_lo7_e2.setText("0");
        tmp.append(self.py_lo7_e2);

        self.py_lo7_l3 = QLabel(self);
        self.py_lo7_l3.setText("3. Input has log pre-applied: ");
        self.py_lo7_l3.move(20, 200);
        tmp.append(self.py_lo7_l3);

        self.py_lo7_cb3 = QComboBox(self);
        self.py_lo7_cb3.move(290, 200);
        self.py_lo7_cb3.addItems(["No", "Yes"]);
        tmp.append(self.py_lo7_cb3);

        self.loss_ui_pytorch.append(tmp)





        tmp = [];
        self.py_lo8_l1 = QLabel(self);
        self.py_lo8_l1.setText("1. Scalar Weight: ");
        self.py_lo8_l1.move(20, 100);
        tmp.append(self.py_lo8_l1);

        self.py_lo8_e1 = QLineEdit(self)
        self.py_lo8_e1.move(150, 100);
        self.py_lo8_e1.setText("1.0");
        tmp.append(self.py_lo8_e1);

        self.py_lo8_l2 = QLabel(self);
        self.py_lo8_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo8_l2.move(20, 150);
        tmp.append(self.py_lo8_l2);

        self.py_lo8_e2 = QLineEdit(self)
        self.py_lo8_e2.move(290, 150);
        self.py_lo8_e2.setText("0");
        tmp.append(self.py_lo8_e2);

        self.py_lo8_l3 = QLabel(self);
        self.py_lo8_l3.setText("3. Input has log pre-applied: ");
        self.py_lo8_l3.move(20, 200);
        tmp.append(self.py_lo8_l3);

        self.py_lo8_cb3 = QComboBox(self);
        self.py_lo8_cb3.move(290, 200);
        self.py_lo8_cb3.addItems(["No", "Yes"]);
        tmp.append(self.py_lo8_cb3);

        self.loss_ui_pytorch.append(tmp)





        tmp = [];
        self.py_lo9_l1 = QLabel(self);
        self.py_lo9_l1.setText("1. Scalar Weight: ");
        self.py_lo9_l1.move(20, 100);
        tmp.append(self.py_lo9_l1);

        self.py_lo9_e1 = QLineEdit(self)
        self.py_lo9_e1.move(150, 100);
        self.py_lo9_e1.setText("1.0");
        tmp.append(self.py_lo9_e1);

        self.py_lo9_l2 = QLabel(self);
        self.py_lo9_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo9_l2.move(20, 150);
        tmp.append(self.py_lo9_l2);

        self.py_lo9_e2 = QLineEdit(self)
        self.py_lo9_e2.move(290, 150);
        self.py_lo9_e2.setText("0");
        tmp.append(self.py_lo9_e2);

        self.py_lo9_l3 = QLabel(self);
        self.py_lo9_l3.setText("3. Threshold for mean estimator: ");
        self.py_lo9_l3.move(20, 200);
        tmp.append(self.py_lo9_l3);

        self.py_lo9_e3 = QLineEdit(self)
        self.py_lo9_e3.move(290, 200);
        self.py_lo9_e3.setText("1.0");
        tmp.append(self.py_lo9_e3);

        self.loss_ui_pytorch.append(tmp)




        tmp = [];
        self.py_lo10_l1 = QLabel(self);
        self.py_lo10_l1.setText("1. Scalar Weight: ");
        self.py_lo10_l1.move(20, 100);
        tmp.append(self.py_lo10_l1);

        self.py_lo10_e1 = QLineEdit(self)
        self.py_lo10_e1.move(150, 100);
        self.py_lo10_e1.setText("1.0");
        tmp.append(self.py_lo10_e1);

        self.py_lo10_l2 = QLabel(self);
        self.py_lo10_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo10_l2.move(20, 150);
        tmp.append(self.py_lo10_l2);

        self.py_lo10_e2 = QLineEdit(self)
        self.py_lo10_e2.move(290, 150);
        self.py_lo10_e2.setText("0");
        tmp.append(self.py_lo10_e2);

        self.py_lo10_l3 = QLabel(self);
        self.py_lo10_l3.setText("3. Margin: ");
        self.py_lo10_l3.move(20, 200);
        tmp.append(self.py_lo10_l3);

        self.py_lo10_e3 = QLineEdit(self)
        self.py_lo10_e3.move(150, 200);
        self.py_lo10_e3.setText("1.0");
        tmp.append(self.py_lo10_e3);

        self.loss_ui_pytorch.append(tmp)




        tmp = [];
        self.py_lo11_l1 = QLabel(self);
        self.py_lo11_l1.setText("1. Scalar Weight: ");
        self.py_lo11_l1.move(20, 100);
        tmp.append(self.py_lo11_l1);

        self.py_lo11_e1 = QLineEdit(self)
        self.py_lo11_e1.move(150, 100);
        self.py_lo11_e1.setText("1.0");
        tmp.append(self.py_lo11_e1);

        self.py_lo11_l2 = QLabel(self);
        self.py_lo11_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo11_l2.move(20, 150);
        tmp.append(self.py_lo11_l2);

        self.py_lo11_e2 = QLineEdit(self)
        self.py_lo11_e2.move(290, 150);
        self.py_lo11_e2.setText("0");
        tmp.append(self.py_lo11_e2);

        self.py_lo11_l3 = QLabel(self);
        self.py_lo11_l3.setText("3. Margin: ");
        self.py_lo11_l3.move(20, 200);
        tmp.append(self.py_lo11_l3);

        self.py_lo11_e3 = QLineEdit(self)
        self.py_lo11_e3.move(150, 200);
        self.py_lo11_e3.setText("1.0");
        tmp.append(self.py_lo11_e3);

        self.loss_ui_pytorch.append(tmp)





        tmp = [];
        self.py_lo12_l1 = QLabel(self);
        self.py_lo12_l1.setText("1. Scalar Weight: ");
        self.py_lo12_l1.move(20, 100);
        tmp.append(self.py_lo12_l1);

        self.py_lo12_e1 = QLineEdit(self)
        self.py_lo12_e1.move(150, 100);
        self.py_lo12_e1.setText("1.0");
        tmp.append(self.py_lo12_e1);

        self.py_lo12_l2 = QLabel(self);
        self.py_lo12_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo12_l2.move(20, 150);
        tmp.append(self.py_lo12_l2);

        self.py_lo12_e2 = QLineEdit(self)
        self.py_lo12_e2.move(290, 150);
        self.py_lo12_e2.setText("0");
        tmp.append(self.py_lo12_e2);

        self.loss_ui_pytorch.append(tmp)




        tmp = [];
        self.py_lo13_l1 = QLabel(self);
        self.py_lo13_l1.setText("1. Scalar Weight: ");
        self.py_lo13_l1.move(20, 100);
        tmp.append(self.py_lo13_l1);

        self.py_lo13_e1 = QLineEdit(self)
        self.py_lo13_e1.move(150, 100);
        self.py_lo13_e1.setText("1.0");
        tmp.append(self.py_lo13_e1);

        self.py_lo13_l2 = QLabel(self);
        self.py_lo13_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo13_l2.move(20, 150);
        tmp.append(self.py_lo13_l2);

        self.py_lo13_e2 = QLineEdit(self)
        self.py_lo13_e2.move(290, 150);
        self.py_lo13_e2.setText("0");
        tmp.append(self.py_lo13_e2);

        self.loss_ui_pytorch.append(tmp)





        tmp = [];
        self.py_lo14_l1 = QLabel(self);
        self.py_lo14_l1.setText("1. Scalar Weight: ");
        self.py_lo14_l1.move(20, 100);
        tmp.append(self.py_lo14_l1);

        self.py_lo14_e1 = QLineEdit(self)
        self.py_lo14_e1.move(150, 100);
        self.py_lo14_e1.setText("1.0");
        tmp.append(self.py_lo14_e1);

        self.py_lo14_l2 = QLabel(self);
        self.py_lo14_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo14_l2.move(20, 150);
        tmp.append(self.py_lo14_l2);

        self.py_lo14_e2 = QLineEdit(self)
        self.py_lo14_e2.move(290, 150);
        self.py_lo14_e2.setText("0");
        tmp.append(self.py_lo14_e2);

        self.loss_ui_pytorch.append(tmp)




        tmp = [];
        self.py_lo15_l1 = QLabel(self);
        self.py_lo15_l1.setText("1. Scalar Weight: ");
        self.py_lo15_l1.move(20, 100);
        tmp.append(self.py_lo15_l1);

        self.py_lo15_e1 = QLineEdit(self)
        self.py_lo15_e1.move(150, 100);
        self.py_lo15_e1.setText("1.0");
        tmp.append(self.py_lo15_e1);

        self.py_lo15_l2 = QLabel(self);
        self.py_lo15_l2.setText("2. Batch Axis (0, 1, 2, 3): ");
        self.py_lo15_l2.move(20, 150);
        tmp.append(self.py_lo15_l2);

        self.py_lo15_e2 = QLineEdit(self)
        self.py_lo15_e2.move(290, 150);
        self.py_lo15_e2.setText("0");
        tmp.append(self.py_lo15_e2);

        self.loss_ui_pytorch.append(tmp)
















        self.select_loss();

        self.tb1 = QTextEdit(self)
        self.tb1.move(550, 20)
        self.tb1.resize(300, 500)
        if(self.system["update"]["losses"]["active"]):
            wr = "";
            wr = json.dumps(self.system["update"]["losses"]["value"], indent=4)
            self.tb1.setText(wr);
        else:
            self.tb1.setText("Using Default loss.")


        self.b4 = QPushButton('Select loss', self)
        self.b4.move(400,400)
        self.b4.clicked.connect(self.add_loss)

        
        self.b6 = QPushButton('Clear ', self)
        self.b6.move(400,500)
        self.b6.clicked.connect(self.clear_loss)









        

        
    def select_loss(self):
        self.current_loss = {};
        self.current_loss["name"] = "";
        self.current_loss["params"] = {};

        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.current_loss["name"] = self.cb1.currentText();
            index = self.mxnet_losses_list.index(self.cb1.currentText());
            for i in range(len(self.loss_ui_mxnet)):
                for j in range(len(self.loss_ui_mxnet[i])):
                    if((index-1)==i):
                        self.loss_ui_mxnet[i][j].show();
                    else:
                        self.loss_ui_mxnet[i][j].hide();

            for i in range(len(self.loss_ui_keras)):
                for j in range(len(self.loss_ui_keras[i])):
                    self.loss_ui_keras[i][j].hide();
            for i in range(len(self.loss_ui_pytorch)):
                for j in range(len(self.loss_ui_pytorch[i])):
                    self.loss_ui_pytorch[i][j].hide();
            


        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.current_loss["name"] = self.cb2.currentText();
            index = self.keras_losses_list.index(self.cb2.currentText());
            for i in range(len(self.loss_ui_keras)):
                for j in range(len(self.loss_ui_keras[i])):
                    if((index-1)==i):
                        self.loss_ui_keras[i][j].show();
                    else:
                        self.loss_ui_keras[i][j].hide();

            for i in range(len(self.loss_ui_mxnet)):
                for j in range(len(self.loss_ui_mxnet[i])):
                    self.loss_ui_mxnet[i][j].hide();
            for i in range(len(self.loss_ui_pytorch)):
                for j in range(len(self.loss_ui_pytorch[i])):
                    self.loss_ui_pytorch[i][j].hide();



        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.current_loss["name"] = self.cb3.currentText();
            index = self.pytorch_losses_list.index(self.cb3.currentText());
            for i in range(len(self.loss_ui_pytorch)):
                for j in range(len(self.loss_ui_pytorch[i])):
                    if((index-1)==i):
                        self.loss_ui_pytorch[i][j].show();
                    else:
                        self.loss_ui_pytorch[i][j].hide();

            for i in range(len(self.loss_ui_keras)):
                for j in range(len(self.loss_ui_keras[i])):
                    self.loss_ui_keras[i][j].hide();
            for i in range(len(self.loss_ui_mxnet)):
                for j in range(len(self.loss_ui_mxnet[i])):
                    self.loss_ui_mxnet[i][j].hide();



    def add_loss(self):
        self.system["update"]["losses"]["active"] = True;
        if(self.system["backend"] == "Mxnet-1.5.1"):
            if(self.current_loss["name"] == self.mxnet_losses_list[1]):
                self.current_loss["params"]["weight"] = self.mx_lo1_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo1_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.mxnet_losses_list[2]):
                self.current_loss["params"]["weight"] = self.mx_lo2_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo2_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.mxnet_losses_list[3]):
                self.current_loss["params"]["weight"] = self.mx_lo3_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo3_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.mxnet_losses_list[4]):
                self.current_loss["params"]["weight"] = self.mx_lo4_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo4_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.mxnet_losses_list[5]):
                self.current_loss["params"]["weight"] = self.mx_lo5_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo5_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.mxnet_losses_list[6]):
                self.current_loss["params"]["weight"] = self.mx_lo6_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo6_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.mxnet_losses_list[7]):
                self.current_loss["params"]["weight"] = self.mx_lo7_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo7_e2.text();
                self.current_loss["params"]["log_pre_applied"] = self.mx_lo7_cb3.currentText();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.mxnet_losses_list[8]):
                self.current_loss["params"]["weight"] = self.mx_lo8_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo8_e2.text();
                self.current_loss["params"]["log_pre_applied"] = self.mx_lo8_cb3.currentText();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.mxnet_losses_list[9]):
                self.current_loss["params"]["weight"] = self.mx_lo9_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo9_e2.text();
                self.current_loss["params"]["threshold_for_mean_estimator"] = self.mx_lo9_e3.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.mxnet_losses_list[10]):
                self.current_loss["params"]["weight"] = self.mx_lo10_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo10_e2.text();
                self.current_loss["params"]["margin"] = self.mx_lo10_e3.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.mxnet_losses_list[11]):
                self.current_loss["params"]["weight"] = self.mx_lo11_e1.text();
                self.current_loss["params"]["batch_axis"] = self.mx_lo11_e2.text();
                self.current_loss["params"]["margin"] = self.mx_lo11_e3.text();
                self.system["update"]["losses"]["value"] = self.current_loss;



        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            if(self.current_loss["name"] == self.keras_losses_list[1]):
                self.current_loss["params"]["weight"] = self.ke_lo1_e1.text();
                self.current_loss["params"]["batch_axis"] = self.ke_lo1_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.keras_losses_list[2]):
                self.current_loss["params"]["weight"] = self.ke_lo2_e1.text();
                self.current_loss["params"]["batch_axis"] = self.ke_lo2_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.keras_losses_list[3]):
                self.current_loss["params"]["weight"] = self.ke_lo3_e1.text();
                self.current_loss["params"]["batch_axis"] = self.ke_lo3_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.keras_losses_list[4]):
                self.current_loss["params"]["weight"] = self.ke_lo4_e1.text();
                self.current_loss["params"]["batch_axis"] = self.ke_lo4_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.keras_losses_list[5]):
                self.current_loss["params"]["weight"] = self.ke_lo5_e1.text();
                self.current_loss["params"]["batch_axis"] = self.ke_lo5_e2.text();
                self.current_loss["params"]["log_pre_applied"] = self.ke_lo5_cb3.currentText();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.keras_losses_list[6]):
                self.current_loss["params"]["weight"] = self.ke_lo6_e1.text();
                self.current_loss["params"]["batch_axis"] = self.ke_lo6_e2.text();
                self.current_loss["params"]["margin"] = self.ke_lo6_e3.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.keras_losses_list[7]):
                self.current_loss["params"]["weight"] = self.ke_lo7_e1.text();
                self.current_loss["params"]["batch_axis"] = self.ke_lo7_e2.text();
                self.current_loss["params"]["margin"] = self.ke_lo7_e3.text();
                self.system["update"]["losses"]["value"] = self.current_loss;



        elif(self.system["backend"] == "Pytorch-1.3.1"):
            if(self.current_loss["name"] == self.pytorch_losses_list[1]):
                self.current_loss["params"]["weight"] = self.py_lo1_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo1_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[2]):
                self.current_loss["params"]["weight"] = self.py_lo2_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo2_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[3]):
                self.current_loss["params"]["weight"] = self.py_lo3_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo3_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[4]):
                self.current_loss["params"]["weight"] = self.py_lo4_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo4_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[5]):
                self.current_loss["params"]["weight"] = self.py_lo5_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo5_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[6]):
                self.current_loss["params"]["weight"] = self.py_lo6_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo6_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[7]):
                self.current_loss["params"]["weight"] = self.py_lo7_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo7_e2.text();
                self.current_loss["params"]["log_pre_applied"] = self.py_lo7_cb3.currentText();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[8]):
                self.current_loss["params"]["weight"] = self.py_lo8_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo8_e2.text();
                self.current_loss["params"]["log_pre_applied"] = self.py_lo8_cb3.currentText();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[9]):
                self.current_loss["params"]["weight"] = self.py_lo9_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo9_e2.text();
                self.current_loss["params"]["threshold_for_mean_estimator"] = self.py_lo9_e3.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[10]):
                self.current_loss["params"]["weight"] = self.py_lo10_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo10_e2.text();
                self.current_loss["params"]["margin"] = self.py_lo10_e3.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[11]):
                self.current_loss["params"]["weight"] = self.py_lo11_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo11_e2.text();
                self.current_loss["params"]["margin"] = self.py_lo11_e3.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[12]):
                self.current_loss["params"]["weight"] = self.py_lo12_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo12_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[13]):
                self.current_loss["params"]["weight"] = self.py_lo13_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo13_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[14]):
                self.current_loss["params"]["weight"] = self.py_lo14_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo14_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;

            elif(self.current_loss["name"] == self.pytorch_losses_list[15]):
                self.current_loss["params"]["weight"] = self.py_lo15_e1.text();
                self.current_loss["params"]["batch_axis"] = self.py_lo15_e2.text();
                self.system["update"]["losses"]["value"] = self.current_loss;






        wr = "";
        wr = json.dumps(self.system["update"]["losses"]["value"], indent=4)
        self.tb1.setText(wr);


    def clear_loss(self):
        self.system["update"]["losses"]["value"] = "";
        self.system["update"]["losses"]["active"] = False;

        wr = "";
        self.tb1.setText(wr);





    def forward(self):        
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_train.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_scheduler_param.emit();



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateLossParam()
screen.show()
sys.exit(app.exec_())
'''