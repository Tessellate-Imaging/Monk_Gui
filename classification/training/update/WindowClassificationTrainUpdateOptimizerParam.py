import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateOptimizerParam(QtWidgets.QWidget):

    forward_scheduler_param = QtCore.pyqtSignal();
    forward_analyse_optimizer_lr = QtCore.pyqtSignal();
    backward_train_param = QtCore.pyqtSignal();



    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Update Optimizer Params'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 600
        self.optimizer_ui_mxnet = [];
        self.optimizer_ui_keras = [];
        self.optimizer_ui_pytorch = [];
        self.current_optimizer = {};
        self.current_optimizer["name"] = "";
        self.current_optimizer["params"] = {};
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
        self.cb1.activated.connect(self.select_optimizer);

        self.cb2 = QComboBox(self);
        self.cb2.move(20, 20);
        self.cb2.activated.connect(self.select_optimizer);

        self.cb3 = QComboBox(self);
        self.cb3.move(20, 20);
        self.cb3.activated.connect(self.select_optimizer);



        self.mxnet_optimizers_list = ["select", "optimizer_sgd", "optimizer_nesterov_sgd", "optimizer_rmsprop", "optimizer_momentum_rmsprop", 
                                    "optimizer_adam", "optimizer_adamax", "optimizer_nesterov_adam", "optimizer_adagrad",  
                                    "optimizer_adadelta"];

        self.keras_optimizers_list = ["select", "optimizer_sgd", "optimizer_nesterov_sgd", "optimizer_rmsprop", "optimizer_adam",
                                    "optimizer_nesterov_adam", "optimizer_adamax", "optimizer_adagrad", "optimizer_adadelta"];

        self.pytorch_optimizers_list = ["select", "optimizer_sgd", "optimizer_nesterov_sgd", "optimizer_rmsprop", "optimizer_momentum_rmsprop", 
                                    "optimizer_adam", "optimizer_adamax", "optimizer_adamw", "optimizer_adagrad", 
                                    "optimizer_adadelta"];


        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.cb1.addItems(self.mxnet_optimizers_list);
            self.cb1.show();
            self.cb2.hide();
            self.cb3.hide();
        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.cb2.addItems(self.keras_optimizers_list);
            self.cb2.show();
            self.cb1.hide();
            self.cb3.hide();
        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.cb3.addItems(self.pytorch_optimizers_list);
            self.cb3.show();
            self.cb1.hide();
            self.cb2.hide();


        self.btn1 = QPushButton('Autotune this hyperparam', self)
        self.btn1.move(300, 20)
        self.btn1.clicked.connect(self.analyse_optimizer_lr);




        tmp = [];
        self.mx_op1_l1 = QLabel(self);
        self.mx_op1_l1.setText("1. Learning Rate:");
        self.mx_op1_l1.move(20, 100);
        tmp.append(self.mx_op1_l1);

        self.mx_op1_e1 = QLineEdit(self)
        self.mx_op1_e1.move(150, 100);
        self.mx_op1_e1.setText("0.001");
        tmp.append(self.mx_op1_e1);


        self.mx_op1_l2 = QLabel(self);
        self.mx_op1_l2.setText("2. Momentum:");
        self.mx_op1_l2.move(20, 150);
        tmp.append(self.mx_op1_l2);

        self.mx_op1_e2 = QLineEdit(self)
        self.mx_op1_e2.move(150, 150);
        self.mx_op1_e2.setText("0.9");
        tmp.append(self.mx_op1_e2);


        self.mx_op1_l3 = QLabel(self);
        self.mx_op1_l3.setText("3. Weight decay:");
        self.mx_op1_l3.move(20, 200);
        tmp.append(self.mx_op1_l3);

        self.mx_op1_e3 = QLineEdit(self)
        self.mx_op1_e3.move(150, 200);
        self.mx_op1_e3.setText("0.0");
        tmp.append(self.mx_op1_e3);

        self.optimizer_ui_mxnet.append(tmp)






        tmp = [];
        self.mx_op2_l1 = QLabel(self);
        self.mx_op2_l1.setText("1. Learning Rate:");
        self.mx_op2_l1.move(20, 100);
        tmp.append(self.mx_op2_l1);

        self.mx_op2_e1 = QLineEdit(self)
        self.mx_op2_e1.move(150, 100);
        self.mx_op2_e1.setText("0.001");
        tmp.append(self.mx_op2_e1);


        self.mx_op2_l2 = QLabel(self);
        self.mx_op2_l2.setText("2. Momentum:");
        self.mx_op2_l2.move(20, 150);
        tmp.append(self.mx_op2_l2);

        self.mx_op2_e2 = QLineEdit(self)
        self.mx_op2_e2.move(150, 150);
        self.mx_op2_e2.setText("0.9");
        tmp.append(self.mx_op2_e2);


        self.mx_op2_l3 = QLabel(self);
        self.mx_op2_l3.setText("3. Weight decay:");
        self.mx_op2_l3.move(20, 200);
        tmp.append(self.mx_op2_l3);

        self.mx_op2_e3 = QLineEdit(self)
        self.mx_op2_e3.move(150, 200);
        self.mx_op2_e3.setText("0.0");
        tmp.append(self.mx_op2_e3);

        self.optimizer_ui_mxnet.append(tmp)






        tmp = [];
        self.mx_op3_l1 = QLabel(self);
        self.mx_op3_l1.setText("1. Learning Rate:");
        self.mx_op3_l1.move(20, 100);
        tmp.append(self.mx_op3_l1);

        self.mx_op3_e1 = QLineEdit(self)
        self.mx_op3_e1.move(150, 100);
        self.mx_op3_e1.setText("0.001");
        tmp.append(self.mx_op3_e1);


        self.mx_op3_l2 = QLabel(self);
        self.mx_op3_l2.setText("2. Decay Rate:");
        self.mx_op3_l2.move(20, 150);
        tmp.append(self.mx_op3_l2);

        self.mx_op3_e2 = QLineEdit(self)
        self.mx_op3_e2.move(150, 150);
        self.mx_op3_e2.setText("0.99");
        tmp.append(self.mx_op3_e2);


        self.mx_op3_l3 = QLabel(self);
        self.mx_op3_l3.setText("3. Weight decay:");
        self.mx_op3_l3.move(20, 200);
        tmp.append(self.mx_op3_l3);

        self.mx_op3_e3 = QLineEdit(self)
        self.mx_op3_e3.move(150, 200);
        self.mx_op3_e3.setText("0.0");
        tmp.append(self.mx_op3_e3);


        self.mx_op3_l4 = QLabel(self);
        self.mx_op3_l4.setText("4. Epsilon:");
        self.mx_op3_l4.move(20, 250);
        tmp.append(self.mx_op3_l4);

        self.mx_op3_e4 = QLineEdit(self)
        self.mx_op3_e4.move(150, 250);
        self.mx_op3_e4.setText("0.00001");
        tmp.append(self.mx_op3_e4);

        self.optimizer_ui_mxnet.append(tmp)







        tmp = [];
        self.mx_op4_l1 = QLabel(self);
        self.mx_op4_l1.setText("1. Learning Rate:");
        self.mx_op4_l1.move(20, 100);
        tmp.append(self.mx_op4_l1);

        self.mx_op4_e1 = QLineEdit(self)
        self.mx_op4_e1.move(150, 100);
        self.mx_op4_e1.setText("0.001");
        tmp.append(self.mx_op4_e1);


        self.mx_op4_l2 = QLabel(self);
        self.mx_op4_l2.setText("2. Decay Rate:");
        self.mx_op4_l2.move(20, 150);
        tmp.append(self.mx_op4_l2);

        self.mx_op4_e2 = QLineEdit(self)
        self.mx_op4_e2.move(150, 150);
        self.mx_op4_e2.setText("0.99");
        tmp.append(self.mx_op4_e2);


        self.mx_op4_l3 = QLabel(self);
        self.mx_op4_l3.setText("3. Weight decay:");
        self.mx_op4_l3.move(20, 200);
        tmp.append(self.mx_op4_l3);

        self.mx_op4_e3 = QLineEdit(self)
        self.mx_op4_e3.move(150, 200);
        self.mx_op4_e3.setText("0.0");
        tmp.append(self.mx_op4_e3);


        self.mx_op4_l4 = QLabel(self);
        self.mx_op4_l4.setText("4. Epsilon:");
        self.mx_op4_l4.move(20, 250);
        tmp.append(self.mx_op4_l4);

        self.mx_op4_e4 = QLineEdit(self)
        self.mx_op4_e4.move(150, 250);
        self.mx_op4_e4.setText("0.00001");
        tmp.append(self.mx_op4_e4);

        self.optimizer_ui_mxnet.append(tmp)






        tmp = [];
        self.mx_op5_l1 = QLabel(self);
        self.mx_op5_l1.setText("1. Learning Rate:");
        self.mx_op5_l1.move(20, 100);
        tmp.append(self.mx_op5_l1);

        self.mx_op5_e1 = QLineEdit(self)
        self.mx_op5_e1.move(150, 100);
        self.mx_op5_e1.setText("0.001");
        tmp.append(self.mx_op5_e1);


        self.mx_op5_l2 = QLabel(self);
        self.mx_op5_l2.setText("2. beta1:");
        self.mx_op5_l2.move(20, 150);
        tmp.append(self.mx_op5_l2);

        self.mx_op5_e2 = QLineEdit(self)
        self.mx_op5_e2.move(150, 150);
        self.mx_op5_e2.setText("0.9");
        tmp.append(self.mx_op5_e2);


        self.mx_op5_l3 = QLabel(self);
        self.mx_op5_l3.setText("3. beta2:");
        self.mx_op5_l3.move(20, 200);
        tmp.append(self.mx_op5_l3);

        self.mx_op5_e3 = QLineEdit(self)
        self.mx_op5_e3.move(150, 200);
        self.mx_op5_e3.setText("0.999");
        tmp.append(self.mx_op5_e3);


        self.mx_op5_l4 = QLabel(self);
        self.mx_op5_l4.setText("4. weight_decay:");
        self.mx_op5_l4.move(20, 250);
        tmp.append(self.mx_op5_l4);

        self.mx_op5_e4 = QLineEdit(self)
        self.mx_op5_e4.move(150, 250);
        self.mx_op5_e4.setText("0.0");
        tmp.append(self.mx_op5_e4);


        self.mx_op5_l5 = QLabel(self);
        self.mx_op5_l5.setText("5. Epsilon:");
        self.mx_op5_l5.move(20, 300);
        tmp.append(self.mx_op5_l5);

        self.mx_op5_e5 = QLineEdit(self)
        self.mx_op5_e5.move(150, 300);
        self.mx_op5_e5.setText("0.00000001");
        tmp.append(self.mx_op5_e5);

        self.optimizer_ui_mxnet.append(tmp)





        tmp = [];
        self.mx_op6_l1 = QLabel(self);
        self.mx_op6_l1.setText("1. Learning Rate:");
        self.mx_op6_l1.move(20, 100);
        tmp.append(self.mx_op6_l1);

        self.mx_op6_e1 = QLineEdit(self)
        self.mx_op6_e1.move(150, 100);
        self.mx_op6_e1.setText("0.001");
        tmp.append(self.mx_op6_e1);


        self.mx_op6_l2 = QLabel(self);
        self.mx_op6_l2.setText("2. beta1:");
        self.mx_op6_l2.move(20, 150);
        tmp.append(self.mx_op6_l2);

        self.mx_op6_e2 = QLineEdit(self)
        self.mx_op6_e2.move(150, 150);
        self.mx_op6_e2.setText("0.9");
        tmp.append(self.mx_op6_e2);


        self.mx_op6_l3 = QLabel(self);
        self.mx_op6_l3.setText("3. beta2:");
        self.mx_op6_l3.move(20, 200);
        tmp.append(self.mx_op6_l3);

        self.mx_op6_e3 = QLineEdit(self)
        self.mx_op6_e3.move(150, 200);
        self.mx_op6_e3.setText("0.999");
        tmp.append(self.mx_op6_e3);


        self.mx_op6_l4 = QLabel(self);
        self.mx_op6_l4.setText("4. weight_decay:");
        self.mx_op6_l4.move(20, 250);
        tmp.append(self.mx_op6_l4);

        self.mx_op6_e4 = QLineEdit(self)
        self.mx_op6_e4.move(150, 250);
        self.mx_op6_e4.setText("0.0");
        tmp.append(self.mx_op6_e4);


        self.mx_op6_l5 = QLabel(self);
        self.mx_op6_l5.setText("5. Epsilon:");
        self.mx_op6_l5.move(20, 300);
        tmp.append(self.mx_op6_l5);

        self.mx_op6_e5 = QLineEdit(self)
        self.mx_op6_e5.move(150, 300);
        self.mx_op6_e5.setText("0.00000001");
        tmp.append(self.mx_op6_e5);

        self.optimizer_ui_mxnet.append(tmp)






        tmp = [];
        self.mx_op7_l1 = QLabel(self);
        self.mx_op7_l1.setText("1. Learning Rate:");
        self.mx_op7_l1.move(20, 100);
        tmp.append(self.mx_op7_l1);

        self.mx_op7_e1 = QLineEdit(self)
        self.mx_op7_e1.move(150, 100);
        self.mx_op7_e1.setText("0.001");
        tmp.append(self.mx_op7_e1);


        self.mx_op7_l2 = QLabel(self);
        self.mx_op7_l2.setText("2. beta1:");
        self.mx_op7_l2.move(20, 150);
        tmp.append(self.mx_op7_l2);

        self.mx_op7_e2 = QLineEdit(self)
        self.mx_op7_e2.move(150, 150);
        self.mx_op7_e2.setText("0.9");
        tmp.append(self.mx_op7_e2);


        self.mx_op7_l3 = QLabel(self);
        self.mx_op7_l3.setText("3. beta2:");
        self.mx_op7_l3.move(20, 200);
        tmp.append(self.mx_op7_l3);

        self.mx_op7_e3 = QLineEdit(self)
        self.mx_op7_e3.move(150, 200);
        self.mx_op7_e3.setText("0.999");
        tmp.append(self.mx_op7_e3);


        self.mx_op7_l4 = QLabel(self);
        self.mx_op7_l4.setText("4. weight_decay:");
        self.mx_op7_l4.move(20, 250);
        tmp.append(self.mx_op7_l4);

        self.mx_op7_e4 = QLineEdit(self)
        self.mx_op7_e4.move(150, 250);
        self.mx_op7_e4.setText("0.0");
        tmp.append(self.mx_op7_e4);


        self.mx_op7_l5 = QLabel(self);
        self.mx_op7_l5.setText("5. Apply amsgrad:");
        self.mx_op7_l5.move(20, 300);
        tmp.append(self.mx_op7_l5);

        self.mx_op7_cb5 = QComboBox(self);
        self.mx_op7_cb5.move(200, 300);
        self.mx_op7_cb5.addItems(["No", "Yes"]);
        tmp.append(self.mx_op7_cb5);

        self.mx_op7_l6 = QLabel(self);
        self.mx_op7_l6.setText("6. Momentum decay:");
        self.mx_op7_l6.move(20, 350);
        tmp.append(self.mx_op7_l6);

        self.mx_op7_e6 = QLineEdit(self)
        self.mx_op7_e6.move(190, 350);
        self.mx_op7_e6.setText("0.004");
        tmp.append(self.mx_op7_e6);

        self.mx_op7_l7 = QLabel(self);
        self.mx_op7_l7.setText("7. Epsilon:");
        self.mx_op7_l7.move(20, 400);
        tmp.append(self.mx_op7_l7);

        self.mx_op7_e7 = QLineEdit(self)
        self.mx_op7_e7.move(150, 400);
        self.mx_op7_e7.setText("0.00000001");
        tmp.append(self.mx_op7_e7);

        self.optimizer_ui_mxnet.append(tmp)






        tmp = [];
        self.mx_op8_l1 = QLabel(self);
        self.mx_op8_l1.setText("1. Learning Rate:");
        self.mx_op8_l1.move(20, 100);
        tmp.append(self.mx_op8_l1);

        self.mx_op8_e1 = QLineEdit(self)
        self.mx_op8_e1.move(150, 100);
        self.mx_op8_e1.setText("0.001");
        tmp.append(self.mx_op8_e1);


        self.mx_op8_l2 = QLabel(self);
        self.mx_op8_l2.setText("2. Learning rate decay:");
        self.mx_op8_l2.move(20, 150);
        tmp.append(self.mx_op8_l2);

        self.mx_op8_e2 = QLineEdit(self)
        self.mx_op8_e2.move(190, 150);
        self.mx_op8_e2.setText("0.0");
        tmp.append(self.mx_op8_e2);


        self.mx_op8_l3 = QLabel(self);
        self.mx_op8_l3.setText("3. Weight decay:");
        self.mx_op8_l3.move(20, 200);
        tmp.append(self.mx_op8_l3);

        self.mx_op8_e3 = QLineEdit(self)
        self.mx_op8_e3.move(150, 200);
        self.mx_op8_e3.setText("0.0");
        tmp.append(self.mx_op8_e3);


        self.mx_op8_l4 = QLabel(self);
        self.mx_op8_l4.setText("4. Epsilon:");
        self.mx_op8_l4.move(20, 250);
        tmp.append(self.mx_op8_l4);

        self.mx_op8_e4 = QLineEdit(self)
        self.mx_op8_e4.move(150, 250);
        self.mx_op8_e4.setText("0.000000001");
        tmp.append(self.mx_op8_e4);

        self.optimizer_ui_mxnet.append(tmp)






        tmp = [];
        self.mx_op9_l1 = QLabel(self);
        self.mx_op9_l1.setText("1. Learning Rate:");
        self.mx_op9_l1.move(20, 100);
        tmp.append(self.mx_op9_l1);

        self.mx_op9_e1 = QLineEdit(self)
        self.mx_op9_e1.move(150, 100);
        self.mx_op9_e1.setText("0.001");
        tmp.append(self.mx_op9_e1);


        self.mx_op9_l2 = QLabel(self);
        self.mx_op9_l2.setText("2. Rho param:");
        self.mx_op9_l2.move(20, 150);
        tmp.append(self.mx_op9_l2);

        self.mx_op9_e2 = QLineEdit(self)
        self.mx_op9_e2.move(150, 150);
        self.mx_op9_e2.setText("0.9");
        tmp.append(self.mx_op9_e2);


        self.mx_op9_l3 = QLabel(self);
        self.mx_op9_l3.setText("3. Weight decay:");
        self.mx_op9_l3.move(20, 200);
        tmp.append(self.mx_op9_l3);

        self.mx_op9_e3 = QLineEdit(self)
        self.mx_op9_e3.move(150, 200);
        self.mx_op9_e3.setText("0.0");
        tmp.append(self.mx_op9_e3);


        self.mx_op9_l4 = QLabel(self);
        self.mx_op9_l4.setText("4. Epsilon:");
        self.mx_op9_l4.move(20, 250);
        tmp.append(self.mx_op9_l4);

        self.mx_op9_e4 = QLineEdit(self)
        self.mx_op9_e4.move(150, 250);
        self.mx_op9_e4.setText("0.000000001");
        tmp.append(self.mx_op9_e4);

        self.optimizer_ui_mxnet.append(tmp)











        tmp = [];
        self.ke_op1_l1 = QLabel(self);
        self.ke_op1_l1.setText("1. Learning Rate:");
        self.ke_op1_l1.move(20, 100);
        tmp.append(self.ke_op1_l1);

        self.ke_op1_e1 = QLineEdit(self)
        self.ke_op1_e1.move(150, 100);
        self.ke_op1_e1.setText("0.001");
        tmp.append(self.ke_op1_e1);


        self.ke_op1_l2 = QLabel(self);
        self.ke_op1_l2.setText("2. Momentum:");
        self.ke_op1_l2.move(20, 150);
        tmp.append(self.ke_op1_l2);

        self.ke_op1_e2 = QLineEdit(self)
        self.ke_op1_e2.move(150, 150);
        self.ke_op1_e2.setText("0.9");
        tmp.append(self.ke_op1_e2);


        self.ke_op1_l3 = QLabel(self);
        self.ke_op1_l3.setText("3. Weight decay:");
        self.ke_op1_l3.move(20, 200);
        tmp.append(self.ke_op1_l3);

        self.ke_op1_e3 = QLineEdit(self)
        self.ke_op1_e3.move(150, 200);
        self.ke_op1_e3.setText("0.0");
        tmp.append(self.ke_op1_e3);

        self.optimizer_ui_keras.append(tmp)






        tmp = [];
        self.ke_op2_l1 = QLabel(self);
        self.ke_op2_l1.setText("1. Learning Rate:");
        self.ke_op2_l1.move(20, 100);
        tmp.append(self.ke_op2_l1);

        self.ke_op2_e1 = QLineEdit(self)
        self.ke_op2_e1.move(150, 100);
        self.ke_op2_e1.setText("0.001");
        tmp.append(self.ke_op2_e1);


        self.ke_op2_l2 = QLabel(self);
        self.ke_op2_l2.setText("2. Momentum:");
        self.ke_op2_l2.move(20, 150);
        tmp.append(self.ke_op2_l2);

        self.ke_op2_e2 = QLineEdit(self)
        self.ke_op2_e2.move(150, 150);
        self.ke_op2_e2.setText("0.9");
        tmp.append(self.ke_op2_e2);


        self.ke_op2_l3 = QLabel(self);
        self.ke_op2_l3.setText("3. Weight decay:");
        self.ke_op2_l3.move(20, 200);
        tmp.append(self.ke_op2_l3);

        self.ke_op2_e3 = QLineEdit(self)
        self.ke_op2_e3.move(150, 200);
        self.ke_op2_e3.setText("0.0");
        tmp.append(self.ke_op2_e3);

        self.optimizer_ui_keras.append(tmp)





        tmp = [];
        self.ke_op3_l1 = QLabel(self);
        self.ke_op3_l1.setText("1. Learning Rate:");
        self.ke_op3_l1.move(20, 100);
        tmp.append(self.ke_op3_l1);

        self.ke_op3_e1 = QLineEdit(self)
        self.ke_op3_e1.move(150, 100);
        self.ke_op3_e1.setText("0.001");
        tmp.append(self.ke_op3_e1);


        self.ke_op3_l2 = QLabel(self);
        self.ke_op3_l2.setText("2. Decay rate:");
        self.ke_op3_l2.move(20, 150);
        tmp.append(self.ke_op3_l2);

        self.ke_op3_e2 = QLineEdit(self)
        self.ke_op3_e2.move(150, 150);
        self.ke_op3_e2.setText("0.99");
        tmp.append(self.ke_op3_e2);


        self.ke_op3_l3 = QLabel(self);
        self.ke_op3_l3.setText("3. Weight decay:");
        self.ke_op3_l3.move(20, 200);
        tmp.append(self.ke_op3_l3);

        self.ke_op3_e3 = QLineEdit(self)
        self.ke_op3_e3.move(150, 200);
        self.ke_op3_e3.setText("0.0");
        tmp.append(self.ke_op3_e3);

        self.ke_op3_l4 = QLabel(self);
        self.ke_op3_l4.setText("4. Epsilon:");
        self.ke_op3_l4.move(20, 250);
        tmp.append(self.ke_op3_l4);

        self.ke_op3_e4 = QLineEdit(self)
        self.ke_op3_e4.move(150, 250);
        self.ke_op3_e4.setText("0.000000001");
        tmp.append(self.ke_op3_e4);

        self.optimizer_ui_keras.append(tmp)





        tmp = [];
        self.ke_op4_l1 = QLabel(self);
        self.ke_op4_l1.setText("1. Learning Rate:");
        self.ke_op4_l1.move(20, 100);
        tmp.append(self.ke_op4_l1);

        self.ke_op4_e1 = QLineEdit(self)
        self.ke_op4_e1.move(150, 100);
        self.ke_op4_e1.setText("0.001");
        tmp.append(self.ke_op4_e1);


        self.ke_op4_l2 = QLabel(self);
        self.ke_op4_l2.setText("2. beta1:");
        self.ke_op4_l2.move(20, 150);
        tmp.append(self.ke_op4_l2);

        self.ke_op4_e2 = QLineEdit(self)
        self.ke_op4_e2.move(150, 150);
        self.ke_op4_e2.setText("0.9");
        tmp.append(self.ke_op4_e2);


        self.ke_op4_l3 = QLabel(self);
        self.ke_op4_l3.setText("3. beta2:");
        self.ke_op4_l3.move(20, 200);
        tmp.append(self.ke_op4_l3);

        self.ke_op4_e3 = QLineEdit(self)
        self.ke_op4_e3.move(150, 200);
        self.ke_op4_e3.setText("0.999");
        tmp.append(self.ke_op4_e3);


        self.ke_op4_l4 = QLabel(self);
        self.ke_op4_l4.setText("4. weight_decay:");
        self.ke_op4_l4.move(20, 250);
        tmp.append(self.ke_op4_l4);

        self.ke_op4_e4 = QLineEdit(self)
        self.ke_op4_e4.move(150, 250);
        self.ke_op4_e4.setText("0.0");
        tmp.append(self.ke_op4_e4);


        self.ke_op4_l5 = QLabel(self);
        self.ke_op4_l5.setText("5. Epsilon:");
        self.ke_op4_l5.move(20, 300);
        tmp.append(self.ke_op4_l5);

        self.ke_op4_e5 = QLineEdit(self)
        self.ke_op4_e5.move(150, 300);
        self.ke_op4_e5.setText("0.00000001");
        tmp.append(self.ke_op4_e5);

        self.optimizer_ui_keras.append(tmp)





        tmp = [];
        self.ke_op5_l1 = QLabel(self);
        self.ke_op5_l1.setText("1. Learning Rate:");
        self.ke_op5_l1.move(20, 100);
        tmp.append(self.ke_op5_l1);

        self.ke_op5_e1 = QLineEdit(self)
        self.ke_op5_e1.move(150, 100);
        self.ke_op5_e1.setText("0.001");
        tmp.append(self.ke_op5_e1);


        self.ke_op5_l2 = QLabel(self);
        self.ke_op5_l2.setText("2. beta1:");
        self.ke_op5_l2.move(20, 150);
        tmp.append(self.ke_op5_l2);

        self.ke_op5_e2 = QLineEdit(self)
        self.ke_op5_e2.move(150, 150);
        self.ke_op5_e2.setText("0.9");
        tmp.append(self.ke_op5_e2);


        self.ke_op5_l3 = QLabel(self);
        self.ke_op5_l3.setText("3. beta2:");
        self.ke_op5_l3.move(20, 200);
        tmp.append(self.ke_op5_l3);

        self.ke_op5_e3 = QLineEdit(self)
        self.ke_op5_e3.move(150, 200);
        self.ke_op5_e3.setText("0.999");
        tmp.append(self.ke_op5_e3);


        self.ke_op5_l4 = QLabel(self);
        self.ke_op5_l4.setText("4. weight_decay:");
        self.ke_op5_l4.move(20, 250);
        tmp.append(self.ke_op5_l4);

        self.ke_op5_e4 = QLineEdit(self)
        self.ke_op5_e4.move(150, 250);
        self.ke_op5_e4.setText("0.0");
        tmp.append(self.ke_op5_e4);


        self.ke_op5_l5 = QLabel(self);
        self.ke_op5_l5.setText("5. Apply amsgrad:");
        self.ke_op5_l5.move(20, 300);
        tmp.append(self.ke_op5_l5);

        self.ke_op5_cb5 = QComboBox(self);
        self.ke_op5_cb5.move(200, 300);
        self.ke_op5_cb5.addItems(["No", "Yes"]);
        tmp.append(self.ke_op5_cb5);

        self.ke_op5_l6 = QLabel(self);
        self.ke_op5_l6.setText("6. Momentum decay:");
        self.ke_op5_l6.move(20, 350);
        tmp.append(self.ke_op5_l6);

        self.ke_op5_e6 = QLineEdit(self)
        self.ke_op5_e6.move(190, 350);
        self.ke_op5_e6.setText("0.004");
        tmp.append(self.ke_op5_e6);

        self.ke_op5_l7 = QLabel(self);
        self.ke_op5_l7.setText("7. Epsilon:");
        self.ke_op5_l7.move(20, 400);
        tmp.append(self.ke_op5_l7);

        self.ke_op5_e7 = QLineEdit(self)
        self.ke_op5_e7.move(150, 400);
        self.ke_op5_e7.setText("0.00000001");
        tmp.append(self.ke_op5_e7);

        self.optimizer_ui_keras.append(tmp)





        tmp = [];
        self.ke_op6_l1 = QLabel(self);
        self.ke_op6_l1.setText("1. Learning Rate:");
        self.ke_op6_l1.move(20, 100);
        tmp.append(self.ke_op6_l1);

        self.ke_op6_e1 = QLineEdit(self)
        self.ke_op6_e1.move(150, 100);
        self.ke_op6_e1.setText("0.001");
        tmp.append(self.ke_op6_e1);


        self.ke_op6_l2 = QLabel(self);
        self.ke_op6_l2.setText("2. beta1:");
        self.ke_op6_l2.move(20, 150);
        tmp.append(self.ke_op6_l2);

        self.ke_op6_e2 = QLineEdit(self)
        self.ke_op6_e2.move(150, 150);
        self.ke_op6_e2.setText("0.9");
        tmp.append(self.ke_op6_e2);


        self.ke_op6_l3 = QLabel(self);
        self.ke_op6_l3.setText("3. beta2:");
        self.ke_op6_l3.move(20, 200);
        tmp.append(self.ke_op6_l3);

        self.ke_op6_e3 = QLineEdit(self)
        self.ke_op6_e3.move(150, 200);
        self.ke_op6_e3.setText("0.999");
        tmp.append(self.ke_op6_e3);


        self.ke_op6_l4 = QLabel(self);
        self.ke_op6_l4.setText("4. weight_decay:");
        self.ke_op6_l4.move(20, 250);
        tmp.append(self.ke_op6_l4);

        self.ke_op6_e4 = QLineEdit(self)
        self.ke_op6_e4.move(150, 250);
        self.ke_op6_e4.setText("0.0");
        tmp.append(self.ke_op6_e4);


        self.ke_op6_l5 = QLabel(self);
        self.ke_op6_l5.setText("5. Epsilon:");
        self.ke_op6_l5.move(20, 300);
        tmp.append(self.ke_op6_l5);

        self.ke_op6_e5 = QLineEdit(self)
        self.ke_op6_e5.move(150, 300);
        self.ke_op6_e5.setText("0.00000001");
        tmp.append(self.ke_op6_e5);

        self.optimizer_ui_keras.append(tmp)





        tmp = [];
        self.ke_op7_l1 = QLabel(self);
        self.ke_op7_l1.setText("1. Learning Rate:");
        self.ke_op7_l1.move(20, 100);
        tmp.append(self.ke_op7_l1);

        self.ke_op7_e1 = QLineEdit(self)
        self.ke_op7_e1.move(150, 100);
        self.ke_op7_e1.setText("0.001");
        tmp.append(self.ke_op7_e1);


        self.ke_op7_l2 = QLabel(self);
        self.ke_op7_l2.setText("2. Learning rate decay:");
        self.ke_op7_l2.move(20, 150);
        tmp.append(self.ke_op7_l2);

        self.ke_op7_e2 = QLineEdit(self)
        self.ke_op7_e2.move(190, 150);
        self.ke_op7_e2.setText("0.0");
        tmp.append(self.ke_op7_e2);


        self.ke_op7_l3 = QLabel(self);
        self.ke_op7_l3.setText("3. Weight decay:");
        self.ke_op7_l3.move(20, 200);
        tmp.append(self.ke_op7_l3);

        self.ke_op7_e3 = QLineEdit(self)
        self.ke_op7_e3.move(150, 200);
        self.ke_op7_e3.setText("0.0");
        tmp.append(self.ke_op7_e3);

        self.ke_op7_l4 = QLabel(self);
        self.ke_op7_l4.setText("4. Epsilon:");
        self.ke_op7_l4.move(20, 250);
        tmp.append(self.ke_op7_l4);

        self.ke_op7_e4 = QLineEdit(self)
        self.ke_op7_e4.move(150, 250);
        self.ke_op7_e4.setText("0.000000001");
        tmp.append(self.ke_op7_e4);

        self.optimizer_ui_keras.append(tmp)





        tmp = [];
        self.ke_op8_l1 = QLabel(self);
        self.ke_op8_l1.setText("1. Learning Rate:");
        self.ke_op8_l1.move(20, 100);
        tmp.append(self.ke_op8_l1);

        self.ke_op8_e1 = QLineEdit(self)
        self.ke_op8_e1.move(150, 100);
        self.ke_op8_e1.setText("0.001");
        tmp.append(self.ke_op8_e1);


        self.ke_op8_l2 = QLabel(self);
        self.ke_op8_l2.setText("2. Rho param:");
        self.ke_op8_l2.move(20, 150);
        tmp.append(self.ke_op8_l2);

        self.ke_op8_e2 = QLineEdit(self)
        self.ke_op8_e2.move(150, 150);
        self.ke_op8_e2.setText("0.9");
        tmp.append(self.ke_op8_e2);


        self.ke_op8_l3 = QLabel(self);
        self.ke_op8_l3.setText("3. Weight decay:");
        self.ke_op8_l3.move(20, 200);
        tmp.append(self.ke_op8_l3);

        self.ke_op8_e3 = QLineEdit(self)
        self.ke_op8_e3.move(150, 200);
        self.ke_op8_e3.setText("0.0");
        tmp.append(self.ke_op8_e3);

        self.ke_op8_l4 = QLabel(self);
        self.ke_op8_l4.setText("4. Epsilon:");
        self.ke_op8_l4.move(20, 250);
        tmp.append(self.ke_op8_l4);

        self.ke_op8_e4 = QLineEdit(self)
        self.ke_op8_e4.move(150, 250);
        self.ke_op8_e4.setText("0.000000001");
        tmp.append(self.ke_op8_e4);

        self.optimizer_ui_keras.append(tmp)







        tmp = [];
        self.py_op1_l1 = QLabel(self);
        self.py_op1_l1.setText("1. Learning Rate:");
        self.py_op1_l1.move(20, 100);
        tmp.append(self.py_op1_l1);

        self.py_op1_e1 = QLineEdit(self)
        self.py_op1_e1.move(150, 100);
        self.py_op1_e1.setText("0.001");
        tmp.append(self.py_op1_e1);


        self.py_op1_l2 = QLabel(self);
        self.py_op1_l2.setText("2. Momentum:");
        self.py_op1_l2.move(20, 150);
        tmp.append(self.py_op1_l2);

        self.py_op1_e2 = QLineEdit(self)
        self.py_op1_e2.move(150, 150);
        self.py_op1_e2.setText("0.9");
        tmp.append(self.py_op1_e2);


        self.py_op1_l3 = QLabel(self);
        self.py_op1_l3.setText("3. Weight decay:");
        self.py_op1_l3.move(20, 200);
        tmp.append(self.py_op1_l3);

        self.py_op1_e3 = QLineEdit(self)
        self.py_op1_e3.move(150, 200);
        self.py_op1_e3.setText("0.0");
        tmp.append(self.py_op1_e3);

        self.optimizer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_op2_l1 = QLabel(self);
        self.py_op2_l1.setText("1. Learning Rate:");
        self.py_op2_l1.move(20, 100);
        tmp.append(self.py_op2_l1);

        self.py_op2_e1 = QLineEdit(self)
        self.py_op2_e1.move(150, 100);
        self.py_op2_e1.setText("0.001");
        tmp.append(self.py_op2_e1);


        self.py_op2_l2 = QLabel(self);
        self.py_op2_l2.setText("2. Momentum:");
        self.py_op2_l2.move(20, 150);
        tmp.append(self.py_op2_l2);

        self.py_op2_e2 = QLineEdit(self)
        self.py_op2_e2.move(150, 150);
        self.py_op2_e2.setText("0.9");
        tmp.append(self.py_op2_e2);


        self.py_op2_l3 = QLabel(self);
        self.py_op2_l3.setText("3. Weight decay:");
        self.py_op2_l3.move(20, 200);
        tmp.append(self.py_op2_l3);

        self.py_op2_e3 = QLineEdit(self)
        self.py_op2_e3.move(150, 200);
        self.py_op2_e3.setText("0.0");
        tmp.append(self.py_op2_e3);

        self.optimizer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_op3_l1 = QLabel(self);
        self.py_op3_l1.setText("1. Learning Rate:");
        self.py_op3_l1.move(20, 100);
        tmp.append(self.py_op3_l1);

        self.py_op3_e1 = QLineEdit(self)
        self.py_op3_e1.move(150, 100);
        self.py_op3_e1.setText("0.001");
        tmp.append(self.py_op3_e1);


        self.py_op3_l2 = QLabel(self);
        self.py_op3_l2.setText("2. Decay rate:");
        self.py_op3_l2.move(20, 150);
        tmp.append(self.py_op3_l2);

        self.py_op3_e2 = QLineEdit(self)
        self.py_op3_e2.move(150, 150);
        self.py_op3_e2.setText("0.99");
        tmp.append(self.py_op3_e2);


        self.py_op3_l3 = QLabel(self);
        self.py_op3_l3.setText("3. Weight decay:");
        self.py_op3_l3.move(20, 200);
        tmp.append(self.py_op3_l3);

        self.py_op3_e3 = QLineEdit(self)
        self.py_op3_e3.move(150, 200);
        self.py_op3_e3.setText("0.0");
        tmp.append(self.py_op3_e3);

        self.py_op3_l4 = QLabel(self);
        self.py_op3_l4.setText("4. Epsilon:");
        self.py_op3_l4.move(20, 250);
        tmp.append(self.py_op3_l4);

        self.py_op3_e4 = QLineEdit(self)
        self.py_op3_e4.move(150, 250);
        self.py_op3_e4.setText("0.000000001");
        tmp.append(self.py_op3_e4);

        self.optimizer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_op4_l1 = QLabel(self);
        self.py_op4_l1.setText("1. Learning Rate:");
        self.py_op4_l1.move(20, 100);
        tmp.append(self.py_op4_l1);

        self.py_op4_e1 = QLineEdit(self)
        self.py_op4_e1.move(150, 100);
        self.py_op4_e1.setText("0.001");
        tmp.append(self.py_op4_e1);


        self.py_op4_l2 = QLabel(self);
        self.py_op4_l2.setText("2. Decay rate:");
        self.py_op4_l2.move(20, 150);
        tmp.append(self.py_op4_l2);

        self.py_op4_e2 = QLineEdit(self)
        self.py_op4_e2.move(150, 150);
        self.py_op4_e2.setText("0.99");
        tmp.append(self.py_op4_e2);


        self.py_op4_l3 = QLabel(self);
        self.py_op4_l3.setText("3. Weight decay:");
        self.py_op4_l3.move(20, 200);
        tmp.append(self.py_op4_l3);

        self.py_op4_e3 = QLineEdit(self)
        self.py_op4_e3.move(150, 200);
        self.py_op4_e3.setText("0.0");
        tmp.append(self.py_op4_e3);

        self.py_op4_l4 = QLabel(self);
        self.py_op4_l4.setText("4. Epsilon:");
        self.py_op4_l4.move(20, 250);
        tmp.append(self.py_op4_l4);

        self.py_op4_e4 = QLineEdit(self)
        self.py_op4_e4.move(150, 250);
        self.py_op4_e4.setText("0.000000001");
        tmp.append(self.py_op4_e4);

        self.optimizer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_op5_l1 = QLabel(self);
        self.py_op5_l1.setText("1. Learning Rate:");
        self.py_op5_l1.move(20, 100);
        tmp.append(self.py_op5_l1);

        self.py_op5_e1 = QLineEdit(self)
        self.py_op5_e1.move(150, 100);
        self.py_op5_e1.setText("0.001");
        tmp.append(self.py_op5_e1);


        self.py_op5_l2 = QLabel(self);
        self.py_op5_l2.setText("2. beta1:");
        self.py_op5_l2.move(20, 150);
        tmp.append(self.py_op5_l2);

        self.py_op5_e2 = QLineEdit(self)
        self.py_op5_e2.move(150, 150);
        self.py_op5_e2.setText("0.9");
        tmp.append(self.py_op5_e2);


        self.py_op5_l3 = QLabel(self);
        self.py_op5_l3.setText("3. beta2:");
        self.py_op5_l3.move(20, 200);
        tmp.append(self.py_op5_l3);

        self.py_op5_e3 = QLineEdit(self)
        self.py_op5_e3.move(150, 200);
        self.py_op5_e3.setText("0.999");
        tmp.append(self.py_op5_e3);


        self.py_op5_l4 = QLabel(self);
        self.py_op5_l4.setText("4. weight_decay:");
        self.py_op5_l4.move(20, 250);
        tmp.append(self.py_op5_l4);

        self.py_op5_e4 = QLineEdit(self)
        self.py_op5_e4.move(150, 250);
        self.py_op5_e4.setText("0.0");
        tmp.append(self.py_op5_e4);


        self.py_op5_l5 = QLabel(self);
        self.py_op5_l5.setText("5. Epsilon:");
        self.py_op5_l5.move(20, 300);
        tmp.append(self.py_op5_l5);

        self.py_op5_e5 = QLineEdit(self)
        self.py_op5_e5.move(150, 300);
        self.py_op5_e5.setText("0.00000001");
        tmp.append(self.py_op5_e5);

        self.optimizer_ui_pytorch.append(tmp)




        tmp = [];
        self.py_op6_l1 = QLabel(self);
        self.py_op6_l1.setText("1. Learning Rate:");
        self.py_op6_l1.move(20, 100);
        tmp.append(self.py_op6_l1);

        self.py_op6_e1 = QLineEdit(self)
        self.py_op6_e1.move(150, 100);
        self.py_op6_e1.setText("0.001");
        tmp.append(self.py_op6_e1);


        self.py_op6_l2 = QLabel(self);
        self.py_op6_l2.setText("2. beta1:");
        self.py_op6_l2.move(20, 150);
        tmp.append(self.py_op6_l2);

        self.py_op6_e2 = QLineEdit(self)
        self.py_op6_e2.move(150, 150);
        self.py_op6_e2.setText("0.9");
        tmp.append(self.py_op6_e2);


        self.py_op6_l3 = QLabel(self);
        self.py_op6_l3.setText("3. beta2:");
        self.py_op6_l3.move(20, 200);
        tmp.append(self.py_op6_l3);

        self.py_op6_e3 = QLineEdit(self)
        self.py_op6_e3.move(150, 200);
        self.py_op6_e3.setText("0.999");
        tmp.append(self.py_op6_e3);


        self.py_op6_l4 = QLabel(self);
        self.py_op6_l4.setText("4. weight_decay:");
        self.py_op6_l4.move(20, 250);
        tmp.append(self.py_op6_l4);

        self.py_op6_e4 = QLineEdit(self)
        self.py_op6_e4.move(150, 250);
        self.py_op6_e4.setText("0.0");
        tmp.append(self.py_op6_e4);


        self.py_op6_l5 = QLabel(self);
        self.py_op6_l5.setText("5. Epsilon:");
        self.py_op6_l5.move(20, 300);
        tmp.append(self.py_op6_l5);

        self.py_op6_e5 = QLineEdit(self)
        self.py_op6_e5.move(150, 300);
        self.py_op6_e5.setText("0.00000001");
        tmp.append(self.py_op6_e5);

        self.optimizer_ui_pytorch.append(tmp)






        tmp = [];
        self.py_op7_l1 = QLabel(self);
        self.py_op7_l1.setText("1. Learning Rate:");
        self.py_op7_l1.move(20, 100);
        tmp.append(self.py_op7_l1);

        self.py_op7_e1 = QLineEdit(self)
        self.py_op7_e1.move(150, 100);
        self.py_op7_e1.setText("0.001");
        tmp.append(self.py_op7_e1);


        self.py_op7_l2 = QLabel(self);
        self.py_op7_l2.setText("2. beta1:");
        self.py_op7_l2.move(20, 150);
        tmp.append(self.py_op7_l2);

        self.py_op7_e2 = QLineEdit(self)
        self.py_op7_e2.move(150, 150);
        self.py_op7_e2.setText("0.9");
        tmp.append(self.py_op7_e2);


        self.py_op7_l3 = QLabel(self);
        self.py_op7_l3.setText("3. beta2:");
        self.py_op7_l3.move(20, 200);
        tmp.append(self.py_op7_l3);

        self.py_op7_e3 = QLineEdit(self)
        self.py_op7_e3.move(150, 200);
        self.py_op7_e3.setText("0.999");
        tmp.append(self.py_op7_e3);


        self.py_op7_l4 = QLabel(self);
        self.py_op7_l4.setText("4. weight_decay:");
        self.py_op7_l4.move(20, 250);
        tmp.append(self.py_op7_l4);

        self.py_op7_e4 = QLineEdit(self)
        self.py_op7_e4.move(150, 250);
        self.py_op7_e4.setText("0.0");
        tmp.append(self.py_op7_e4);


        self.py_op7_l5 = QLabel(self);
        self.py_op7_l5.setText("5. Apply amsgrad:");
        self.py_op7_l5.move(20, 300);
        tmp.append(self.py_op7_l5);

        self.mx_op7_cb5 = QComboBox(self);
        self.mx_op7_cb5.move(200, 300);
        self.mx_op7_cb5.addItems(["No", "Yes"]);
        tmp.append(self.mx_op7_cb5);

        self.optimizer_ui_pytorch.append(tmp)




        tmp = [];
        self.py_op8_l1 = QLabel(self);
        self.py_op8_l1.setText("1. Learning Rate:");
        self.py_op8_l1.move(20, 100);
        tmp.append(self.py_op8_l1);

        self.py_op8_e1 = QLineEdit(self)
        self.py_op8_e1.move(150, 100);
        self.py_op8_e1.setText("0.001");
        tmp.append(self.py_op8_e1);


        self.py_op8_l2 = QLabel(self);
        self.py_op8_l2.setText("2. Learning rate decay:");
        self.py_op8_l2.move(20, 150);
        tmp.append(self.py_op8_l2);

        self.py_op8_e2 = QLineEdit(self)
        self.py_op8_e2.move(150, 150);
        self.py_op8_e2.setText("0.0");
        tmp.append(self.py_op8_e2);


        self.py_op8_l3 = QLabel(self);
        self.py_op8_l3.setText("3. Weight decay:");
        self.py_op8_l3.move(20, 200);
        tmp.append(self.py_op8_l3);

        self.py_op8_e3 = QLineEdit(self)
        self.py_op8_e3.move(150, 200);
        self.py_op8_e3.setText("0.0");
        tmp.append(self.py_op8_e3);

        self.py_op8_l4 = QLabel(self);
        self.py_op8_l4.setText("4. Epsilon:");
        self.py_op8_l4.move(20, 250);
        tmp.append(self.py_op8_l4);

        self.py_op8_e4 = QLineEdit(self)
        self.py_op8_e4.move(150, 250);
        self.py_op8_e4.setText("0.000000001");
        tmp.append(self.py_op8_e4);

        self.optimizer_ui_pytorch.append(tmp)





        tmp = [];
        self.py_op9_l1 = QLabel(self);
        self.py_op9_l1.setText("1. Learning Rate:");
        self.py_op9_l1.move(20, 100);
        tmp.append(self.py_op9_l1);

        self.py_op9_e1 = QLineEdit(self)
        self.py_op9_e1.move(150, 100);
        self.py_op9_e1.setText("0.001");
        tmp.append(self.py_op9_e1);


        self.py_op9_l2 = QLabel(self);
        self.py_op9_l2.setText("2. Rho param:");
        self.py_op9_l2.move(20, 150);
        tmp.append(self.py_op9_l2);

        self.py_op9_e2 = QLineEdit(self)
        self.py_op9_e2.move(150, 150);
        self.py_op9_e2.setText("0.9");
        tmp.append(self.py_op9_e2);


        self.py_op9_l3 = QLabel(self);
        self.py_op9_l3.setText("3. Weight decay:");
        self.py_op9_l3.move(20, 200);
        tmp.append(self.py_op9_l3);

        self.py_op9_e3 = QLineEdit(self)
        self.py_op9_e3.move(150, 200);
        self.py_op9_e3.setText("0.0");
        tmp.append(self.py_op9_e3);

        self.py_op9_l4 = QLabel(self);
        self.py_op9_l4.setText("4. Epsilon:");
        self.py_op9_l4.move(20, 250);
        tmp.append(self.py_op9_l4);

        self.py_op9_e4 = QLineEdit(self)
        self.py_op9_e4.move(150, 250);
        self.py_op9_e4.setText("0.000000001");
        tmp.append(self.py_op9_e4);

        self.optimizer_ui_pytorch.append(tmp)








        self.select_optimizer();

        self.tb1 = QTextEdit(self)
        self.tb1.move(550, 20)
        self.tb1.resize(300, 500)
        self.tb1.setText("Using Default SGD optimizer.")


        self.b4 = QPushButton('Select Optimizer', self)
        self.b4.move(400,400)
        self.b4.clicked.connect(self.add_optimizer)

        
        self.b6 = QPushButton('Clear ', self)
        self.b6.move(400,500)
        self.b6.clicked.connect(self.clear_optimizer)










    def select_optimizer(self):
        self.current_optimizer = {};
        self.current_optimizer["name"] = "";
        self.current_optimizer["params"] = {};

        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.current_optimizer["name"] = self.cb1.currentText();
            index = self.mxnet_optimizers_list.index(self.cb1.currentText());
            for i in range(len(self.optimizer_ui_mxnet)):
                for j in range(len(self.optimizer_ui_mxnet[i])):
                    if((index-1)==i):
                        self.optimizer_ui_mxnet[i][j].show();
                    else:
                        self.optimizer_ui_mxnet[i][j].hide();

            for i in range(len(self.optimizer_ui_keras)):
                for j in range(len(self.optimizer_ui_keras[i])):
                    self.optimizer_ui_keras[i][j].hide();
            for i in range(len(self.optimizer_ui_pytorch)):
                for j in range(len(self.optimizer_ui_pytorch[i])):
                    self.optimizer_ui_pytorch[i][j].hide();
            


        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.current_optimizer["name"] = self.cb2.currentText();
            index = self.keras_optimizers_list.index(self.cb2.currentText());
            for i in range(len(self.optimizer_ui_keras)):
                for j in range(len(self.optimizer_ui_keras[i])):
                    if((index-1)==i):
                        self.optimizer_ui_keras[i][j].show();
                    else:
                        self.optimizer_ui_keras[i][j].hide();

            for i in range(len(self.optimizer_ui_mxnet)):
                for j in range(len(self.optimizer_ui_mxnet[i])):
                    self.optimizer_ui_mxnet[i][j].hide();
            for i in range(len(self.optimizer_ui_pytorch)):
                for j in range(len(self.optimizer_ui_pytorch[i])):
                    self.optimizer_ui_pytorch[i][j].hide();



        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.current_optimizer["name"] = self.cb3.currentText();
            index = self.pytorch_optimizers_list.index(self.cb3.currentText());
            for i in range(len(self.optimizer_ui_pytorch)):
                for j in range(len(self.optimizer_ui_pytorch[i])):
                    if((index-1)==i):
                        self.optimizer_ui_pytorch[i][j].show();
                    else:
                        self.optimizer_ui_pytorch[i][j].hide();

            for i in range(len(self.optimizer_ui_keras)):
                for j in range(len(self.optimizer_ui_keras[i])):
                    self.optimizer_ui_keras[i][j].hide();
            for i in range(len(self.optimizer_ui_mxnet)):
                for j in range(len(self.optimizer_ui_mxnet[i])):
                    self.optimizer_ui_mxnet[i][j].hide();




    def add_optimizer(self):
        self.system["update"]["optimizers"]["active"] = True;
        if(self.system["backend"] == "Mxnet-1.5.1"):
            if(self.current_optimizer["name"] == self.mxnet_optimizers_list[1]):
                self.current_optimizer["params"]["learning_rate"] = self.mx_op1_e1.text();
                self.current_optimizer["params"]["momentum"] = self.mx_op1_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.mx_op1_e3.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.mxnet_optimizers_list[2]):
                self.current_optimizer["params"]["learning_rate"] = self.mx_op2_e1.text();
                self.current_optimizer["params"]["momentum"] = self.mx_op2_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.mx_op2_e3.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.mxnet_optimizers_list[3]):
                self.current_optimizer["params"]["learning_rate"] = self.mx_op3_e1.text();
                self.current_optimizer["params"]["decay_rate"] = self.mx_op3_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.mx_op3_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.mx_op3_e4.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.mxnet_optimizers_list[4]):
                self.current_optimizer["params"]["learning_rate"] = self.mx_op4_e1.text();
                self.current_optimizer["params"]["decay_rate"] = self.mx_op4_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.mx_op4_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.mx_op4_e3.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.mxnet_optimizers_list[5]):
                self.current_optimizer["params"]["learning_rate"] = self.mx_op5_e1.text();
                self.current_optimizer["params"]["beta1"] = self.mx_op5_e2.text();
                self.current_optimizer["params"]["beta2"] = self.mx_op5_e3.text();
                self.current_optimizer["params"]["weight_decay"] = self.mx_op5_e4.text();
                self.current_optimizer["params"]["epsilon"] = self.mx_op5_e5.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.mxnet_optimizers_list[6]):
                self.current_optimizer["params"]["learning_rate"] = self.mx_op6_e1.text();
                self.current_optimizer["params"]["beta1"] = self.mx_op6_e2.text();
                self.current_optimizer["params"]["beta2"] = self.mx_op6_e3.text();
                self.current_optimizer["params"]["weight_decay"] = self.mx_op6_e4.text();
                self.current_optimizer["params"]["epsilon"] = self.mx_op6_e5.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.mxnet_optimizers_list[7]):
                self.current_optimizer["params"]["learning_rate"] = self.mx_op7_e1.text();
                self.current_optimizer["params"]["beta1"] = self.mx_op7_e2.text();
                self.current_optimizer["params"]["beta2"] = self.mx_op7_e3.text();
                self.current_optimizer["params"]["weight_decay"] = self.mx_op7_e4.text();
                self.current_optimizer["params"]["amsgrad"] = self.mx_op7_cb5.currentText();
                self.current_optimizer["params"]["momentum_decay"] = self.mx_op7_e6.text();
                self.current_optimizer["params"]["epsilon"] = self.mx_op7_e7.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.mxnet_optimizers_list[8]):
                self.current_optimizer["params"]["learning_rate"] = self.mx_op8_e1.text();
                self.current_optimizer["params"]["learning_rate_decay"] = self.mx_op8_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.mx_op8_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.mx_op8_e4.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.mxnet_optimizers_list[9]):
                self.current_optimizer["params"]["learning_rate"] = self.mx_op9_e1.text();
                self.current_optimizer["params"]["rho"] = self.mx_op9_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.mx_op9_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.mx_op9_e4.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;



        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            if(self.current_optimizer["name"] == self.keras_optimizers_list[1]):
                self.current_optimizer["params"]["learning_rate"] = self.ke_op1_e1.text();
                self.current_optimizer["params"]["momentum"] = self.ke_op1_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.ke_op1_e3.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.keras_optimizers_list[2]):
                self.current_optimizer["params"]["learning_rate"] = self.ke_op2_e1.text();
                self.current_optimizer["params"]["momentum"] = self.ke_op2_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.ke_op2_e3.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.keras_optimizers_list[3]):
                self.current_optimizer["params"]["learning_rate"] = self.ke_op3_e1.text();
                self.current_optimizer["params"]["decay_rate"] = self.ke_op3_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.ke_op3_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.ke_op3_e4.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.keras_optimizers_list[4]):
                self.current_optimizer["params"]["learning_rate"] = self.ke_op4_e1.text();
                self.current_optimizer["params"]["beta1"] = self.ke_op4_e2.text();
                self.current_optimizer["params"]["beta2"] = self.ke_op4_e3.text();
                self.current_optimizer["params"]["weight_decay"] = self.ke_op4_e4.text();
                self.current_optimizer["params"]["epsilon"] = self.ke_op4_e5.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.keras_optimizers_list[5]):
                self.current_optimizer["params"]["learning_rate"] = self.ke_op5_e1.text();
                self.current_optimizer["params"]["beta1"] = self.ke_op5_e2.text();
                self.current_optimizer["params"]["beta2"] = self.ke_op5_e3.text();
                self.current_optimizer["params"]["weight_decay"] = self.ke_op5_e4.text();
                self.current_optimizer["params"]["amsgrad"] = self.ke_op5_cb5.currentText();
                self.current_optimizer["params"]["momentum_decay"] = self.ke_op5_e6.text();
                self.current_optimizer["params"]["epsilon"] = self.ke_op5_e7.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.keras_optimizers_list[6]):
                self.current_optimizer["params"]["learning_rate"] = self.ke_op6_e1.text();
                self.current_optimizer["params"]["beta1"] = self.ke_op6_e2.text();
                self.current_optimizer["params"]["beta2"] = self.ke_op6_e3.text();
                self.current_optimizer["params"]["weight_decay"] = self.ke_op6_e4.text();
                self.current_optimizer["params"]["epsilon"] = self.ke_op6_e5.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.keras_optimizers_list[7]):
                self.current_optimizer["params"]["learning_rate"] = self.ke_op7_e1.text();
                self.current_optimizer["params"]["learning_rate_decay"] = self.ke_op7_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.ke_op7_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.ke_op7_e4.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.keras_optimizers_list[8]):
                self.current_optimizer["params"]["learning_rate"] = self.ke_op8_e1.text();
                self.current_optimizer["params"]["rho"] = self.ke_op8_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.ke_op8_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.ke_op8_e4.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;



        elif(self.system["backend"] == "Pytorch-1.3.1"):
            if(self.current_optimizer["name"] == self.pytorch_optimizers_list[1]):
                self.current_optimizer["params"]["learning_rate"] = self.py_op1_e1.text();
                self.current_optimizer["params"]["momentum"] = self.py_op1_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.py_op1_e3.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.pytorch_optimizers_list[2]):
                self.current_optimizer["params"]["learning_rate"] = self.py_op2_e1.text();
                self.current_optimizer["params"]["momentum"] = self.py_op2_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.py_op2_e3.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.pytorch_optimizers_list[3]):
                self.current_optimizer["params"]["learning_rate"] = self.py_op3_e1.text();
                self.current_optimizer["params"]["decay_rate"] = self.py_op3_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.py_op3_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.py_op3_e4.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.pytorch_optimizers_list[4]):
                self.current_optimizer["params"]["learning_rate"] = self.py_op4_e1.text();
                self.current_optimizer["params"]["decay_rate"] = self.py_op4_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.py_op4_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.py_op4_e4.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.pytorch_optimizers_list[5]):
                self.current_optimizer["params"]["learning_rate"] = self.py_op5_e1.text();
                self.current_optimizer["params"]["beta1"] = self.py_op5_e2.text();
                self.current_optimizer["params"]["beta2"] = self.py_op5_e3.text();
                self.current_optimizer["params"]["weight_decay"] = self.py_op5_e4.text();
                self.current_optimizer["params"]["epsilon"] = self.py_op5_e5.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.pytorch_optimizers_list[6]):
                self.current_optimizer["params"]["learning_rate"] = self.py_op6_e1.text();
                self.current_optimizer["params"]["beta1"] = self.py_op6_e2.text();
                self.current_optimizer["params"]["beta2"] = self.py_op6_e3.text();
                self.current_optimizer["params"]["weight_decay"] = self.py_op6_e4.text();
                self.current_optimizer["params"]["epsilon"] = self.py_op6_e5.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.pytorch_optimizers_list[7]):
                self.current_optimizer["params"]["learning_rate"] = self.py_op7_e1.text();
                self.current_optimizer["params"]["beta1"] = self.py_op7_e2.text();
                self.current_optimizer["params"]["beta2"] = self.py_op7_e3.text();
                self.current_optimizer["params"]["weight_decay"] = self.py_op7_e4.text();
                self.current_optimizer["params"]["amsgrad"] = self.mx_op7_cb5.currentText();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.pytorch_optimizers_list[8]):
                self.current_optimizer["params"]["learning_rate"] = self.py_op8_e1.text();
                self.current_optimizer["params"]["learning_rate_decay"] = self.py_op8_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.py_op8_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.py_op8_e4.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;

            elif(self.current_optimizer["name"] == self.pytorch_optimizers_list[9]):
                self.current_optimizer["params"]["learning_rate"] = self.py_op9_e1.text();
                self.current_optimizer["params"]["rho"] = self.py_op9_e2.text();
                self.current_optimizer["params"]["weight_decay"] = self.py_op9_e3.text();
                self.current_optimizer["params"]["epsilon"] = self.py_op9_e4.text();
                self.system["update"]["optimizers"]["value"] = self.current_optimizer;





        wr = "";
        wr = json.dumps(self.system["update"]["optimizers"]["value"], indent=4)
        self.tb1.setText(wr);


    def clear_optimizer(self):
        self.system["update"]["optimizers"]["value"] = "";
        self.system["update"]["optimizers"]["active"] = False;

        wr = "";
        self.tb1.setText(wr);



    def analyse_optimizer_lr(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_analyse_optimizer_lr.emit();


    def forward(self):        
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_scheduler_param.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_train_param.emit();



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateOptimizerParam()
screen.show()
sys.exit(app.exec_())
'''