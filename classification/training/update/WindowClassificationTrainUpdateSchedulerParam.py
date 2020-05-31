import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateSchedulerParam(QtWidgets.QWidget):

    forward_loss_param = QtCore.pyqtSignal();
    backward_optimizer_param = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Update Scheduler Params'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 600
        self.scheduler_ui_mxnet = [];
        self.scheduler_ui_keras = [];
        self.scheduler_ui_pytorch = [];
        self.current_scheduler = {};
        self.current_scheduler["name"] = "";
        self.current_scheduler["params"] = {};
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
        self.cb1.activated.connect(self.select_scheduler);

        self.cb2 = QComboBox(self);
        self.cb2.move(20, 20);
        self.cb2.activated.connect(self.select_scheduler);

        self.cb3 = QComboBox(self);
        self.cb3.move(20, 20);
        self.cb3.activated.connect(self.select_scheduler);


        self.mxnet_schedulers_list = ["select", "lr_fixed", "lr_step_decrease", "lr_multistep_decrease"];

        self.keras_schedulers_list = ["select", "lr_fixed", "lr_step_decrease", "lr_exponential_decrease", "lr_plateau_decrease"];

        self.pytorch_schedulers_list = ["select", "lr_fixed", "lr_step_decrease", "lr_multistep_decrease", "lr_exponential_decrease", 
                                        "lr_plateau_decrease"];


        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.cb1.addItems(self.mxnet_schedulers_list);
            self.cb1.show();
            self.cb2.hide();
            self.cb3.hide();
        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.cb2.addItems(self.keras_schedulers_list);
            self.cb2.show();
            self.cb1.hide();
            self.cb3.hide();
        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.cb3.addItems(self.pytorch_schedulers_list);
            self.cb3.show();
            self.cb1.hide();
            self.cb2.hide();






        tmp = [];
        self.mx_sc1_l1 = QLabel(self);
        self.mx_sc1_l1.setText("No parameters (Arguments) to set.");
        self.mx_sc1_l1.move(20, 100);
        tmp.append(self.mx_sc1_l1);

        self.scheduler_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_sc2_l1 = QLabel(self);
        self.mx_sc2_l1.setText("1. Step Size: ");
        self.mx_sc2_l1.move(20, 100);
        tmp.append(self.mx_sc2_l1);

        self.mx_sc2_e1 = QLineEdit(self)
        self.mx_sc2_e1.move(150, 100);
        self.mx_sc2_e1.setText("5");
        tmp.append(self.mx_sc2_e1);

        self.mx_sc2_l2 = QLabel(self);
        self.mx_sc2_l2.setText("2. Learning rate multiplicative factor: ");
        self.mx_sc2_l2.move(20, 150);
        tmp.append(self.mx_sc2_l2);

        self.mx_sc2_e2 = QLineEdit(self)
        self.mx_sc2_e2.move(290, 150);
        self.mx_sc2_e2.setText("0.1");
        tmp.append(self.mx_sc2_e2);

        self.scheduler_ui_mxnet.append(tmp)




        tmp = [];
        self.mx_sc3_l1 = QLabel(self);
        self.mx_sc3_l1.setText("1. Milestones: ");
        self.mx_sc3_l1.move(20, 100);
        tmp.append(self.mx_sc3_l1);

        self.mx_sc3_e1 = QLineEdit(self)
        self.mx_sc3_e1.move(150, 100);
        self.mx_sc3_e1.setText("5, 10, 15");
        tmp.append(self.mx_sc3_e1);

        self.mx_sc3_l2 = QLabel(self);
        self.mx_sc3_l2.setText("2. Learning rate multiplicative factor: ");
        self.mx_sc3_l2.move(20, 150);
        tmp.append(self.mx_sc3_l2);

        self.mx_sc3_e2 = QLineEdit(self)
        self.mx_sc3_e2.move(290, 150);
        self.mx_sc3_e2.setText("0.1");
        tmp.append(self.mx_sc3_e2);

        self.scheduler_ui_mxnet.append(tmp)








        tmp = [];
        self.ke_sc1_l1 = QLabel(self);
        self.ke_sc1_l1.setText("No parameters (Arguments) to set.");
        self.ke_sc1_l1.move(20, 100);
        tmp.append(self.ke_sc1_l1);

        self.scheduler_ui_keras.append(tmp)



        tmp = [];
        self.ke_sc2_l1 = QLabel(self);
        self.ke_sc2_l1.setText("1. Step Size: ");
        self.ke_sc2_l1.move(20, 100);
        tmp.append(self.ke_sc2_l1);

        self.ke_sc2_e1 = QLineEdit(self)
        self.ke_sc2_e1.move(150, 100);
        self.ke_sc2_e1.setText("5");
        tmp.append(self.ke_sc2_e1);

        self.ke_sc2_l2 = QLabel(self);
        self.ke_sc2_l2.setText("2. Learning rate multiplicative factor: ");
        self.ke_sc2_l2.move(20, 150);
        tmp.append(self.ke_sc2_l2);

        self.ke_sc2_e2 = QLineEdit(self)
        self.ke_sc2_e2.move(290, 150);
        self.ke_sc2_e2.setText("0.1");
        tmp.append(self.ke_sc2_e2);

        self.scheduler_ui_keras.append(tmp)



        tmp = [];
        self.ke_sc3_l1 = QLabel(self);
        self.ke_sc3_l1.setText("1. Learning rate multiplicative factor: ");
        self.ke_sc3_l1.move(20, 100);
        tmp.append(self.ke_sc3_l1);

        self.ke_sc3_e1 = QLineEdit(self)
        self.ke_sc3_e1.move(290, 100);
        self.ke_sc3_e1.setText("0.9");
        tmp.append(self.ke_sc3_e1);

        self.scheduler_ui_keras.append(tmp)






        tmp = [];
        self.ke_sc4_l1 = QLabel(self);
        self.ke_sc4_l1.setText("1. Mode: ");
        self.ke_sc4_l1.move(20, 100);
        tmp.append(self.ke_sc4_l1);

        self.ke_sc4_cb1 = QComboBox(self);
        self.ke_sc4_cb1.move(200, 100);
        self.ke_sc4_cb1.addItems(["Min", "Max"]);
        tmp.append(self.ke_sc4_cb1);

        self.ke_sc4_l2 = QLabel(self);
        self.ke_sc4_l2.setText("2. Learning rate multiplicative factor: ");
        self.ke_sc4_l2.move(20, 150);
        tmp.append(self.ke_sc4_l2);

        self.ke_sc4_e2 = QLineEdit(self)
        self.ke_sc4_e2.move(290, 150);
        self.ke_sc4_e2.setText("0.1");
        tmp.append(self.ke_sc4_e2);

        self.ke_sc4_l3 = QLabel(self);
        self.ke_sc4_l3.setText("3. Number of epochs to wait: ");
        self.ke_sc4_l3.move(20, 200);
        tmp.append(self.ke_sc4_l3);

        self.ke_sc4_e3 = QLineEdit(self)
        self.ke_sc4_e3.move(290, 200);
        self.ke_sc4_e3.setText("10");
        tmp.append(self.ke_sc4_e3);

        self.ke_sc4_l4 = QLabel(self);
        self.ke_sc4_l4.setText("4. Threshold: ");
        self.ke_sc4_l4.move(20, 250);
        tmp.append(self.ke_sc4_l4);

        self.ke_sc4_e4 = QLineEdit(self)
        self.ke_sc4_e4.move(290, 250);
        self.ke_sc4_e4.setText("0.0001");
        tmp.append(self.ke_sc4_e4);

        self.ke_sc4_l5 = QLabel(self);
        self.ke_sc4_l5.setText("5. Minimum learning rate: ");
        self.ke_sc4_l5.move(20, 300);
        tmp.append(self.ke_sc4_l5);

        self.ke_sc4_e5 = QLineEdit(self)
        self.ke_sc4_e5.move(290, 300);
        self.ke_sc4_e5.setText("0.0");
        tmp.append(self.ke_sc4_e5);        

        self.scheduler_ui_keras.append(tmp)








        tmp = [];
        self.py_sc1_l1 = QLabel(self);
        self.py_sc1_l1.setText("No parameters (Arguments) to set.");
        self.py_sc1_l1.move(20, 100);
        tmp.append(self.py_sc1_l1);

        self.scheduler_ui_pytorch.append(tmp)




        tmp = [];
        self.py_sc2_l1 = QLabel(self);
        self.py_sc2_l1.setText("1. Step Size: ");
        self.py_sc2_l1.move(20, 100);
        tmp.append(self.py_sc2_l1);

        self.py_sc2_e1 = QLineEdit(self)
        self.py_sc2_e1.move(150, 100);
        self.py_sc2_e1.setText("5");
        tmp.append(self.py_sc2_e1);

        self.py_sc2_l2 = QLabel(self);
        self.py_sc2_l2.setText("2. Learning rate multiplicative factor: ");
        self.py_sc2_l2.move(20, 150);
        tmp.append(self.py_sc2_l2);

        self.py_sc2_e2 = QLineEdit(self)
        self.py_sc2_e2.move(290, 150);
        self.py_sc2_e2.setText("0.1");
        tmp.append(self.py_sc2_e2);

        self.scheduler_ui_pytorch.append(tmp)




        tmp = [];
        self.py_sc3_l1 = QLabel(self);
        self.py_sc3_l1.setText("1. Milestones: ");
        self.py_sc3_l1.move(20, 100);
        tmp.append(self.py_sc3_l1);

        self.py_sc3_e1 = QLineEdit(self)
        self.py_sc3_e1.move(150, 100);
        self.py_sc3_e1.setText("5, 10, 15");
        tmp.append(self.py_sc3_e1);

        self.py_sc3_l2 = QLabel(self);
        self.py_sc3_l2.setText("2. Learning rate multiplicative factor: ");
        self.py_sc3_l2.move(20, 150);
        tmp.append(self.py_sc3_l2);

        self.py_sc3_e2 = QLineEdit(self)
        self.py_sc3_e2.move(290, 150);
        self.py_sc3_e2.setText("0.1");
        tmp.append(self.py_sc3_e2);

        self.scheduler_ui_pytorch.append(tmp)




        tmp = [];
        self.py_sc4_l1 = QLabel(self);
        self.py_sc4_l1.setText("1. Learning rate multiplicative factor: ");
        self.py_sc4_l1.move(20, 100);
        tmp.append(self.py_sc4_l1);

        self.py_sc4_e1 = QLineEdit(self)
        self.py_sc4_e1.move(290, 100);
        self.py_sc4_e1.setText("0.9");
        tmp.append(self.py_sc4_e1);

        self.scheduler_ui_pytorch.append(tmp)





        tmp = [];
        self.py_sc5_l1 = QLabel(self);
        self.py_sc5_l1.setText("1. Mode: ");
        self.py_sc5_l1.move(20, 100);
        tmp.append(self.py_sc5_l1);

        self.py_sc5_cb1 = QComboBox(self);
        self.py_sc5_cb1.move(200, 100);
        self.py_sc5_cb1.addItems(["Min", "Max"]);
        tmp.append(self.py_sc5_cb1);

        self.py_sc5_l2 = QLabel(self);
        self.py_sc5_l2.setText("2. Learning rate multiplicative factor: ");
        self.py_sc5_l2.move(20, 150);
        tmp.append(self.py_sc5_l2);

        self.py_sc5_e2 = QLineEdit(self)
        self.py_sc5_e2.move(290, 150);
        self.py_sc5_e2.setText("0.1");
        tmp.append(self.py_sc5_e2);

        self.py_sc5_l3 = QLabel(self);
        self.py_sc5_l3.setText("3. Number of epochs to wait: ");
        self.py_sc5_l3.move(20, 200);
        tmp.append(self.py_sc5_l3);

        self.py_sc5_e3 = QLineEdit(self)
        self.py_sc5_e3.move(290, 200);
        self.py_sc5_e3.setText("10");
        tmp.append(self.py_sc5_e3);

        self.py_sc5_l4 = QLabel(self);
        self.py_sc5_l4.setText("4. Threshold: ");
        self.py_sc5_l4.move(20, 250);
        tmp.append(self.py_sc5_l4);

        self.py_sc5_e4 = QLineEdit(self)
        self.py_sc5_e4.move(290, 250);
        self.py_sc5_e4.setText("0.0001");
        tmp.append(self.py_sc5_e4);

        self.py_sc5_l5 = QLabel(self);
        self.py_sc5_l5.setText("5. Minimum learning rate: ");
        self.py_sc5_l5.move(20, 300);
        tmp.append(self.py_sc5_l5);

        self.py_sc5_e5 = QLineEdit(self)
        self.py_sc5_e5.move(290, 300);
        self.py_sc5_e5.setText("0.0");
        tmp.append(self.py_sc5_e5);        

        self.scheduler_ui_pytorch.append(tmp)
































        self.select_scheduler();

        self.tb1 = QTextEdit(self)
        self.tb1.move(550, 20)
        self.tb1.resize(300, 500)
        if(self.system["update"]["schedulers"]["active"]):
            wr = "";
            wr = json.dumps(self.system["update"]["schedulers"]["value"], indent=4)
            self.tb1.setText(wr);
        else:
            self.tb1.setText("Using Default Scheduler.")


        self.b4 = QPushButton('Select Scheduler', self)
        self.b4.move(400,400)
        self.b4.clicked.connect(self.add_scheduler)

        
        self.b6 = QPushButton('Clear ', self)
        self.b6.move(400,500)
        self.b6.clicked.connect(self.clear_scheduler)

    







    def select_scheduler(self):
        self.current_scheduler = {};
        self.current_scheduler["name"] = "";
        self.current_scheduler["params"] = {};

        if(self.system["backend"] == "Mxnet-1.5.1"):
            self.current_scheduler["name"] = self.cb1.currentText();
            index = self.mxnet_schedulers_list.index(self.cb1.currentText());
            for i in range(len(self.scheduler_ui_mxnet)):
                for j in range(len(self.scheduler_ui_mxnet[i])):
                    if((index-1)==i):
                        self.scheduler_ui_mxnet[i][j].show();
                    else:
                        self.scheduler_ui_mxnet[i][j].hide();

            for i in range(len(self.scheduler_ui_keras)):
                for j in range(len(self.scheduler_ui_keras[i])):
                    self.scheduler_ui_keras[i][j].hide();
            for i in range(len(self.scheduler_ui_pytorch)):
                for j in range(len(self.scheduler_ui_pytorch[i])):
                    self.scheduler_ui_pytorch[i][j].hide();
            


        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            self.current_scheduler["name"] = self.cb2.currentText();
            index = self.keras_schedulers_list.index(self.cb2.currentText());
            for i in range(len(self.scheduler_ui_keras)):
                for j in range(len(self.scheduler_ui_keras[i])):
                    if((index-1)==i):
                        self.scheduler_ui_keras[i][j].show();
                    else:
                        self.scheduler_ui_keras[i][j].hide();

            for i in range(len(self.scheduler_ui_mxnet)):
                for j in range(len(self.scheduler_ui_mxnet[i])):
                    self.scheduler_ui_mxnet[i][j].hide();
            for i in range(len(self.scheduler_ui_pytorch)):
                for j in range(len(self.scheduler_ui_pytorch[i])):
                    self.scheduler_ui_pytorch[i][j].hide();



        elif(self.system["backend"] == "Pytorch-1.3.1"):
            self.current_scheduler["name"] = self.cb3.currentText();
            index = self.pytorch_schedulers_list.index(self.cb3.currentText());
            for i in range(len(self.scheduler_ui_pytorch)):
                for j in range(len(self.scheduler_ui_pytorch[i])):
                    if((index-1)==i):
                        self.scheduler_ui_pytorch[i][j].show();
                    else:
                        self.scheduler_ui_pytorch[i][j].hide();

            for i in range(len(self.scheduler_ui_keras)):
                for j in range(len(self.scheduler_ui_keras[i])):
                    self.scheduler_ui_keras[i][j].hide();
            for i in range(len(self.scheduler_ui_mxnet)):
                for j in range(len(self.scheduler_ui_mxnet[i])):
                    self.scheduler_ui_mxnet[i][j].hide();



    def add_scheduler(self):
        self.system["update"]["schedulers"]["active"] = True;
        if(self.system["backend"] == "Mxnet-1.5.1"):
            if(self.current_scheduler["name"] == self.mxnet_schedulers_list[1]):
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;

            elif(self.current_scheduler["name"] == self.mxnet_schedulers_list[2]):
                self.current_scheduler["params"]["step_size"] = self.mx_sc2_e1.text();
                self.current_scheduler["params"]["gamma"] = self.mx_sc2_e2.text();
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;

            elif(self.current_scheduler["name"] == self.mxnet_schedulers_list[3]):
                self.current_scheduler["params"]["milestones"] = self.mx_sc3_e1.text();
                self.current_scheduler["params"]["gamma"] = self.mx_sc3_e2.text();
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;



        elif(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
            if(self.current_scheduler["name"] == self.keras_schedulers_list[1]):
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;

            elif(self.current_scheduler["name"] == self.keras_schedulers_list[2]):
                self.current_scheduler["params"]["step_size"] = self.ke_sc2_e1.text();
                self.current_scheduler["params"]["gamma"] = self.ke_sc2_e2.text();
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;

            elif(self.current_scheduler["name"] == self.keras_schedulers_list[3]):
                self.current_scheduler["params"]["gamma"] = self.ke_sc3_e1.text();
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;

            elif(self.current_scheduler["name"] == self.keras_schedulers_list[4]):
                self.current_scheduler["params"]["mode"] = self.ke_sc4_cb1.currentText();
                self.current_scheduler["params"]["factor"] = self.ke_sc4_e2.text();
                self.current_scheduler["params"]["patience"] = self.ke_sc4_e3.text();
                self.current_scheduler["params"]["threshold"] = self.ke_sc4_e4.text();
                self.current_scheduler["params"]["min_lr"] = self.ke_sc4_e5.text();
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;



        elif(self.system["backend"] == "Pytorch-1.3.1"):
            if(self.current_scheduler["name"] == self.pytorch_schedulers_list[1]):
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;

            elif(self.current_scheduler["name"] == self.keras_schedulers_list[2]):
                self.current_scheduler["params"]["step_size"] = self.py_sc2_e1.text();
                self.current_scheduler["params"]["gamma"] = self.py_sc2_e2.text();
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;

            elif(self.current_scheduler["name"] == self.keras_schedulers_list[3]):
                self.current_scheduler["params"]["milestones"] = self.py_sc3_e1.text();
                self.current_scheduler["params"]["gamma"] = self.py_sc3_e2.text();
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;

            elif(self.current_scheduler["name"] == self.keras_schedulers_list[4]):
                self.current_scheduler["params"]["gamma"] = self.py_sc4_e1.text();
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;

            elif(self.current_scheduler["name"] == self.keras_schedulers_list[5]):
                self.current_scheduler["params"]["mode"] = self.py_sc5_cb1.currentText();
                self.current_scheduler["params"]["factor"] = self.py_sc5_e2.text();
                self.current_scheduler["params"]["patience"] = self.py_sc5_e3.text();
                self.current_scheduler["params"]["threshold"] = self.py_sc5_e4.text();
                self.current_scheduler["params"]["min_lr"] = self.py_sc5_e5.text();
                self.system["update"]["schedulers"]["value"] = self.current_scheduler;






        wr = "";
        wr = json.dumps(self.system["update"]["schedulers"]["value"], indent=4)
        self.tb1.setText(wr);


    def clear_scheduler(self):
        self.system["update"]["schedulers"]["value"] = "";
        self.system["update"]["schedulers"]["active"] = False;

        wr = "";
        self.tb1.setText(wr);




    def forward(self):        
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_loss_param.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_optimizer_param.emit();



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateSchedulerParam()
screen.show()
sys.exit(app.exec_())
'''