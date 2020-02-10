import os
import sys
import json
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationProjectMain(QtWidgets.QWidget):

    forward_experiment = QtCore.pyqtSignal();
    forward_copy_experiment = QtCore.pyqtSignal();
    backward_classification_main = QtCore.pyqtSignal();
    forward_compare_current = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Classification - Project Name - {}'.format(self.system["project"])
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 600
        self.initUI()

    def cfg_setup(self):
        with open('base_classification.json') as json_file:
            self.system = json.load(json_file)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);


        # Backward
        self.b2 = QPushButton('Back', self)
        self.b2.move(600,550)
        self.b2.clicked.connect(self.backward)


        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(700,550)
        self.b3.clicked.connect(self.close)


        self.l1 = QLabel(self);
        self.l1.setText("Create a new experiment:");
        self.l1.move(30, 20);

        self.e1 = QLineEdit(self)
        self.e1.move(220, 20);
        self.e1.setText(self.system["experiment"]);
        self.e1.resize(200, 25);

        self.b1 = QPushButton('Create New', self)
        self.b1.move(430,20)
        self.b1.clicked.connect(self.forward1);

        self.b2 = QPushButton('Compare Selected Experiments', self)
        self.b2.move(20,550)
        self.b2.clicked.connect(self.compare);

        self.wid = [];

        if os.path.isdir("workspace/" + self.system["project"]):
            self.experiment_list = sorted(os.listdir("workspace/" + self.system["project"]));
        else:
            self.experiment_list = [];

        self.createLayout_Container();
        

    def createLayout_group(self, experiment_name):
        sgroupbox = QGroupBox("Experiment - {}:".format(experiment_name), self)
        layout_groupbox = QGridLayout(sgroupbox)

        tmp = [];        

        chb1 = QCheckBox("Select for comparison", sgroupbox)
        layout_groupbox.addWidget(chb1, 1, 1)
        tmp.append(chb1);
        
        tb1 = QTextEdit(sgroupbox)
        tb1.resize(100, 300)
        layout_groupbox.addWidget(tb1, 1, 2)
        tb1.setText(self.get_training_details2(experiment_name))
        tmp.append(tb1);

        b1 = QPushButton('Select for Re-Train/Infer/Evaluate', sgroupbox)
        layout_groupbox.addWidget(b1, 2, 1)
        b1.clicked.connect(lambda checked, arg1=experiment_name: self.forward2(arg1));
        tmp.append(b1);


        b2 = QPushButton('Copy Experiment and Create New', sgroupbox)
        layout_groupbox.addWidget(b2, 2, 2)
        b2.clicked.connect(lambda checked, arg1=self.system["project"], arg2=experiment_name + "_copy": self.forward3(arg1, arg2));
        tmp.append(b2);
        b2.setEnabled(False);



        self.wid.append(tmp)

        return sgroupbox




    def createLayout_Container(self):
        self.scrollarea = QScrollArea(self)
        self.scrollarea.setFixedSize(700, 480)
        self.scrollarea.setWidgetResizable(True)

        widget = QWidget()
        self.scrollarea.setWidget(widget)
        self.layout_SArea = QVBoxLayout(widget)

        for i in range(len(self.experiment_list)):
            self.layout_SArea.addWidget(self.createLayout_group(self.experiment_list[i]))
        self.layout_SArea.addStretch(1)

        self.scrollarea.move(20, 50)



    def get_training_details2(self, experiment):
        with open("workspace/" + self.system["project"] + "/" + experiment + "/experiment_state.json") as json_file:
            data = json.load(json_file)
        wr = "";
        if(data["library"] == "Keras"):
            wr += "Backend - {}\n".format("Keras-2.2.5_Tensorflow-1");
        elif(data["library"] == "Pytorch"):
            wr += "Backend - {}\n".format("Pytorch-1.3.1");
        elif(data["library"] == "Mxnet"):
            wr += "Backend - {}\n".format("Mxnet-1.5.1");

        if(data["training"]["status"]):
            tmp = np.load(data["training"]["outputs"]["log_train_acc_history_relative"], allow_pickle=True);
            wr += "Train acc - {}\n".format(tmp[-1]);
            tmp = np.load(data["training"]["outputs"]["log_val_acc_history_relative"], allow_pickle=True);
            wr += "Val acc - {}\n".format(tmp[-1]);
            tmp = np.load(data["training"]["outputs"]["log_train_loss_history_relative"], allow_pickle=True);
            wr += "Train loss - {}\n".format(tmp[-1]);
            tmp = np.load(data["training"]["outputs"]["log_val_loss_history_relative"], allow_pickle=True);
            wr += "Val losss - {}\n".format(tmp[-1]);
        else:
            wr += "Training Incomplete. Do not select for comparison\n";

        return wr;



    def forward1(self):
        experiment_name = self.e1.text();
        self.move_ahead = False;
        if(experiment_name == ""):
            QMessageBox.about(self, "Warning", "Experiment name cannot be blank");
        elif(" " in experiment_name or "/" in experiment_name):
            QMessageBox.about(self, "Warning", "Experiment name cannot have special characters or spaces except underscore(_)");
        elif(experiment_name in self.experiment_list):
            qm = QMessageBox(self)
            ret = qm.question(self, 'Warning', "Experiment " + experiment_name + " exists. Do you want to delete and overwrite it?", qm.Yes | qm.No)
            if(ret == qm.Yes):
                os.system("rm -r workspace/" + self.system["project"] + "/" + experiment_name);
                self.move_ahead = True;
        else:
            self.move_ahead = True;

        if self.move_ahead:
            self.system["experiment"] = experiment_name;
            with open('base_classification.json', 'w') as outfile:
                json.dump(self.system, outfile)
            self.forward_experiment.emit();



    def forward2(self, experiment_name):
        self.system["experiment"] = experiment_name;
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_experiment.emit();



    def forward3(self, project_name, experiment_name):
        self.system["experiment"] = experiment_name;
        tmp1 = experiment_name.split("_");
        tmp2 = [];
        for i in range(len(tmp1)):
            if(tmp1[i] != "copy"):
                tmp2.append(tmp1[i]);
        experiment = "_".join(tmp2);


        self.system["copy_from"] = [project_name, experiment];
        with open('workspace/' + project_name + '/' + experiment + '/experiment_state.json') as json_file:
            data = json.load(json_file)

        if(data["library"] == "Mxnet"):
            self.system["backend"] = "Mxnet-1.5.1";
        elif(data["library"] == "Pytorch"):
            self.system["backend"] = "Pytorch-1.3.1";        
        elif(data["library"] == "Keras"):
            self.system["backend"] = "Keras-2.2.5_Tensorflow-1";


        self.move_ahead = False;
        if(experiment_name == ""):
            QMessageBox.about(self, "Warning", "Experiment name cannot be blank");
        elif(" " in experiment_name or "/" in experiment_name):
            QMessageBox.about(self, "Warning", "Experiment name cannot have special characters or spaces except underscore(_)");
        else:
            self.move_ahead = True;

        if self.move_ahead:
            self.system["experiment"] = experiment_name;
            with open('base_classification.json', 'w') as outfile:
                json.dump(self.system, outfile)
            self.forward_copy_experiment.emit();


    def compare(self):
        self.system["compare"]["project"] = self.system["project"];
        self.system["compare"]["experiments"] = [];
        for i in range(len(self.wid)):
            if(self.wid[i][0].isChecked()):
                self.system["compare"]["experiments"].append(self.experiment_list[i]);

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

        if(len(self.system["compare"]["experiments"]) > 1):
            self.forward_compare_current.emit();
        else:
            QMessageBox.about(self, "Warning", "Select more than two experiments for comparison");




    def backward(self):
        self.system["experiment"] = self.e1.text();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_classification_main.emit();


'''
app = QApplication(sys.argv)
screen = WindowClassificationProjectMain()
screen.show()
sys.exit(app.exec_())
'''