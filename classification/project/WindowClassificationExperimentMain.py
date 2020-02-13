import os
import sys
import json
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationExperimentMain(QtWidgets.QWidget):

    forward_run_mode = QtCore.pyqtSignal();
    forward_infer = QtCore.pyqtSignal();
    forward_validate = QtCore.pyqtSignal();
    backward_project_main = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Classification - Experiment Name - {}'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width_1 = 900
        self.height_1 = 600
        self.width_2 = 600
        self.height_2 = 500
        self.initUI()

    def cfg_setup(self):
        with open('base_classification.json') as json_file:
            self.system = json.load(json_file)

    def initUI(self):
        if(os.path.isdir("workspace/" + self.system["project"] + "/" + self.system["experiment"])):
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width_1, self.height_1);

            self.b1 = QPushButton('Back', self)
            self.b1.move(700, 550)
            self.b1.clicked.connect(self.backward)


            # Quit
            self.b2 = QPushButton('Quit', self)
            self.b2.move(800, 550)
            self.b2.clicked.connect(self.close)
            

            self.l1 = QLabel(self);
            self.l1.setText("1. General Details");
            self.l1.move(5, 20);

            self.tb1 = QTextEdit(self)
            self.tb1.move(5, 50)
            self.tb1.setText(self.get_general_details());
            self.tb1.resize(280, 200)
            self.tb1.setReadOnly(True)

            self.l2 = QLabel(self);
            self.l2.setText("2. Dataset Details");
            self.l2.move(5, 270);

            self.tb2 = QTextEdit(self)
            self.tb2.move(5, 290)
            self.tb2.setText(self.get_dataset_details());
            self.tb2.resize(280, 200)
            self.tb2.setReadOnly(True)

            self.l3 = QLabel(self);
            self.l3.setText("3. Model Details");
            self.l3.move(300, 20);

            self.tb3 = QTextEdit(self)
            self.tb3.move(300, 50)
            self.tb3.setText(self.get_model_details());
            self.tb3.resize(280, 200)
            self.tb3.setReadOnly(True)

            self.l4 = QLabel(self);
            self.l4.setText("4. Hyper param Details");
            self.l4.move(300, 270);

            self.tb4 = QTextEdit(self)
            self.tb4.move(300, 290)
            self.tb4.setText(self.get_hyperparam_details());
            self.tb4.resize(280, 200)
            self.tb4.setReadOnly(True)

            self.l5 = QLabel(self);
            self.l5.setText("5. Training params");
            self.l5.move(600, 20);

            self.tb5 = QTextEdit(self)
            self.tb5.move(600, 50)
            self.tb5.setText(self.get_training_details());
            self.tb5.resize(280, 200)
            self.tb5.setReadOnly(True)

            self.l6 = QLabel(self);
            self.l6.setText("6. Accuracies and losses");
            self.l6.move(600, 270);

            self.tb5 = QTextEdit(self)
            self.tb5.move(600, 290)
            self.tb5.setText(self.get_training_details2());
            self.tb5.resize(280, 100)
            self.tb5.setReadOnly(True)


            self.b3 = QPushButton('Train Again', self)
            self.b3.move(620, 410)
            self.b3.clicked.connect(self.train)


            self.b4 = QPushButton('Validate', self)
            self.b4.move(620, 450)
            self.b4.clicked.connect(self.validate)


            self.b5 = QPushButton('Infer', self)
            self.b5.move(620, 490)
            self.b5.clicked.connect(self.infer)




        else:

            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width_2, self.height_2);


            # Backward
            self.b1 = QPushButton('Back', self)
            self.b1.move(300, 450)
            self.b1.clicked.connect(self.backward)

            # Backward
            self.b2 = QPushButton('Train', self)
            self.b2.move(400, 450)
            self.b2.clicked.connect(self.forward)


            # Quit
            self.b3 = QPushButton('Quit', self)
            self.b3.move(500, 450)
            self.b3.clicked.connect(self.close)


            self.l1 = QLabel(self);
            self.l1.setText("Select Backend");
            self.l1.move(20, 20);

            self.cb1 = QComboBox(self);
            self.backends = ["Mxnet-1.5.1", "Pytorch-1.3.1", "Keras-2.2.5_Tensorflow-1"]
            self.backends.insert(0, "Select")
            self.cb1.addItems(self.backends);
            self.cb1.move(150, 20);
            index = self.cb1.findText(self.system["backend"], QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.cb1.setCurrentIndex(index)
            self.cb1.activated.connect(self.select_backend);



            self.l2 = QLabel(self);
            self.l2.setText("Run Installation");
            self.l2.move(20, 60);


            self.l3 = QLabel(self);
            self.l3.setText("Ignore if already ran once");
            self.l3.move(20, 90);

            self.cb4 = QComboBox(self);
            self.cudas = ["CPU-Ubuntu", "CPU-MacOS", "Cuda-9.0-Ubuntu", "Cuda-10.0-Ubuntu"];
            self.cb4.addItems(self.cudas);
            self.cb4.move(20, 110);


            self.b4 = QPushButton('Start Installation', self)
            self.b4.move(200,110)
            self.b4.clicked.connect(self.install)

            self.tb1 = QTextEdit(self)
            self.tb1.move(350, 110)
            self.tb1.setText("");
            self.tb1.resize(200, 25)
            self.tb1.setReadOnly(True)

            self.te1 = QTextBrowser(self);
            self.te1.move(20, 140);
            self.te1.setFixedSize(500, 300);


            self.process = QtCore.QProcess(self)
            self.process.readyReadStandardOutput.connect(self.stdoutReady)
            self.process.readyReadStandardError.connect(self.stderrReady)
            self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)





    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')
        if("Completed" in text):
            QMessageBox.about(self, "Installation Status", "Completed");
            self.tb1.setText("Installation Complete");
        self.append(text)



    def stderrReady(self):
        text = str(self.process.readAllStandardError().data(), encoding='utf-8')
        QMessageBox.about(self, "Installation Status", "Errors Found");
        self.tb1.setText("Error in installation");
        self.append(text)


    def append(self, text):
        cursor = self.te1.textCursor()  
        self.te1.ensureCursorVisible() 
        cursor.movePosition(cursor.End)
        cursor.insertText(text)




        
    def select_backend(self):
        self.system["backend"] = self.cb1.currentText();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)



    def install(self):
        self.tb1.setText("Installation Running");
        if(self.cb4.currentText() == "CPU-Ubuntu"):
            os.system("cp cfg/classification/install_cls_cpu_ubuntu.sh .");
            os.system("chmod +x install_cls_cpu_ubuntu.sh");
            self.process.start('bash', ['install_cls_cpu_ubuntu.sh'])
            self.append("Process PID: " + str(self.process.pid()) + "\n");
        elif(self.cb4.currentText() == "CPU-MacOS"):
            os.system("cp cfg/classification/install_cls_cpu_macos.sh .");
            os.system("chmod +x install_cls_cpu_macos.sh");
            self.process.start('bash', ['install_cls_cpu_macos.sh'])
            self.append("Process PID: " + str(self.process.pid()) + "\n");
        elif(self.cb4.currentText() == "Cuda-9.0-Ubuntu"):
            os.system("cp cfg/classification/install_cls_cuda90_ubuntu.sh .");
            os.system("chmod +x install_cls_cuda90_ubuntu.sh");
            self.process.start('bash', ['install_cls_cuda90_ubuntu.sh'])
            self.append("Process PID: " + str(self.process.pid()) + "\n");
        elif(self.cb4.currentText() == "Cuda-10.0-Ubuntu"):
            os.system("cp cfg/classification/install_cls_cuda100_ubuntu.sh .");
            os.system("chmod +x install_cls_cuda100_ubuntu.sh");
            self.process.start('bash', ['install_cls_cuda100_ubuntu.sh'])
            self.append("Process PID: " + str(self.process.pid()) + "\n");


    def get_general_details(self):
        with open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/experiment_state.json") as json_file:
            data = json.load(json_file)
        wr = "";
        wr += "Project - {}\n".format(data["project_name"]);
        wr += "Experiment - {}\n".format(data["experiment_name"]);
        wr += "Origin - {}\n".format(data["origin"]);
        wr += "Backend - {}\n".format(self.system["backend"]);
        wr += "Output Dir - {}\n".format(data["output_dir_relative"]);
        wr += "Model Dir - {}\n".format(data["model_dir_relative"]);
        wr += "Log Dir - {}\n".format(data["log_dir_relative"]);

        return wr;


    def get_dataset_details(self):
        with open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/experiment_state.json") as json_file:
            data = json.load(json_file)
        wr = "";
        tmp = json.dumps(data["dataset"], indent=4)
        wr += "{}\n".format(tmp);
        return wr;


    def get_model_details(self):
        with open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/experiment_state.json") as json_file:
            data = json.load(json_file)
        wr = "";
        tmp = json.dumps(data["model"], indent=4)
        wr += "{}\n".format(tmp);
        return wr;


    def get_hyperparam_details(self):
        with open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/experiment_state.json") as json_file:
            data = json.load(json_file)
        wr = "";
        tmp = json.dumps(data["hyper-parameters"], indent=4)
        wr += "{}\n".format(tmp);
        return wr;

    def get_training_details(self):
        with open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/experiment_state.json") as json_file:
            data = json.load(json_file)
        wr = "";
        tmp = json.dumps(data["training"], indent=4)
        wr += "{}\n".format(tmp);
        tmp = json.dumps(data["testing"], indent=4)
        wr += "{}\n".format(tmp);
        return wr;


    def get_training_details2(self):
        with open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/experiment_state.json") as json_file:
            data = json.load(json_file)
        wr = "";
        if(data["training"]["status"]):
            tmp = np.load(data["training"]["outputs"]["log_train_acc_history_relative"], allow_pickle=True);
            wr += "Train acc - {}\n".format(tmp[-1]);
            tmp = np.load(data["training"]["outputs"]["log_val_acc_history_relative"], allow_pickle=True);
            wr += "Val acc - {}\n".format(tmp[-1]);
            tmp = np.load(data["training"]["outputs"]["log_train_loss_history_relative"], allow_pickle=True);
            wr += "Train loss - {}\n".format(tmp[-1]);
            tmp = np.load(data["training"]["outputs"]["log_val_loss_history_relative"], allow_pickle=True);
            wr += "Val losss - {}\n".format(tmp[-1]);
        wr += "Training Incomplete";

        return wr;


    def train(self):
        self.forward_run_mode.emit();

    def validate(self):
        self.forward_validate.emit();

    def infer(self):
        self.forward_infer.emit();




    def forward(self):
        if(not os.path.isdir("workspace/" + self.system["project"] + "/" + self.system["experiment"])):
            if(self.system["backend"] == "Select"):
                QMessageBox.about(self, "Warning", "Select backend framework");
            else:
                self.system["backend"] = self.cb1.currentText();
                with open('base_classification.json', 'w') as outfile:
                    json.dump(self.system, outfile)
                self.forward_run_mode.emit();
        else:
            self.forward_run_mode.emit();


    def backward(self):
        if(not os.path.isdir("workspace/" + self.system["project"] + "/" + self.system["experiment"])):
            self.system["backend"] = self.cb1.currentText();
            with open('base_classification.json', 'w') as outfile:
                json.dump(self.system, outfile)
        self.backward_project_main.emit();


'''
app = QApplication(sys.argv)
screen = WindowClassificationExperimentMain()
screen.show()
sys.exit(app.exec_())
'''