import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowClassificationMain(QtWidgets.QWidget):

    backward_main = QtCore.pyqtSignal();
    forward_project = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Monk Classification'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 300
        self.cfg_setup()
        self.initUI()

    def cfg_setup(self):
        if(os.path.isfile("base_classification.json")):
            with open('base_classification.json') as json_file:
                self.system = json.load(json_file)
        else:
            self.system = {};
            self.system["project"] = "";
            self.system["experiment"] = "";
            self.system["backend"] = "";
            self.system["mode"] = "quick";
            
            self.system["datatype"] = "image";
            self.system["structuretype"] = "foldered"; 
            self.system["labeltype"] = "single";
            self.system["traindata"] = {};
            self.system["valdata"] = {};
            self.system["val"] = "yes";
            self.system["traindata"]["dir"] = "monk_v1/monk/system_check_tests/datasets/dataset_cats_dogs_train/";
            self.system["valdata"]["dir"] = "monk_v1/monk/system_check_tests/datasets/dataset_cats_dogs_eval/";
            self.system["traindata"]["cdir"] = "monk_v1/monk/system_check_tests/datasets/dataset_csv_id/train/";
            self.system["valdata"]["cdir"] = "monk_v1/monk/system_check_tests/datasets/dataset_csv_id/val/";
            self.system["traindata"]["csv"] = "monk_v1/monk/system_check_tests/datasets/dataset_csv_id/train.csv";
            self.system["valdata"]["csv"] = "monk_v1/monk/system_check_tests/datasets/dataset_csv_id/val.csv";

            self.system["evaluate"] = {};
            self.system["test_data"] = "Single Image";
            self.system["img_file"] = "monk_v1/monk/system_check_tests/datasets/dataset_cats_dogs_test/0.jpg";
            self.system["img_folder"] = "monk_v1/monk/system_check_tests/datasets/dataset_cats_dogs_test/";
            self.system["evaluate"]["dir"] = "monk_v1/monk/system_check_tests/datasets/dataset_cats_dogs_eval/";
            self.system["evaluate"]["structuretype"] = "foldered";
            self.system["evaluate"]["csv"] = "monk_v1/monk/system_check_tests/datasets/dataset_csv_id/val.csv";
            self.system["evaluate"]["cdir"] = "monk_v1/monk/system_check_tests/datasets/dataset_csv_id/val/";

            self.system["model"] = "";
            self.system["freeze_base_model"] = "yes";
            self.system["epochs"] = "2";

            self.system["copy_from"] = ["", ""];            

            self.system["compare"] = {};
            self.system["compare"]["project"] = "";
            self.system["compare"]["experiments"] = [];


            with open('base_classification.json', 'w') as outfile:
                json.dump(self.system, outfile)


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);

        self.wid = [];

        self.l1 = QLabel(self);
        self.l1.setText("Create a new project:");
        self.l1.move(30, 20);
        self.wid.append(self.l1);

        self.e1 = QLineEdit(self)
        self.e1.move(200, 20);
        self.e1.setText(self.system["project"]);
        self.e1.resize(200, 25);
        self.wid.append(self.e1);

        self.b1 = QPushButton('Create New', self)
        self.b1.move(200,60)
        self.b1.clicked.connect(self.forward1)
        self.wid.append(self.b1);


        self.l2 = QLabel(self);
        self.l2.setText("Select an existing project:");
        self.l2.move(30, 120);
        self.wid.append(self.l2);

        if(os.path.isdir("workspace")):
            self.cb1 = QComboBox(self);
            self.project_list = os.listdir("workspace/")
            if "comparison" in self.project_list:
                self.project_list.remove("comparison");
            self.project_list.insert(0, "Select")
            self.cb1.addItems(self.project_list);
            self.cb1.move(230, 120);
        else:
            self.cb1 = QComboBox(self);
            self.project_list = [];
            self.cb1.addItems(["Projects not created yet"]);
            self.cb1.move(230, 120);
            self.cb1.setEnabled(False);
        self.cb1.activated.connect(self.forward2)






        # Backward
        self.b2 = QPushButton('Back', self)
        self.b2.move(600,250)
        self.b2.clicked.connect(self.backward)


        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(700,250)
        self.b3.clicked.connect(self.close)


        self.te1 = QTextBrowser(self);
        self.te1.move(450, 20);
        self.te1.setFixedSize(300, 200);


        
        # Forward
        if not os.path.isdir("monk_v1"):
            self.b1.setText("Cloning (Wait)")
            for x in self.wid:
                x.setEnabled(False);

            self.process = QtCore.QProcess(self)
            self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.stdoutReady)
            self.process.readyReadStandardError.connect(self.stderrReady)
            self.process.finished.connect(self.finished)

            self.process.start('git clone https://github.com/Tessellate-Imaging/monk_v1') 
        else:
            self.te1.setText("Monk Classification Library Cloned.");           

        

    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')           
        self.append(text)

    def finished(self):
        for x in self.wid:
            x.setEnabled(True);
        self.b1.setText('Create New');
        self.te1.setText("Monk Classification Library Cloned.");  


    def stderrReady(self):
        text = str(self.process.readAllStandardError().data(), encoding='utf-8')
        self.append(text)

    def append(self, text):
        cursor = self.te1.textCursor()  
        self.te1.ensureCursorVisible() 
        cursor.movePosition(cursor.End)
        cursor.insertText(text) 



    def forward1(self):
        project_name = self.e1.text();
        self.move_ahead = False;
        if(project_name == ""):
            QMessageBox.about(self, "Message", "Project name cannot be blank");
        elif(" " in project_name or "/" in project_name):
            QMessageBox.about(self, "Message", "Project name cannot have special characters or spaces except underscore(_)");
        elif(project_name in self.project_list):
            qm = QMessageBox(self)
            ret = qm.question(self, 'Warning', "Project " + project_name + " exists. Do you want to delete and overwrite it?", qm.Yes | qm.No)
            if(ret == qm.Yes):
                self.move_ahead = True;
        else:
            self.move_ahead = True;

        if self.move_ahead:
            self.system["project"] = project_name;
            with open('base_classification.json', 'w') as outfile:
                json.dump(self.system, outfile)

            self.forward_project.emit();

    def forward2(self):
        self.system["project"] = self.cb1.currentText();

        with open('base_classification.json', 'w') as outfile:
                json.dump(self.system, outfile)

        self.forward_project.emit();


    def backward(self):
        self.system["project"] = self.e1.text();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_main.emit();


'''
app = QApplication(sys.argv)
screen = WindowClassificationMain()
screen.show()
sys.exit(app.exec_())
'''