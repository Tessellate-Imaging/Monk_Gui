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


            self.system["update"] = {};
            self.system["update"]["input_size"] = {};
            self.system["update"]["input_size"]["active"] = False;
            self.system["update"]["input_size"]["value"] = "224";
            self.system["update"]["batch_size"] = {};
            self.system["update"]["batch_size"]["active"] = False;
            self.system["update"]["batch_size"]["value"] = "4";
            self.system["update"]["shuffle_data"] = {};
            self.system["update"]["shuffle_data"]["active"] = False;
            self.system["update"]["shuffle_data"]["value"] = "True";
            self.system["update"]["num_processors"] = {};
            self.system["update"]["num_processors"]["active"] = False;
            self.system["update"]["num_processors"]["value"] = "3";
            self.system["update"]["trainval_split"] = {};
            self.system["update"]["trainval_split"]["active"] = False;
            self.system["update"]["trainval_split"]["value"] = "0.7";

            self.system["update"]["transforms"] = {};
            self.system["update"]["transforms"]["active"] = False;
            self.system["update"]["transforms"]["value"] = [];

            self.system["update"]["model_name"] = {};
            self.system["update"]["model_name"]["active"] = False;
            self.system["update"]["model_name"]["value"] = "";

            self.system["update"]["use_gpu"] = {};
            self.system["update"]["use_gpu"]["active"] = False;
            self.system["update"]["use_gpu"]["value"] = "True";

            self.system["update"]["use_pretrained"] = {};
            self.system["update"]["use_pretrained"]["active"] = False;
            self.system["update"]["use_pretrained"]["value"] = "True";

            self.system["update"]["freeze_base_network"] = {};
            self.system["update"]["freeze_base_network"]["active"] = False;
            self.system["update"]["freeze_base_network"]["value"] = "True";

            self.system["update"]["freeze_layers"] = {};
            self.system["update"]["freeze_layers"]["active"] = False;
            self.system["update"]["freeze_layers"]["value"] = "10";

            self.system["update"]["layers"] = {};
            self.system["update"]["layers"]["active"] = False;
            self.system["update"]["layers"]["value"] = [];

            self.system["update"]["realtime_progress"] = {};
            self.system["update"]["realtime_progress"]["active"] = False;
            self.system["update"]["realtime_progress"]["value"] = "True";

            self.system["update"]["progress"] = {};
            self.system["update"]["progress"]["active"] = False;
            self.system["update"]["progress"]["value"] = "True";

            self.system["update"]["save_intermediate"] = {};
            self.system["update"]["save_intermediate"]["active"] = False;
            self.system["update"]["save_intermediate"]["value"] = "True";

            self.system["update"]["save_logs"] = {};
            self.system["update"]["save_logs"]["active"] = False;
            self.system["update"]["save_logs"]["value"] = "True";

            self.system["update"]["optimizers"] = {};
            self.system["update"]["optimizers"]["active"] = False;
            self.system["update"]["optimizers"]["value"] = "";

            self.system["update"]["schedulers"] = {};
            self.system["update"]["schedulers"]["active"] = False;
            self.system["update"]["schedulers"]["value"] = "";

            self.system["update"]["losses"] = {};
            self.system["update"]["losses"]["active"] = False;
            self.system["update"]["losses"]["value"] = "";


            self.system["analysis"] = {};
            self.system["analysis"]["input_size"] = {};
            self.system["analysis"]["input_size"]["analysis_name"] = "analyse_input_sizes";
            self.system["analysis"]["input_size"]["list"] = "128, 224, 256, 512";
            self.system["analysis"]["input_size"]["percent"] = "30";
            self.system["analysis"]["input_size"]["epochs"] = "10";
            self.system["analysis"]["input_size"]["analysis"] = "";

            self.system["analysis"]["batch_size"] = {};
            self.system["analysis"]["batch_size"]["analysis_name"] = "analyse_batch_sizes";
            self.system["analysis"]["batch_size"]["list"] = "4, 8, 16, 32";
            self.system["analysis"]["batch_size"]["percent"] = "30";
            self.system["analysis"]["batch_size"]["epochs"] = "10";
            self.system["analysis"]["batch_size"]["analysis"] = "";

            self.system["analysis"]["trainval_split"] = {};
            self.system["analysis"]["trainval_split"]["analysis_name"] = "analyse_trainval_split";
            self.system["analysis"]["trainval_split"]["list"] = "0.3, 0.6, 0.9";
            self.system["analysis"]["trainval_split"]["percent"] = "30";
            self.system["analysis"]["trainval_split"]["epochs"] = "10";
            self.system["analysis"]["trainval_split"]["analysis"] = "";


            self.system["analysis"]["model_list"] = {};
            self.system["analysis"]["model_list"]["analysis_name"] = "analyse_model_list";
            self.system["analysis"]["model_list"]["list"] = "";
            self.system["analysis"]["model_list"]["percent"] = "30";
            self.system["analysis"]["model_list"]["epochs"] = "10";
            self.system["analysis"]["model_list"]["analysis"] = "";


            self.system["analysis"]["use_pretrained"] = {};
            self.system["analysis"]["use_pretrained"]["analysis_name"] = "analyse_use_pretrained";
            self.system["analysis"]["use_pretrained"]["list"] = "yes, no";
            self.system["analysis"]["use_pretrained"]["percent"] = "30";
            self.system["analysis"]["use_pretrained"]["epochs"] = "10";
            self.system["analysis"]["use_pretrained"]["analysis"] = "";


            self.system["analysis"]["freeze_base"] = {};
            self.system["analysis"]["freeze_base"]["analysis_name"] = "analyse_freeze_base";
            self.system["analysis"]["freeze_base"]["list"] = "yes, no";
            self.system["analysis"]["freeze_base"]["percent"] = "30";
            self.system["analysis"]["freeze_base"]["epochs"] = "10";
            self.system["analysis"]["freeze_base"]["analysis"] = "";


            self.system["analysis"]["freeze_layer"] = {};
            self.system["analysis"]["freeze_layer"]["analysis_name"] = "analyse_freeze_layers";
            self.system["analysis"]["freeze_layer"]["list"] = "";
            self.system["analysis"]["freeze_layer"]["percent"] = "30";
            self.system["analysis"]["freeze_layer"]["epochs"] = "10";
            self.system["analysis"]["freeze_layer"]["analysis"] = "";


            self.system["analysis"]["optimizer_lr"] = {};
            self.system["analysis"]["optimizer_lr"]["analysis_name"] = "analyse_optimizer_lr";
            self.system["analysis"]["optimizer_lr"]["list"] = "optimizer_sgd, 0.001, optimizer_adadelta, 0.001";
            self.system["analysis"]["optimizer_lr"]["percent"] = "30";
            self.system["analysis"]["optimizer_lr"]["epochs"] = "10";
            self.system["analysis"]["optimizer_lr"]["analysis"] = "";
















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
        # Cloning not required with pip based mon installation
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
            #self.process.start("wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rG-U1mS8hDU7_wM56a1kc-li_zHLtbq2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rG-U1mS8hDU7_wM56a1kc-li_zHLtbq2\" -O datasets.zip && rm -rf /tmp/cookies.txt");
            
        else:
            self.te1.setText("Monk Classification Library Cloned.");
            self.te1.setText("System State Normal.");        

        

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