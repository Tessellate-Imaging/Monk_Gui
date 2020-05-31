import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateTrain(QtWidgets.QWidget):


    backward_loss_param = QtCore.pyqtSignal();
    forward_infer = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Train'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 600
        self.initUI()

    def cfg_setup(self):
        with open('base_classification.json') as json_file:
            self.system = json.load(json_file)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);


        # Backward
        self.b1 = QPushButton('Back', self)
        self.b1.move(600, 550)
        self.b1.clicked.connect(self.backward)

        # Backward
        self.b2 = QPushButton('Infer', self)
        self.b2.move(700, 550)
        self.b2.clicked.connect(self.forward)

        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(800, 550)
        self.b3.clicked.connect(self.close)


        self.l4 = QLabel(self);
        self.l4.setText("Epochs:");
        self.l4.move(20, 20);

        self.e4 = QLineEdit(self)
        self.e4.move(100, 20);
        self.e4.setText(self.system["epochs"]);
        self.e4.resize(200, 25);

        self.b4 = QPushButton('Train', self)
        self.b4.move(20, 50)
        self.b4.clicked.connect(self.train)

        self.b5 = QPushButton('Stop', self)
        self.b5.move(110, 50)
        self.b5.clicked.connect(self.stop)

        self.b6 = QPushButton('Resume', self)
        self.b6.move(200, 50)
        self.b6.clicked.connect(self.resume)

        self.tb1 = QTextEdit(self)
        self.tb1.move(20, 80)
        self.tb1.resize(200, 25)
        self.tb1.setText("Not Started");
        self.tb1.setReadOnly(True)

        self.te1 = QTextBrowser(self);
        self.te1.move(20, 125);
        self.te1.setFixedSize(300, 460);

        self.cb5 = QComboBox(self);
        self.cb5.addItems(["Accuracies", "Losses"]);
        self.cb5.move(400, 10);
        self.cb5.activated.connect(self.show_trainval_curves)

        self.l6 = QLabel(self)
        self.l6.move(350, 40);
        self.l6.resize(500, 500)

        self.l7 = QLabel(self)
        self.l7.move(350, 40);
        self.l7.resize(500, 500)

        if self.cb5.currentText() == "Accuracies":
            self.l6.show();
            self.l7.hide();
        else:
            self.l6.hide();
            self.l7.show();


        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.readyReadStandardError.connect(self.stderrReady)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.finished.connect(self.finished)
        
    

    def show_trainval_curves(self):
        if self.cb5.currentText() == "Accuracies":
            self.l6.show();
            self.l7.hide();
        else:
            self.l6.hide();
            self.l7.show();


    def train(self):
        self.move_ahead = True;
        if(os.path.isdir("workspace/" + self.system["project"] + "/" + self.system["experiment"])):
            qm = QMessageBox(self)
            ret = qm.question(self, 'Warning', "Experiment " + self.system["experiment"] + " exists. Do you want to delete and overwrite it?", qm.Yes | qm.No)
            if(ret == qm.Yes):
                self.move_ahead = True;
            else:
                self.move_ahead = False;

        if(self.move_ahead):
            self.te1.setText("");
            self.tb1.setText("Running");
            self.system["epochs"] = self.e4.text();
            with open('base_classification.json', 'w') as outfile:
                json.dump(self.system, outfile)

            if self.system["datatype"] == "image" and self.system["labeltype"] == "single":
                os.system("cp cfg/classification/update/train_cls_image_single.py .");
                os.system("cp cfg/classification/update/train_cls_image_single.sh .");

            self.process.start('bash', ['train_cls_image_single.sh'])
            self.append("Process PID: " + str(self.process.pid()) + "\n");

    def stop(self):
        self.tb1.setText("Interrupted");
        self.process.kill();
        self.append("Training Stopped\n")
        QMessageBox.about(self, "Training Status", "Interrupted");

    def resume(self):
        self.move_ahead = False;
        if(not os.path.isfile("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/experiment_state.json")):
            QMessageBox.about(self, "Warning", "File {} not found. Run a training session first.".format("workspace/" 
                + self.system["project"] + "/" + self.system["experiment"] + "/experiment_state.json"));
        else:
            self.move_ahead = True;
        
        if(self.move_ahead):
            self.te1.setText("");
            self.tb1.setText("Running");
            self.system["epochs"] = self.e4.text();
            with open('base_classification.json', 'w') as outfile:
                json.dump(self.system, outfile)

            if self.system["datatype"] == "image" and self.system["labeltype"] == "single":
                os.system("cp cfg/classification/update/resume_cls_image_single.py .");
                os.system("cp cfg/classification/update/resume_cls_image_single.sh .");

            self.process.start('bash', ['resume_cls_image_single.sh'])
            self.append("Process PID: " + str(self.process.pid()) + "\n");


    def finished(self):
        pass;


    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')
        if("Completed" in text):
            QMessageBox.about(self, "Training Status", "Completed");
            self.tb1.setText("Completed");
        if("Error" in text or "error" in text or "ImportError" in text):
            self.tb1.setText("Errors Found");
        if("Epoch" in text):
            plots = [];
            if(self.system["backend"] == "Keras-2.2.5_Tensorflow-1"):
                if(os.path.isfile("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/model_history_log.csv")):
                    f = open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/model_history_log.csv", 'r');
                    data = f.read();
                    f.close();
                    if("acc" in data):
                        df = pd.read_csv("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/model_history_log.csv");
                        train_acc = df["acc"].tolist();
                        train_loss = df["loss"].tolist();
                        val_acc = df["val_acc"].tolist();
                        val_loss = df["val_loss"].tolist();

                        self.create_train_val_plots([train_acc, val_acc], "accuracy", "workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/");
                        self.create_train_val_plots([train_loss, val_loss], "loss", "workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/");

                        img = Image.open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_accuracy.png");
                        img = img.resize((500, 500)) 
                        img.save("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_accuracy.png")
                        pixmap = QPixmap("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_accuracy.png")
                        pixmap = pixmap.scaledToWidth(500)
                        pixmap = pixmap.scaledToHeight(500)
                        self.l6.setPixmap(pixmap)

                        img = Image.open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_loss.png");
                        img = img.resize((500, 500)) 
                        img.save("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_loss.png")
                        pixmap = QPixmap("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_loss.png")
                        pixmap = pixmap.scaledToWidth(500)
                        pixmap = pixmap.scaledToHeight(500)
                        self.l7.setPixmap(pixmap)

            else:
                if(os.path.isfile("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_acc_history.npy")):
                    img = Image.open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_accuracy.png");
                    img = img.resize((500, 500)) 
                    img.save("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_accuracy.png")
                    pixmap = QPixmap("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_accuracy.png")
                    pixmap = pixmap.scaledToWidth(500)
                    pixmap = pixmap.scaledToHeight(500)
                    self.l6.setPixmap(pixmap)

                    img = Image.open("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_loss.png");
                    img = img.resize((500, 500)) 
                    img.save("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_loss.png")
                    pixmap = QPixmap("workspace/" + self.system["project"] + "/" + self.system["experiment"] + "/output/logs/train_val_loss.png")
                    pixmap = pixmap.scaledToWidth(500)
                    pixmap = pixmap.scaledToHeight(500)
                    self.l7.setPixmap(pixmap)
        self.append(text)



    def create_train_val_plots(self, plots, label, base_path):
        plt.plot(plots[0], marker='o', label='Training')
        plt.plot(plots[1], marker='x', label='Validation')
        plt.gca().legend(('Training','Validation'))
        plt.xlabel("Eoch Num");
        plt.ylabel(label);
        file_name = base_path + "train_val_" + label + ".png" 
        plt.savefig(file_name)
        plt.clf()



    def stderrReady(self):
        text = str(self.process.readAllStandardError().data(), encoding='utf-8')
        QMessageBox.about(self, "Training Status", "Errors Found");
        self.tb1.setText("Errors Found");
        self.append(text)


    def append(self, text):
        cursor = self.te1.textCursor()  
        self.te1.ensureCursorVisible() 
        cursor.movePosition(cursor.End)
        cursor.insertText(text)


    def forward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_infer.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_loss_param.emit();


'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateTrain()
screen.show()
sys.exit(app.exec_())
'''