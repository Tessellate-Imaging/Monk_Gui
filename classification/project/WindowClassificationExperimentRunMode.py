import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

lst = [u"D", u"E", u"EF", u"F", u"FG", u"G", u"H", u"JS", u"J", u"K", u"M", u"P", u"R", u"S", u"T", u"U", u"V", u"X", u"Y", u"Z"]


class WindowClassificationExperimentRunMode(QtWidgets.QWidget):

    forward_data_param = QtCore.pyqtSignal();
    backward_experiment_main = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Set Mode'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width = 600
        self.height = 500
        self.initUI()

    def cfg_setup(self):
        with open('base_classification.json') as json_file:
            self.system = json.load(json_file)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);


        # Backward
        self.b1 = QPushButton('Back', self)
        self.b1.move(300,450)
        self.b1.clicked.connect(self.backward)

        # Backward
        self.b2 = QPushButton('Next', self)
        self.b2.move(400,450)
        self.b2.clicked.connect(self.forward)


        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(500, 450)
        self.b3.clicked.connect(self.close)


        self.r1 = QRadioButton("Quick Mode", self)
        if self.system["mode"] == "quick":
            self.r1.setChecked(True)
        self.r1.move(20, 20)
        self.r1.toggled.connect(self.quick);

        self.r2 = QRadioButton("Update Mode", self)
        if self.system["mode"] == "update":
            self.r2.setChecked(True)
        self.r2.move(220, 20)
        self.r2.toggled.connect(self.update);

        self.r3 = QRadioButton("Expert Mode", self)
        if self.system["mode"] == "expert":
            self.r3.setChecked(True)
        self.r3.move(420, 20)
        self.r3.toggled.connect(self.expert);
        

        self.tb1 = QTextEdit(self)
        self.tb1.move(20, 50)
        self.tb1.setText("");
        self.tb1.resize(500, 390)
        self.tb1.setReadOnly(True)
        self.quick();



    def quick(self):
        wr = "";
        wr += "Use Case\n";
        wr += "1. Quick prototyping\n"
        wr += "2. Transfer learning\n\n"
        wr += "Modifiable Elements\n";
        wr += "1. Dataset\n";
        wr += "2. Transfer learning base model\n";
        wr += "3. Epochs\n\n";
        wr += "Overview steps\n";
        wr += "1. Default(dataset_path=\"train\", model_name=\"resnet18_v1\", num_epochs=2)\n"
        wr += "2. Train();\n"
        self.tb1.setText(wr);

        self.system["mode"] = "quick";
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)

    def update(self):
        wr = "";
        self.tb1.setText(wr);

        self.system["mode"] = "update";
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def expert(self):
        wr = "";
        self.tb1.setText(wr);

        self.system["mode"] = "expert";
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)



    def forward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_data_param.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_experiment_main.emit();


'''
app = QApplication(sys.argv)
screen = WindowClassificationExperimentRunMode()
screen.show()
sys.exit(app.exec_())
'''