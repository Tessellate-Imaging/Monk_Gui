import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


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
        self.r1.move(20, 20)
        self.r1.toggled.connect(self.quick);

        self.r2 = QRadioButton("Update Mode", self)
        self.r2.move(220, 20)
        self.r2.toggled.connect(self.update);

        self.r3 = QRadioButton("Expert Mode", self)
        self.r3.move(420, 20)
        self.r3.toggled.connect(self.expert);
        

        self.tb1 = QTextEdit(self)
        self.tb1.move(20, 50)
        self.tb1.setText("");
        self.tb1.resize(500, 390)
        self.tb1.setReadOnly(True)
        
        if self.system["mode"] == "quick":
            self.r1.setChecked(True)
            self.quick();
        elif self.system["mode"] == "update":
            self.r2.setChecked(True)
            self.update();
        elif self.system["mode"] == "expert":
            self.r3.setChecked(True)
            self.expert();



    def quick(self):
        wr = "";
        wr += "Information\n";
        wr += "1. Desgined with default values of hyper-parameters\n";
        wr += "2. Only few modifiable parameters.\n\n";
        wr += "Use Case\n";
        wr += "1. Quick prototyping\n"
        wr += "2. Transfer learning\n"
        wr += "3. Good starting point in this field\n";
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
        wr += "In Development\n\n\n";
        wr += "Information\n";
        wr += "1. Desgined with default, yet updatable values of hyper-parameters\n\n";
        wr += "Use Case\n";
        wr += "1. Quick prototyping and debugging parameters\n"
        wr += "2. Transfer learning or build custom CNNs \n"
        wr += "3. For understanding role of hyper-parameters in deep learning\n\n";
        wr += "Modifiable Elements\n";
        wr += "1. Dataset\n";
        wr += "2. Dataset params - \n";
        wr += "    - Data input size\n";
        wr += "    - Batch size\n";
        wr += "    - Data shuffling\n";
        wr += "    - Num CPU processors\n";
        wr += "    - Train-Val Split\n";
        wr += "3. Data Transformations\n";
        wr += "4. Transfer learning base model or build custom CNNs\n";
        wr += "5. Model params\n";
        wr += "    - Use Gpu/Cpu\n";
        wr += "    - Use pretrained weights or not\n";
        wr += "    - Freeze Base network or not\n";
        wr += "    - Freeze certain layers in network\n";
        wr += "6. Append layers to transfer learning model\n";
        wr += "7. Loss functions\n";
        wr += "8. Optimizers\n";
        wr += "9. Schedulers\n";
        wr += "10. Epochs\n\n"

        self.tb1.setText(wr);

        self.system["mode"] = "update";
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def expert(self):
        wr = "Development yet to start";
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



app = QApplication(sys.argv)
screen = WindowClassificationExperimentRunMode()
screen.show()
sys.exit(app.exec_())
