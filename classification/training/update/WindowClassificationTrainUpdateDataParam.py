import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateDataParam(QtWidgets.QWidget):

    forward_transform_param = QtCore.pyqtSignal();
    backward_model_param = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Update Dataset Params'.format(self.system["experiment"])
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


        self.l1 = QLabel(self);
        self.l1.setText("1. Input Image size:");
        self.l1.move(20, 20);

        self.e1 = QLineEdit(self)
        self.e1.move(170, 20);
        self.e1.setText(self.system["update"]["input_size"]["value"]);



        self.l2 = QLabel(self);
        self.l2.setText("2. Batch size:");
        self.l2.move(20, 70);

        self.e2 = QLineEdit(self)
        self.e2.move(120, 70);
        self.e2.setText(self.system["update"]["batch_size"]["value"]);


        self.l3 = QLabel(self);
        self.l3.setText("3. Shuffle data:");
        self.l3.move(20, 120);

        self.cb3 = QComboBox(self);
        self.cb3.move(140, 120);
        self.cb3.addItems(["True", "False"]);
        index = self.cb3.findText(self.system["update"]["shuffle_data"]["value"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.cb3.setCurrentIndex(index)
        self.cb3.activated.connect(self.shuffle_data);



        self.l4 = QLabel(self);
        self.l4.setText("4. Num processors:");
        self.l4.move(20, 170);

        self.e4 = QLineEdit(self)
        self.e4.move(160, 170);
        self.e4.setText(self.system["update"]["num_processors"]["value"]);



        self.l5 = QLabel(self);
        self.l5.setText("5. Train validation split (0-1):");
        self.l5.move(20, 220);

        self.e5 = QLineEdit(self)
        self.e5.move(230, 220);
        self.e5.setText(self.system["update"]["trainval_split"]["value"]);




        

    

    def shuffle_data(self):
        self.system["update"]["shuffle_data"]["active"] = True;
        self.system["update"]["shuffle_data"]["value"] = self.cb3.currentText();




    def forward(self):
        if(str(self.e1.text()) != "224"):
            self.system["update"]["input_size"]["active"] = True;
            self.system["update"]["input_size"]["value"] = str(self.e1.text());
        if(str(self.e2.text()) != "4"):
            self.system["update"]["batch_size"]["active"] = True;
            self.system["update"]["batch_size"]["value"] = str(self.e2.text());
        if(self.cb3.currentText() != "True"):
            self.system["update"]["shuffle_data"]["active"] = True;
            self.system["update"]["shuffle_data"]["value"] = self.cb3.currentText();
        if(str(self.e4.text()) != "3"):
            self.system["update"]["num_processors"]["active"] = True;
            self.system["update"]["num_processors"]["value"] = str(self.e4.text());
        if(str(self.e5.text()) != "0.7"):
            self.system["update"]["trainval_split"]["active"] = True;
            self.system["update"]["trainval_split"]["value"] = str(self.e5.text());
        
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_transform_param.emit();


    def backward(self):
        if(str(self.e1.text()) != "224"):
            self.system["update"]["input_size"]["active"] = True;
            self.system["update"]["input_size"]["value"] = str(self.e1.text());
        if(str(self.e2.text()) != "4"):
            self.system["update"]["batch_size"]["active"] = True;
            self.system["update"]["batch_size"]["value"] = str(self.e2.text());
        if(self.cb3.currentText() != "True"):
            self.system["update"]["shuffle_data"]["active"] = True;
            self.system["update"]["shuffle_data"]["value"] = self.cb3.currentText();
        if(str(self.e4.text()) != "3"):
            self.system["update"]["num_processors"]["active"] = True;
            self.system["update"]["num_processors"]["value"] = str(self.e4.text());
        if(str(self.e5.text()) != "0.7"):
            self.system["update"]["trainval_split"]["active"] = True;
            self.system["update"]["trainval_split"]["value"] = str(self.e5.text());

        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_model_param.emit();



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateDataParam()
screen.show()
sys.exit(app.exec_())
'''