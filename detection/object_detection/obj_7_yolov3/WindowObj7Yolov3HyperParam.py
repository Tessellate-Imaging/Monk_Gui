import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowObj7Yolov3HyperParam(QtWidgets.QWidget):

    backward_model_param = QtCore.pyqtSignal();
    forward_train = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Yolo V3 - Hyper Param'
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 400
        self.load_cfg();
        self.initUI()


    def load_cfg(self):
        if(os.path.isfile("obj_7_yolov3.json")):
            with open('obj_7_yolov3.json') as json_file:
                self.system = json.load(json_file)


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);

        # Forward
        self.b1 = QPushButton('Next', self)
        self.b1.move(300, 360)
        self.b1.clicked.connect(self.forward)

        # Backward
        self.b2 = QPushButton('Back', self)
        self.b2.move(200, 360)
        self.b2.clicked.connect(self.backward)


        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(400, 360)
        self.b3.clicked.connect(self.close)


        self.l1 = QLabel(self);
        self.l1.setText("1. learning_rate:");
        self.l1.move(20, 20);

        self.e1 = QLineEdit(self)
        self.e1.move(150, 20);
        self.e1.setText(self.system["lr"]);
        self.e1.resize(200, 25);


        self.l2 = QLabel(self);
        self.l2.setText("2. Optimizer :");
        self.l2.move(20, 70);

        self.cb2 = QComboBox(self);
        self.optimizer = ["sgd", "adam"];
        self.cb2.addItems(self.optimizer);
        index = self.cb2.findText(self.system["optimizer"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb2.setCurrentIndex(index)
        self.cb2.move(120, 70);
        self.cb2.activated.connect(self.select_optimizer);


        self.l3 = QLabel(self);
        self.l3.setText("3. Multi Scale :");
        self.l3.move(20, 120);

        self.cb3 = QComboBox(self);
        self.multi_scale = ["yes", "no"];
        self.cb3.addItems(self.multi_scale);
        index = self.cb3.findText(self.system["multi_scale"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb3.setCurrentIndex(index)
        self.cb3.move(120, 120);
        self.cb3.activated.connect(self.select_multi_scale);


        self.l4 = QLabel(self);
        self.l4.setText("3. Evolve hyperparameters :");
        self.l4.move(20, 170);

        self.cb4 = QComboBox(self);
        self.evolve = ["yes", "no"];
        self.cb4.addItems(self.evolve);
        index = self.cb4.findText(self.system["evolve"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb4.setCurrentIndex(index)
        self.cb4.move(220, 170);
        self.cb4.activated.connect(self.select_evolve);



        self.l5 = QLabel(self);
        self.l5.setText("4. Mixed precision:");
        self.l5.move(20, 220);

        self.cb5 = QComboBox(self);
        self.mixed_precision = ["yes", "no"];
        self.cb5.addItems(self.evolve);
        index = self.cb5.findText(self.system["mixed_precision"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb5.setCurrentIndex(index)
        self.cb5.move(160, 220);
        self.cb5.activated.connect(self.select_mixed_precision);



        self.l6 = QLabel(self);
        self.l6.setText("5. Cache Images:");
        self.l6.move(20, 270);

        self.cb6 = QComboBox(self);
        self.cache_images = ["yes", "no"];
        self.cb6.addItems(self.cache_images);
        index = self.cb6.findText(self.system["cache_images"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb6.setCurrentIndex(index)
        self.cb6.move(150, 270);
        self.cb6.activated.connect(self.select_cache_images);






        





    def select_optimizer(self):
        self.system["optimizer"] = self.cb2.currentText();


    def select_multi_scale(self):
        self.system["multi_scale"] = self.cb3.currentText();


    def select_evolve(self):
        self.system["evolve"] = self.cb4.currentText();


    def select_mixed_precision(self):
        self.system["mixed_precision"] = self.cb5.currentText();


    def select_cache_images(self):
        self.system["cache_images"] = self.cb6.currentText();








    def forward(self):
        self.system["lr"] = self.e1.text();
        self.system["optimizer"] = self.cb2.currentText();
        self.system["multi_scale"] = self.cb3.currentText();
        self.system["evolve"] = self.cb4.currentText();
        self.system["mixed_precision"] = self.cb5.currentText();
        self.system["cache_images"] = self.cb6.currentText();

        with open('obj_7_yolov3.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.forward_train.emit();

    def backward(self):
        self.system["lr"] = self.e1.text();
        self.system["optimizer"] = self.cb2.currentText();
        self.system["multi_scale"] = self.cb3.currentText();
        self.system["evolve"] = self.cb4.currentText();
        self.system["mixed_precision"] = self.cb5.currentText();
        self.system["cache_images"] = self.cb6.currentText();

        with open('obj_7_yolov3.json', 'w') as outfile:
            json.dump(self.system, outfile)
        
        self.backward_model_param.emit();


'''
app = QApplication(sys.argv)
screen = WindowObj7Yolov3HyperParam()
screen.show()
sys.exit(app.exec_())
'''