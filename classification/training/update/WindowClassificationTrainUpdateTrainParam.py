import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateTrainParam(QtWidgets.QWidget):

    forward_optimizer_param = QtCore.pyqtSignal();
    backward_layer_param = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Update Exoeriment training Params'.format(self.system["experiment"])
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
        self.l1.setText("1. Display inter-epoch level progress:");
        self.l1.move(20, 20);

        self.cb1 = QComboBox(self);
        self.cb1.move(280, 20);
        self.cb1.activated.connect(self.select_display_realtime_progress);
        self.items = ["True", "False"];
        self.cb1.addItems(self.items);
        index = self.cb1.findText(self.system["update"]["realtime_progress"]["value"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.cb1.setCurrentIndex(index)


        self.l2 = QLabel(self);
        self.l2.setText("2. Display general progress:");
        self.l2.move(20, 70);

        self.cb2 = QComboBox(self);
        self.cb2.move(220, 70);
        self.cb2.activated.connect(self.select_display_progress);
        self.items = ["True", "False"];
        self.cb2.addItems(self.items);
        index = self.cb2.findText(self.system["update"]["realtime_progress"]["value"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.cb2.setCurrentIndex(index)


        self.l3 = QLabel(self);
        self.l3.setText("3. Save intermediate epoch weight:");
        self.l3.move(20, 120);

        self.cb3 = QComboBox(self);
        self.cb3.move(270, 120);
        self.cb3.activated.connect(self.select_intermediate_save);
        self.items = ["True", "False"];
        self.cb3.addItems(self.items);
        index = self.cb3.findText(self.system["update"]["save_intermediate"]["value"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.cb3.setCurrentIndex(index)


        self.l4 = QLabel(self);
        self.l4.setText("4. Save training logs:");
        self.l4.move(20, 170);

        self.cb4 = QComboBox(self);
        self.cb4.move(170, 170);
        self.cb4.activated.connect(self.select_save_logs);
        self.items = ["True", "False"];
        self.cb4.addItems(self.items);
        index = self.cb4.findText(self.system["update"]["save_logs"]["value"], QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.cb4.setCurrentIndex(index)









    


    def select_display_realtime_progress(self):
        self.system["update"]["realtime_progress"]["active"] = "True";
        self.system["update"]["realtime_progress"]["value"] = self.cb1.currentText();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_display_progress(self):
        self.system["update"]["progress"]["active"] = "True";
        self.system["update"]["progress"]["value"] = self.cb2.currentText();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_intermediate_save(self):
        self.system["update"]["save_intermediate"]["active"] = "True";
        self.system["update"]["save_intermediate"]["value"] = self.cb3.currentText();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)


    def select_save_logs(self):
        self.system["update"]["save_logs"]["active"] = "True";
        self.system["update"]["save_logs"]["value"] = self.cb4.currentText();
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)




    def forward(self):        
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.forward_optimizer_param.emit();


    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_layer_param.emit();



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateTrainParam()
screen.show()
sys.exit(app.exec_())
'''