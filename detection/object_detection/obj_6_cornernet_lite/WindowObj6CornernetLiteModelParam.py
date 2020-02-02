import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowObj6CornernetLiteModelParam(QtWidgets.QWidget):

    backward_6_cornernet_lite_valdata_param = QtCore.pyqtSignal();
    forward_hyper_param = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Cornernet Lite - Model Param'
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 400
        self.load_cfg();
        self.initUI()
        

    def load_cfg(self):
        if(os.path.isfile("obj_6_cornernet_lite.json")):
            with open('obj_6_cornernet_lite.json') as json_file:
                self.system = json.load(json_file)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);

        # Forward
        self.b1 = QPushButton('Next', self)
        self.b1.move(300,350)
        self.b1.clicked.connect(self.forward)

        # Backward
        self.b2 = QPushButton('Back', self)
        self.b2.move(200,350)
        self.b2.clicked.connect(self.backward)

        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(400,350)
        self.b3.clicked.connect(self.close)

        self.l1 = QLabel(self);
        self.l1.setText("1. Model :");
        self.l1.move(20, 20);

        self.cb1 = QComboBox(self);
        self.models = ["CornerNet_Saccade", "CornerNet_Squeeze"];
        self.cb1.addItems(self.models);
        index = self.cb1.findText(self.system["model"], QtCore.Qt.MatchFixedString)
        if index >= 0:
             self.cb1.setCurrentIndex(index)
        self.cb1.move(120, 20);


        



    def forward(self):
        self.system["model"] = str(self.cb1.currentText())

        with open('obj_6_cornernet_lite.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.forward_hyper_param.emit();

    def backward(self):
        self.system["model"] = str(self.cb1.currentText())

        with open('obj_6_cornernet_lite.json', 'w') as outfile:
            json.dump(self.system, outfile)

        self.backward_6_cornernet_lite_valdata_param.emit();


'''
app = QApplication(sys.argv)
screen = WindowObj6CornernetLiteModelParam()
screen.show()
sys.exit(app.exec_())
'''