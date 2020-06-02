import os
import sys
import json
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *



class WindowClassificationTrainUpdateAnalysisFreezeBaseVisualize(QtWidgets.QWidget):

    backward_model_param = QtCore.pyqtSignal();
    backward_analyse = QtCore.pyqtSignal();


    def __init__(self):
        super().__init__()
        self.cfg_setup()
        self.title = 'Experiment {} - Visualize Analysis of freezing base network'.format(self.system["experiment"])
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 600
        self.initUI()

    def cfg_setup(self):
        with open('base_classification.json') as json_file:
            self.system = json.load(json_file)

        self.comparison_name = "Comparison_" + self.system["analysis"]["freeze_base"]["analysis_name"]

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);


        # Backward
        self.b1 = QPushButton('Back To Analysis', self)
        self.b1.move(400,550)
        self.b1.clicked.connect(self.backward1)

        # Forward
        self.b2 = QPushButton('Back to Update model param', self)
        self.b2.move(550,550)
        self.b2.clicked.connect(self.backward2)

        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(800,550)
        self.b3.clicked.connect(self.close)


        self.createLayout_Container();


    def createLayout_group(self, label, img_file):
        sgroupbox = QGroupBox("Graph - {}:".format(label), self)
        layout_groupbox = QVBoxLayout(sgroupbox)

        l1 = QLabel(self)
        l1.resize(700, 450)
        layout_groupbox.addWidget(l1);
        
        if(os.path.isfile(img_file)):
            img = Image.open(img_file);
            img = img.resize((700, 450));
            img_file = img_file.split(".")[0] + "_.png"; 
            img.save(img_file)

        pixmap = QPixmap(img_file)
        l1.setPixmap(pixmap)

        return sgroupbox




    def createLayout_Container(self):
        self.scrollarea = QScrollArea(self)
        self.scrollarea.setFixedSize(700, 480)
        self.scrollarea.setWidgetResizable(True)

        widget = QWidget()
        self.scrollarea.setWidget(widget)
        self.layout_SArea = QVBoxLayout(widget)

        label_list = ["Train Accuracy", "Train Loss", "Validation Accuracy", 
                        "Validation Loss", "Training Time", "Gpu Usage", 
                        "Best validation accuracy"]

        image_list = ["workspace/comparison/" + self.comparison_name + "/train_accuracy.png",
                "workspace/comparison/" + self.comparison_name + "/train_loss.png",
                "workspace/comparison/" + self.comparison_name + "/val_accuracy.png",
                "workspace/comparison/" + self.comparison_name + "/val_loss.png",
                "workspace/comparison/" + self.comparison_name + "/stats_training_time.png",
                "workspace/comparison/" + self.comparison_name + "/stats_max_gpu_usage.png",
                "workspace/comparison/" + self.comparison_name + "/stats_best_val_acc.png"
                ]

        for i in range(len(image_list)):
            self.layout_SArea.addWidget(self.createLayout_group(label_list[i], image_list[i]))
        self.layout_SArea.addStretch(1)

        self.scrollarea.move(10, 10)


        
        



    def backward1(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_analyse.emit();


    def backward2(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_model_param.emit();


    



'''
app = QApplication(sys.argv)
screen = WindowClassificationTrainUpdateAnalysisFreezeBaseVisualize()
screen.show()
sys.exit(app.exec_())
'''