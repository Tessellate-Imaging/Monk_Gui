import os
import sys
import json
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class WindowClassificationComparisonCurrent(QtWidgets.QWidget):

    backward_csl_main = QtCore.pyqtSignal();
    forward_project = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Monk Classification'
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 600
        self.cfg_setup()
        self.initUI()

    def cfg_setup(self):
        if(os.path.isfile("base_classification.json")):
            with open('base_classification.json') as json_file:
                self.system = json.load(json_file)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);

        # Backward
        self.b2 = QPushButton('Back', self)
        self.b2.move(700, 550)
        self.b2.clicked.connect(self.backward)


        # Quit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(800, 550)
        self.b3.clicked.connect(self.close)


        self.te1 = QTextBrowser(self);
        self.te1.move(20, 500);
        self.te1.setFixedSize(500, 90);


        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.readyReadStandardError.connect(self.stderrReady)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.finished.connect(self.finished)


        os.system("cp cfg/classification/comparison/compare.sh .");
        os.system("cp cfg/classification/comparison/compare.py .")
        self.process.start('bash', ['compare.sh'])
        self.append("Process PID: " + str(self.process.pid()) + "\n");


        self.createLayout_Container();
        self.scrollarea.hide();


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

        image_list = ["workspace/comparison/Sample_Comparison/train_accuracy.png",
                "workspace/comparison/Sample_Comparison/train_loss.png",
                "workspace/comparison/Sample_Comparison/val_accuracy.png",
                "workspace/comparison/Sample_Comparison/val_loss.png",
                "workspace/comparison/Sample_Comparison/stats_training_time.png",
                "workspace/comparison/Sample_Comparison/stats_max_gpu_usage.png",
                "workspace/comparison/Sample_Comparison/stats_best_val_acc.png"
                ]

        for i in range(len(image_list)):
            self.layout_SArea.addWidget(self.createLayout_group(label_list[i], image_list[i]))
        self.layout_SArea.addStretch(1)

        self.scrollarea.move(10, 10)



    def stop(self):
        self.process.kill();
        self.append("Prediction Stopped\n")


    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput().data(), encoding='utf-8')

        if("Completed" in text):
            QMessageBox.about(self, "Training Status", "Completed");
            self.createLayout_Container();
            self.scrollarea.show()

        self.append(text)

    def finished(self):
        pass;
        #self.tb7.setText(self.predictions);


    def stderrReady(self):
        text = str(self.process.readAllStandardError().data(), encoding='utf-8')
        self.append(text)

    def append(self, text):
        cursor = self.te1.textCursor()  
        self.te1.ensureCursorVisible() 
        cursor.movePosition(cursor.End)
        cursor.insertText(text)



        

    def backward(self):
        with open('base_classification.json', 'w') as outfile:
            json.dump(self.system, outfile)
        self.backward_csl_main.emit();


'''
app = QApplication(sys.argv)
screen = WindowClassificationComparisonCurrent()
screen.show()
sys.exit(app.exec_())
'''