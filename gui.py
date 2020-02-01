import os
import sys
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from WindowMain import WindowMain
from classification.WindowClassificationMain import WindowClassificationMain
from detection.WindowDetectionMain import WindowDetectionMain

from detection.object_detection.obj_1_gluoncv_finetune.WindowObj1GluoncvFinetune import WindowObj1GluoncvFinetune
from detection.object_detection.obj_1_gluoncv_finetune.WindowObj1GluoncvFinetuneDataParam import WindowObj1GluoncvFinetuneDataParam
from detection.object_detection.obj_1_gluoncv_finetune.WindowObj1GluoncvFinetuneModelParam import WindowObj1GluoncvFinetuneModelParam
from detection.object_detection.obj_1_gluoncv_finetune.WindowObj1GluoncvFinetuneHyperParam import WindowObj1GluoncvFinetuneHyperParam
from detection.object_detection.obj_1_gluoncv_finetune.WindowObj1GluoncvFinetuneTrain import WindowObj1GluoncvFinetuneTrain
from detection.object_detection.obj_1_gluoncv_finetune.WindowObj1GluoncvFinetuneInfer import WindowObj1GluoncvFinetuneInfer



from detection.object_detection.obj_2_pytorch_finetune.WindowObj2PytorchFinetune import WindowObj2PytorchFinetune
from detection.object_detection.obj_2_pytorch_finetune.WindowObj2PytorchFinetuneDataParam import WindowObj2PytorchFinetuneDataParam
from detection.object_detection.obj_2_pytorch_finetune.WindowObj2PytorchFinetuneValDataParam import WindowObj2PytorchFinetuneValDataParam
from detection.object_detection.obj_2_pytorch_finetune.WindowObj2PytorchFinetuneModelParam import WindowObj2PytorchFinetuneModelParam
from detection.object_detection.obj_2_pytorch_finetune.WindowObj2PytorchFinetuneHyperParam import WindowObj2PytorchFinetuneHyperParam
from detection.object_detection.obj_2_pytorch_finetune.WindowObj2PytorchFinetuneTrain import WindowObj2PytorchFinetuneTrain
from detection.object_detection.obj_2_pytorch_finetune.WindowObj2PytorchFinetuneInfer import WindowObj2PytorchFinetuneInfer


from detection.object_detection.obj_3_mxrcnn.WindowObj3Mxrcnn import WindowObj3Mxrcnn
from detection.object_detection.obj_3_mxrcnn.WindowObj3MxrcnnDataParam import WindowObj3MxrcnnDataParam
from detection.object_detection.obj_3_mxrcnn.WindowObj3MxrcnnDataPreproc import WindowObj3MxrcnnDataPreproc
from detection.object_detection.obj_3_mxrcnn.WindowObj3MxrcnnModelParam import WindowObj3MxrcnnModelParam
from detection.object_detection.obj_3_mxrcnn.WindowObj3MxrcnnHyperParam import WindowObj3MxrcnnHyperParam
from detection.object_detection.obj_3_mxrcnn.WindowObj3MxrcnnTrain import WindowObj3MxrcnnTrain
from detection.object_detection.obj_3_mxrcnn.WindowObj3MxrcnnInfer import WindowObj3MxrcnnInfer


from detection.object_detection.obj_4_efficientdet.WindowObj4Efficientdet import WindowObj4Efficientdet
from detection.object_detection.obj_5_pytorch_retinanet.WindowObj5PytorchRetinanet import WindowObj5PytorchRetinanet
from detection.object_detection.obj_6_cornernet_lite.WindowObj6CornernetLite import WindowObj6CornernetLite
from detection.object_detection.obj_7_yolov3.WindowObj7Yolov3 import WindowObj7Yolov3



class WindowMain(QtWidgets.QWidget):

    forward_classification = QtCore.pyqtSignal();
    forward_detection = QtCore.pyqtSignal();

    def __init__(self):
        super().__init__()
        self.title = 'Monk'
        self.left = 100
        self.top = 100
        self.width = 300
        self.height = 200
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height);

        # Image classification
        self.b1 = QPushButton('Image Classification', self)
        self.b1.move(30,10)
        self.b1.clicked.connect(self.image_classification);

        # Object Detection
        self.b1 = QPushButton('Object Detection', self)
        self.b1.move(30,50)
        self.b1.clicked.connect(self.object_detection);

        # Exit
        self.b3 = QPushButton('Quit', self)
        self.b3.move(100,150)
        self.b3.clicked.connect(self.close)


    def image_classification(self):
        self.forward_classification.emit();

    def object_detection(self):
        self.forward_detection.emit();











class Controller:

    def __init__(self):
        self.windows = [];


    def close_other_windows(self, window):
        for i in range(len(self.windows)):
            self.windows[i].close();
        self.windows = [];
        self.windows.append(window);


    def show_main(self):
        self.window_main = WindowMain();

        #forwards
        self.window_main.forward_classification.connect(self.show_classification_main); 
        self.window_main.forward_detection.connect(self.show_detection_main);
        
        #close other 
        self.close_other_windows(self.window_main);
        
        #show_current_window
        self.window_main.show(); 


    def show_classification_main(self):
        self.window_classification_main = WindowClassificationMain();

        #backward
        self.window_classification_main.backward_main.connect(self.show_main);

        #close other 
        self.close_other_windows(self.window_classification_main);

        #show_current_window
        self.window_classification_main.show(); 


    def show_detection_main(self):
        self.window_detection_main = WindowDetectionMain();

        #forwards
        self.window_detection_main.forward_obj_1_gluoncv_finetune.connect(self.show_obj_1_gluoncv_finetune)
        self.window_detection_main.forward_obj_2_pytorch_finetune.connect(self.show_obj_2_pytorch_finetune)
        self.window_detection_main.forward_obj_3_mxrcnn.connect(self.show_obj_3_mxrcnn)
        self.window_detection_main.forward_obj_4_efficientdet.connect(self.show_obj_4_efficientdet)
        self.window_detection_main.forward_obj_5_pytorch_retinanet.connect(self.show_obj_5_pytorch_retinanet)
        self.window_detection_main.forward_obj_6_cornernet_lite.connect(self.show_obj_6_cornernet_lite)
        self.window_detection_main.forward_obj_7_yolov3.connect(self.show_obj_7_yolov3)

        #backward
        self.window_detection_main.backward_main.connect(self.show_main);

        #close other 
        self.close_other_windows(self.window_detection_main);

        #show_current_window
        self.window_detection_main.show(); 




    # Object Detection Windows
    def show_obj_1_gluoncv_finetune(self):
        self.obj_1_gluoncv_finetune = WindowObj1GluoncvFinetune();

        #forward
        self.obj_1_gluoncv_finetune.forward_train.connect(self.show_obj_1_gluoncv_finetune_data_param)
        self.obj_1_gluoncv_finetune.forward_infer.connect(self.show_obj_1_gluoncv_finetune_infer)

        #backward
        self.obj_1_gluoncv_finetune.backward_obj.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_1_gluoncv_finetune);

        #show_current_window
        self.obj_1_gluoncv_finetune.show();


    def show_obj_1_gluoncv_finetune_data_param(self):
        self.obj_1_gluoncv_finetune_data_param = WindowObj1GluoncvFinetuneDataParam();

        #forward
        self.obj_1_gluoncv_finetune_data_param.forward_model_param.connect(self.show_obj_1_gluoncv_finetune_model_param)

        #backward
        self.obj_1_gluoncv_finetune_data_param.backward_1_gluoncv_finetune.connect(self.show_obj_1_gluoncv_finetune);


        #close other 
        self.close_other_windows(self.obj_1_gluoncv_finetune_data_param);

        #show_current_window
        self.obj_1_gluoncv_finetune_data_param.show();


    def show_obj_1_gluoncv_finetune_model_param(self):
        self.obj_1_gluoncv_finetune_model_param = WindowObj1GluoncvFinetuneModelParam();

        #forward
        self.obj_1_gluoncv_finetune_model_param.forward_hyper_param.connect(self.show_obj_1_gluoncv_finetune_hyper_param);

        #backward
        self.obj_1_gluoncv_finetune_model_param.backward_1_gluoncv_finetune_data_param.connect(self.show_obj_1_gluoncv_finetune_data_param);


        #close other 
        self.close_other_windows(self.obj_1_gluoncv_finetune_model_param);

        #show_current_window
        self.obj_1_gluoncv_finetune_model_param.show();


    def show_obj_1_gluoncv_finetune_hyper_param(self):
        self.obj_1_gluoncv_finetune_hyper_param = WindowObj1GluoncvFinetuneHyperParam();

        #forward
        self.obj_1_gluoncv_finetune_hyper_param.forward_train.connect(self.show_obj_1_gluoncv_finetune_train)

        #backward
        self.obj_1_gluoncv_finetune_hyper_param.backward_model_model_param.connect(self.show_obj_1_gluoncv_finetune_model_param);


        #close other 
        self.close_other_windows(self.obj_1_gluoncv_finetune_hyper_param);

        #show_current_window
        self.obj_1_gluoncv_finetune_hyper_param.show();


    def show_obj_1_gluoncv_finetune_train(self):
        self.obj_1_gluoncv_finetune_train = WindowObj1GluoncvFinetuneTrain();

        #forward
        self.obj_1_gluoncv_finetune_train.forward_1_gluoncv_finetune.connect(self.show_obj_1_gluoncv_finetune)

        #backward
        self.obj_1_gluoncv_finetune_train.backward_hyper_param.connect(self.show_obj_1_gluoncv_finetune_hyper_param);


        #close other 
        self.close_other_windows(self.obj_1_gluoncv_finetune_train);

        #show_current_window
        self.obj_1_gluoncv_finetune_train.show();


    def show_obj_1_gluoncv_finetune_infer(self):
        self.obj_1_gluoncv_finetune_infer = WindowObj1GluoncvFinetuneInfer();

        #backward
        self.obj_1_gluoncv_finetune_infer.backward_1_gluoncv_finetune.connect(self.show_obj_1_gluoncv_finetune);


        #close other 
        self.close_other_windows(self.obj_1_gluoncv_finetune_infer);

        #show_current_window
        self.obj_1_gluoncv_finetune_infer.show();








    def show_obj_2_pytorch_finetune(self):
        self.obj_2_pytorch_finetune = WindowObj2PytorchFinetune();

        #forward
        self.obj_2_pytorch_finetune.forward_train.connect(self.show_obj_2_pytorch_finetune_data_param)
        self.obj_2_pytorch_finetune.forward_infer.connect(self.show_obj_2_pytorch_finetune_infer)

        #backward
        self.obj_2_pytorch_finetune.backward_obj.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_2_pytorch_finetune);

        #show_current_window
        self.obj_2_pytorch_finetune.show();


    
    def show_obj_2_pytorch_finetune_data_param(self):
        self.obj_2_pytorch_finetune_data_param = WindowObj2PytorchFinetuneDataParam();

        #forward
        self.obj_2_pytorch_finetune_data_param.forward_val_data.connect(self.show_obj_2_pytorch_finetune_valdata_param)

        #backward
        self.obj_2_pytorch_finetune_data_param.backward_2_pytorch_finetune.connect(self.show_obj_2_pytorch_finetune);


        #close other 
        self.close_other_windows(self.obj_2_pytorch_finetune_data_param);

        #show_current_window
        self.obj_2_pytorch_finetune_data_param.show();


    def show_obj_2_pytorch_finetune_valdata_param(self):
        self.obj_2_pytorch_finetune_valdata_param = WindowObj2PytorchFinetuneValDataParam();

        #forward
        self.obj_2_pytorch_finetune_valdata_param.forward_model_param.connect(self.show_obj_2_pytorch_finetune_model_param)

        #backward
        self.obj_2_pytorch_finetune_valdata_param.backward_2_pytorch_finetune_data_param.connect(self.show_obj_2_pytorch_finetune_data_param);


        #close other 
        self.close_other_windows(self.obj_2_pytorch_finetune_valdata_param);

        #show_current_window
        self.obj_2_pytorch_finetune_valdata_param.show();


    def show_obj_2_pytorch_finetune_model_param(self):
        self.obj_2_pytorch_finetune_model_param = WindowObj2PytorchFinetuneModelParam();

        #forward
        self.obj_2_pytorch_finetune_model_param.forward_hyper_param.connect(self.show_obj_2_pytorch_finetune_hyper_param);

        #backward
        self.obj_2_pytorch_finetune_model_param.backward_2_pytorch_finetune_valdata_param.connect(self.show_obj_2_pytorch_finetune_valdata_param);


        #close other 
        self.close_other_windows(self.obj_2_pytorch_finetune_model_param);

        #show_current_window
        self.obj_2_pytorch_finetune_model_param.show();


    def show_obj_2_pytorch_finetune_hyper_param(self):
        self.obj_2_pytorch_finetune_hyper_param = WindowObj2PytorchFinetuneHyperParam();

        #forward
        self.obj_2_pytorch_finetune_hyper_param.forward_train.connect(self.show_obj_2_pytorch_finetune_train)

        #backward
        self.obj_2_pytorch_finetune_hyper_param.backward_model_param.connect(self.show_obj_2_pytorch_finetune_model_param);


        #close other 
        self.close_other_windows(self.obj_2_pytorch_finetune_hyper_param);

        #show_current_window
        self.obj_2_pytorch_finetune_hyper_param.show();


    def show_obj_2_pytorch_finetune_train(self):
        self.obj_2_pytorch_finetune_train = WindowObj2PytorchFinetuneTrain();

        #forward
        self.obj_2_pytorch_finetune_train.forward_2_finetune_finetune.connect(self.show_obj_2_pytorch_finetune)

        #backward
        self.obj_2_pytorch_finetune_train.backward_hyper_param.connect(self.show_obj_2_pytorch_finetune_hyper_param);


        #close other 
        self.close_other_windows(self.obj_2_pytorch_finetune_train);

        #show_current_window
        self.obj_2_pytorch_finetune_train.show();


    def show_obj_2_pytorch_finetune_infer(self):
        self.obj_2_pytorch_finetune_infer = WindowObj2PytorchFinetuneInfer();

        #backward
        self.obj_2_pytorch_finetune_infer.backward_2_pytorch_finetune.connect(self.show_obj_2_pytorch_finetune);


        #close other 
        self.close_other_windows(self.obj_2_pytorch_finetune_infer);

        #show_current_window
        self.obj_2_pytorch_finetune_infer.show();
	










    def show_obj_3_mxrcnn(self):
        self.obj_3_mxrcnn = WindowObj3Mxrcnn();

        #forward
        self.obj_3_mxrcnn.forward_train.connect(self.show_obj_3_mxrcnn_data_param)
        self.obj_3_mxrcnn.forward_infer.connect(self.show_obj_3_mxrcnn_infer)

        #backward
        self.obj_3_mxrcnn.backward_obj.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_3_mxrcnn);

        #show_current_window
        self.obj_3_mxrcnn.show();


    def show_obj_3_mxrcnn_data_param(self):
        self.obj_3_mxrcnn_data_param = WindowObj3MxrcnnDataParam();

        #forward
        self.obj_3_mxrcnn_data_param.forward_data_preproc.connect(self.show_obj_3_mxrcnn_data_preproc)

        #backward
        self.obj_3_mxrcnn_data_param.backward_3_mxrcnn.connect(self.show_obj_3_mxrcnn);


        #close other 
        self.close_other_windows(self.obj_3_mxrcnn_data_param);

        #show_current_window
        self.obj_3_mxrcnn_data_param.show();


    def show_obj_3_mxrcnn_data_preproc(self):
        self.obj_3_mxrcnn_data_preproc = WindowObj3MxrcnnDataPreproc();

        #forward
        self.obj_3_mxrcnn_data_preproc.forward_model_param.connect(self.show_obj_3_mxrcnn_model_param)

        #backward
        self.obj_3_mxrcnn_data_preproc.backward_3_mxrcnn_data_param.connect(self.show_obj_3_mxrcnn_data_param);


        #close other 
        self.close_other_windows(self.obj_3_mxrcnn_data_preproc);

        #show_current_window
        self.obj_3_mxrcnn_data_preproc.show();


    def show_obj_3_mxrcnn_model_param(self):
        self.obj_3_mxrcnn_model_param = WindowObj3MxrcnnModelParam();

        #forward
        self.obj_3_mxrcnn_model_param.forward_hyper_param.connect(self.show_obj_3_mxrcnn_hyper_param);

        #backward
        self.obj_3_mxrcnn_model_param.backward_3_mxrcnn_data_preproc.connect(self.show_obj_3_mxrcnn_data_preproc);


        #close other 
        self.close_other_windows(self.obj_3_mxrcnn_model_param);

        #show_current_window
        self.obj_3_mxrcnn_model_param.show();


    def show_obj_3_mxrcnn_hyper_param(self):
        self.obj_3_mxrcnn_hyper_param = WindowObj3MxrcnnHyperParam();

        #forward
        self.obj_3_mxrcnn_hyper_param.forward_train.connect(self.show_obj_3_mxrcnn_train)

        #backward
        self.obj_3_mxrcnn_hyper_param.backward_model_param.connect(self.show_obj_3_mxrcnn_model_param);


        #close other 
        self.close_other_windows(self.obj_3_mxrcnn_hyper_param);

        #show_current_window
        self.obj_3_mxrcnn_hyper_param.show();


    def show_obj_3_mxrcnn_train(self):
        self.obj_3_mxrcnn_train = WindowObj3MxrcnnTrain();

        #forward
        self.obj_3_mxrcnn_train.forward_3_mxrcnn.connect(self.show_obj_3_mxrcnn)

        #backward
        self.obj_3_mxrcnn_train.backward_hyper_param.connect(self.show_obj_3_mxrcnn_hyper_param);


        #close other 
        self.close_other_windows(self.obj_3_mxrcnn_train);

        #show_current_window
        self.obj_3_mxrcnn_train.show();


    def show_obj_3_mxrcnn_infer(self):
        self.obj_3_mxrcnn_infer = WindowObj3MxrcnnInfer();

        #backward
        self.obj_3_mxrcnn_infer.backward_3_mxrcnn.connect(self.show_obj_3_mxrcnn);


        #close other 
        self.close_other_windows(self.obj_3_mxrcnn_infer);

        #show_current_window
        self.obj_3_mxrcnn_infer.show();









    def show_obj_4_efficientdet(self):
        self.obj_4_efficientdet = WindowObj4Efficientdet();

        #backward
        self.obj_4_efficientdet.backward_main.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_4_efficientdet);

        #show_current_window
        self.obj_4_efficientdet.show();



    def show_obj_5_pytorch_retinanet(self):
        self.obj_5_pytorch_retinanet = WindowObj5PytorchRetinanet();

        #backward
        self.obj_5_pytorch_retinanet.backward_main.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_5_pytorch_retinanet);

        #show_current_window
        self.obj_5_pytorch_retinanet.show();



    def show_obj_6_cornernet_lite(self):
        self.obj_6_cornernet_lite = WindowObj6CornernetLite();

        #backward
        self.obj_6_cornernet_lite.backward_main.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_6_cornernet_lite);

        #show_current_window
        self.obj_6_cornernet_lite.show();



    def show_obj_7_yolov3(self):
        self.obj_7_yolov3 = WindowObj7Yolov3();

        #backward
        self.obj_7_yolov3.backward_main.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_7_yolov3);

        #show_current_window
        self.obj_7_yolov3.show();

           




app = QtWidgets.QApplication(sys.argv)
controller = Controller()
controller.show_main()
sys.exit(app.exec_())