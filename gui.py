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
from detection.object_detection.obj_4_efficientdet.WindowObj4EfficientdetDataParam import WindowObj4EfficientdetDataParam
from detection.object_detection.obj_4_efficientdet.WindowObj4EfficientdetValDataParam import WindowObj4EfficientdetValDataParam
from detection.object_detection.obj_4_efficientdet.WindowObj4EfficientdetModelParam import WindowObj4EfficientdetModelParam
from detection.object_detection.obj_4_efficientdet.WindowObj4EfficientdetHyperParam import WindowObj4EfficientdetHyperParam
from detection.object_detection.obj_4_efficientdet.WindowObj4EfficientdetTrain import WindowObj4EfficientdetTrain
from detection.object_detection.obj_4_efficientdet.WindowObj4EfficientdetInfer  import WindowObj4EfficientdetInfer


from detection.object_detection.obj_5_pytorch_retinanet.WindowObj5PytorchRetinanet import WindowObj5PytorchRetinanet
from detection.object_detection.obj_5_pytorch_retinanet.WindowObj5PytorchRetinanetDataParam import WindowObj5PytorchRetinanetDataParam
from detection.object_detection.obj_5_pytorch_retinanet.WindowObj5PytorchRetinanetValDataParam import WindowObj5PytorchRetinanetValDataParam
from detection.object_detection.obj_5_pytorch_retinanet.WindowObj5PytorchRetinanetModelParam import WindowObj5PytorchRetinanetModelParam
from detection.object_detection.obj_5_pytorch_retinanet.WindowObj5PytorchRetinanetHyperParam import WindowObj5PytorchRetinanetHyperParam
from detection.object_detection.obj_5_pytorch_retinanet.WindowObj5PytorchRetinanetTrain import WindowObj5PytorchRetinanetTrain
from detection.object_detection.obj_5_pytorch_retinanet.WindowObj5PytorchRetinanetInfer import WindowObj5PytorchRetinanetInfer


from detection.object_detection.obj_6_cornernet_lite.WindowObj6CornernetLite import WindowObj6CornernetLite
from detection.object_detection.obj_6_cornernet_lite.WindowObj6CornernetLiteDataParam import WindowObj6CornernetLiteDataParam
from detection.object_detection.obj_6_cornernet_lite.WindowObj6CornernetLiteValDataParam import WindowObj6CornernetLiteValDataParam
from detection.object_detection.obj_6_cornernet_lite.WindowObj6CornernetLiteModelParam import WindowObj6CornernetLiteModelParam
from detection.object_detection.obj_6_cornernet_lite.WindowObj6CornernetLiteHyperParam import WindowObj6CornernetLiteHyperParam
from detection.object_detection.obj_6_cornernet_lite.WindowObj6CornernetLiteTrain import WindowObj6CornernetLiteTrain
from detection.object_detection.obj_6_cornernet_lite.WindowObj6CornernetLiteInfer import WindowObj6CornernetLiteInfer



from detection.object_detection.obj_7_yolov3.WindowObj7Yolov3 import WindowObj7Yolov3
from detection.object_detection.obj_7_yolov3.WindowObj7Yolov3DataParam import WindowObj7Yolov3DataParam
from detection.object_detection.obj_7_yolov3.WindowObj7Yolov3ValDataParam import WindowObj7Yolov3ValDataParam
from detection.object_detection.obj_7_yolov3.WindowObj7Yolov3ModelParam import WindowObj7Yolov3ModelParam
from detection.object_detection.obj_7_yolov3.WindowObj7Yolov3HyperParam import WindowObj7Yolov3HyperParam
from detection.object_detection.obj_7_yolov3.WindowObj7Yolov3Train import WindowObj7Yolov3Train
from detection.object_detection.obj_7_yolov3.WindowObj7Yolov3Infer import WindowObj7Yolov3Infer


from classification.project.WindowClassificationProjectMain import WindowClassificationProjectMain
from classification.project.WindowClassificationExperimentMain import WindowClassificationExperimentMain
from classification.project.WindowClassificationExperimentRunMode import WindowClassificationExperimentRunMode

from classification.training.quick.WindowClassificationTrainQuickDataParam import WindowClassificationTrainQuickDataParam
from classification.training.quick.WindowClassificationTrainQuickModelParam import WindowClassificationTrainQuickModelParam
from classification.training.quick.WindowClassificationTrainQuickTrain import WindowClassificationTrainQuickTrain

from classification.training.update.WindowClassificationTrainUpdateDataParam import WindowClassificationTrainUpdateDataParam
from classification.training.update.WindowClassificationTrainUpdateTransformParam import WindowClassificationTrainUpdateTransformParam
from classification.training.update.WindowClassificationTrainUpdateModelParam import WindowClassificationTrainUpdateModelParam
from classification.training.update.WindowClassificationTrainUpdateLayerParam import WindowClassificationTrainUpdateLayerParam
from classification.training.update.WindowClassificationTrainUpdateTrainParam import WindowClassificationTrainUpdateTrainParam
from classification.training.update.WindowClassificationTrainUpdateOptimizerParam import WindowClassificationTrainUpdateOptimizerParam
from classification.training.update.WindowClassificationTrainUpdateSchedulerParam import WindowClassificationTrainUpdateSchedulerParam
from classification.training.update.WindowClassificationTrainUpdateLossParam import WindowClassificationTrainUpdateLossParam
from classification.training.update.WindowClassificationTrainUpdateTrain import WindowClassificationTrainUpdateTrain

from classification.training.update.WindowClassificationTrainUpdateAnalysisInputSize import WindowClassificationTrainUpdateAnalysisInputSize
from classification.training.update.WindowClassificationTrainUpdateAnalysisInputSizeVisualize import WindowClassificationTrainUpdateAnalysisInputSizeVisualize
from classification.training.update.WindowClassificationTrainUpdateAnalysisBatchSize import WindowClassificationTrainUpdateAnalysisBatchSize
from classification.training.update.WindowClassificationTrainUpdateAnalysisBatchSizeVisualize import WindowClassificationTrainUpdateAnalysisBatchSizeVisualize
from classification.training.update.WindowClassificationTrainUpdateAnalysisTrainvalSplit import WindowClassificationTrainUpdateAnalysisTrainvalSplit
from classification.training.update.WindowClassificationTrainUpdateAnalysisTrainvalSplitVisualize import WindowClassificationTrainUpdateAnalysisTrainvalSplitVisualize

from classification.training.update.WindowClassificationTrainUpdateAnalysisModelList import WindowClassificationTrainUpdateAnalysisModelList
from classification.training.update.WindowClassificationTrainUpdateAnalysisModelListVisualize import WindowClassificationTrainUpdateAnalysisModelListVisualize
from classification.training.update.WindowClassificationTrainUpdateAnalysisUsePretrained import WindowClassificationTrainUpdateAnalysisUsePretrained
from classification.training.update.WindowClassificationTrainUpdateAnalysisUsePretrainedVisualize import WindowClassificationTrainUpdateAnalysisUsePretrainedVisualize
from classification.training.update.WindowClassificationTrainUpdateAnalysisFreezeBase import WindowClassificationTrainUpdateAnalysisFreezeBase
from classification.training.update.WindowClassificationTrainUpdateAnalysisFreezeBaseVisualize import WindowClassificationTrainUpdateAnalysisFreezeBaseVisualize
from classification.training.update.WindowClassificationTrainUpdateAnalysisFreezeLayer import WindowClassificationTrainUpdateAnalysisFreezeLayer
from classification.training.update.WindowClassificationTrainUpdateAnalysisFreezeLayerVisualize import WindowClassificationTrainUpdateAnalysisFreezeLayerVisualize
from classification.training.update.WindowClassificationTrainUpdateAnalysisOptimizerLr import WindowClassificationTrainUpdateAnalysisOptimizerLr
from classification.training.update.WindowClassificationTrainUpdateAnalysisOptimizerLrVisualize import WindowClassificationTrainUpdateAnalysisOptimizerLrVisualize

from classification.infer.quick.WindowClassificationInferQuick import WindowClassificationInferQuick
from classification.infer.update.WindowClassificationInferUpdate import WindowClassificationInferUpdate


from classification.validate.quick.WindowClassificationValidateQuick import WindowClassificationValidateQuick
from classification.validate.update.WindowClassificationValidateUpdate import WindowClassificationValidateUpdate

from classification.comparison.WindowClassificationComparisonCurrent import WindowClassificationComparisonCurrent



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

        #forward
        self.window_classification_main.forward_project.connect(self.show_cls_project_main)

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

        #forward
        self.obj_4_efficientdet.forward_train.connect(self.show_obj_4_efficientdet_data_param)
        self.obj_4_efficientdet.forward_infer.connect(self.show_obj_4_efficientdet_infer)

        #backward
        self.obj_4_efficientdet.backward_obj.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_4_efficientdet);

        #show_current_window
        self.obj_4_efficientdet.show();


    def show_obj_4_efficientdet_data_param(self):
        self.obj_4_efficientdet_data_param = WindowObj4EfficientdetDataParam();

        #forward
        self.obj_4_efficientdet_data_param.forward_valdata_param.connect(self.show_obj_4_efficientdet_valdata_param)

        #backward
        self.obj_4_efficientdet_data_param.backward_4_efficientdet.connect(self.show_obj_4_efficientdet);


        #close other 
        self.close_other_windows(self.obj_4_efficientdet_data_param);

        #show_current_window
        self.obj_4_efficientdet_data_param.show();


    def show_obj_4_efficientdet_valdata_param(self):
        self.obj_4_efficientdet_valdata_param = WindowObj4EfficientdetValDataParam();

        #forward
        self.obj_4_efficientdet_valdata_param.forward_model_param.connect(self.show_obj_4_efficientdet_model_param)

        #backward
        self.obj_4_efficientdet_valdata_param.backward_4_efficientdet_data_preproc.connect(self.show_obj_4_efficientdet_data_param);


        #close other 
        self.close_other_windows(self.obj_4_efficientdet_valdata_param);

        #show_current_window
        self.obj_4_efficientdet_valdata_param.show();



    def show_obj_4_efficientdet_model_param(self):
        self.obj_4_efficientdet_model_param = WindowObj4EfficientdetModelParam();

        #forward
        self.obj_4_efficientdet_model_param.forward_hyper_param.connect(self.show_obj_4_efficientdet_hyper_param);

        #backward
        self.obj_4_efficientdet_model_param.backward_4_efficientdet_valdata_param.connect(self.show_obj_4_efficientdet_valdata_param);


        #close other 
        self.close_other_windows(self.obj_4_efficientdet_model_param);

        #show_current_window
        self.obj_4_efficientdet_model_param.show();



    def show_obj_4_efficientdet_hyper_param(self):
        self.obj_4_efficientdet_hyper_param = WindowObj4EfficientdetHyperParam();

        #forward
        self.obj_4_efficientdet_hyper_param.forward_train.connect(self.show_obj_4_efficientdet_train)

        #backward
        self.obj_4_efficientdet_hyper_param.backward_model_param.connect(self.show_obj_4_efficientdet_model_param);


        #close other 
        self.close_other_windows(self.obj_4_efficientdet_hyper_param);

        #show_current_window
        self.obj_4_efficientdet_hyper_param.show();


    def show_obj_4_efficientdet_train(self):
        self.obj_4_efficientdet_train = WindowObj4EfficientdetTrain();

        #forward
        self.obj_4_efficientdet_train.forward_4_efficientdet.connect(self.show_obj_4_efficientdet)

        #backward
        self.obj_4_efficientdet_train.backward_hyper_param.connect(self.show_obj_4_efficientdet_hyper_param);


        #close other 
        self.close_other_windows(self.obj_4_efficientdet_train);

        #show_current_window
        self.obj_4_efficientdet_train.show();



    def show_obj_4_efficientdet_infer(self):
        self.obj_4_efficientdet_infer = WindowObj4EfficientdetInfer();

        #backward
        self.obj_4_efficientdet_infer.backward_4_efficientdet.connect(self.show_obj_4_efficientdet);


        #close other 
        self.close_other_windows(self.obj_4_efficientdet_infer);

        #show_current_window
        self.obj_4_efficientdet_infer.show();












    def show_obj_5_pytorch_retinanet(self):
        self.obj_5_pytorch_retinanet = WindowObj5PytorchRetinanet();

        #forward
        self.obj_5_pytorch_retinanet.forward_train.connect(self.show_obj_5_pytorch_retinanet_data_param)
        self.obj_5_pytorch_retinanet.forward_infer.connect(self.show_obj_5_pytorch_retinanet_infer)

        #backward
        self.obj_5_pytorch_retinanet.backward_obj.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_5_pytorch_retinanet);

        #show_current_window
        self.obj_5_pytorch_retinanet.show();


    def show_obj_5_pytorch_retinanet_data_param(self):
        self.obj_5_pytorch_retinanet_data_param = WindowObj5PytorchRetinanetDataParam();

        #forward
        self.obj_5_pytorch_retinanet_data_param.forward_valdata_param.connect(self.show_obj_5_pytorch_retinanet_valdata_param)

        #backward
        self.obj_5_pytorch_retinanet_data_param.backward_5_pytorch_retinanet.connect(self.show_obj_5_pytorch_retinanet);


        #close other 
        self.close_other_windows(self.obj_5_pytorch_retinanet_data_param);

        #show_current_window
        self.obj_5_pytorch_retinanet_data_param.show();


    def show_obj_5_pytorch_retinanet_valdata_param(self):
        self.obj_5_pytorch_retinanet_valdata_param = WindowObj5PytorchRetinanetValDataParam();

        #forward
        self.obj_5_pytorch_retinanet_valdata_param.forward_model_param.connect(self.show_obj_5_pytorch_retinanet_model_param)

        #backward
        self.obj_5_pytorch_retinanet_valdata_param.backward_5_pytorch_retinanet_data_preproc.connect(self.show_obj_5_pytorch_retinanet_data_param);


        #close other 
        self.close_other_windows(self.obj_5_pytorch_retinanet_valdata_param);

        #show_current_window
        self.obj_5_pytorch_retinanet_valdata_param.show();


    def show_obj_5_pytorch_retinanet_model_param(self):
        self.obj_5_pytorch_retinanet_model_param = WindowObj5PytorchRetinanetModelParam();

        #forward
        self.obj_5_pytorch_retinanet_model_param.forward_hyper_param.connect(self.show_obj_5_pytorch_retinanet_hyper_param);

        #backward
        self.obj_5_pytorch_retinanet_model_param.backward_5_pytorch_retinanet_valdata_param.connect(self.show_obj_5_pytorch_retinanet_valdata_param);


        #close other 
        self.close_other_windows(self.obj_5_pytorch_retinanet_model_param);

        #show_current_window
        self.obj_5_pytorch_retinanet_model_param.show();


    def show_obj_5_pytorch_retinanet_hyper_param(self):
        self.obj_5_pytorch_retinanet_hyper_param = WindowObj5PytorchRetinanetHyperParam();

        #forward
        self.obj_5_pytorch_retinanet_hyper_param.forward_train.connect(self.show_obj_5_pytorch_retinanet_train)

        #backward
        self.obj_5_pytorch_retinanet_hyper_param.backward_model_param.connect(self.show_obj_5_pytorch_retinanet_model_param);


        #close other 
        self.close_other_windows(self.obj_5_pytorch_retinanet_hyper_param);

        #show_current_window
        self.obj_5_pytorch_retinanet_hyper_param.show();


    def show_obj_5_pytorch_retinanet_train(self):
        self.obj_5_pytorch_retinanet_train = WindowObj5PytorchRetinanetTrain();

        #forward
        self.obj_5_pytorch_retinanet_train.forward_5_pytorch_retinanet.connect(self.show_obj_5_pytorch_retinanet)

        #backward
        self.obj_5_pytorch_retinanet_train.backward_hyper_param.connect(self.show_obj_5_pytorch_retinanet_hyper_param);


        #close other 
        self.close_other_windows(self.obj_5_pytorch_retinanet_train);

        #show_current_window
        self.obj_5_pytorch_retinanet_train.show();


    def show_obj_5_pytorch_retinanet_infer(self):
        self.obj_5_pytorch_retinanet_infer = WindowObj5PytorchRetinanetInfer();

        #backward
        self.obj_5_pytorch_retinanet_infer.backward_5_pytorch_retinanet.connect(self.show_obj_5_pytorch_retinanet);


        #close other 
        self.close_other_windows(self.obj_5_pytorch_retinanet_infer);

        #show_current_window
        self.obj_5_pytorch_retinanet_infer.show();















    def show_obj_6_cornernet_lite(self):
        self.obj_6_cornernet_lite = WindowObj6CornernetLite();

        #forward
        self.obj_6_cornernet_lite.forward_train.connect(self.show_obj_6_cornernet_lite_data_param)
        self.obj_6_cornernet_lite.forward_infer.connect(self.show_obj_6_cornernet_lite_infer)

        #backward
        self.obj_6_cornernet_lite.backward_obj.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_6_cornernet_lite);

        #show_current_window
        self.obj_6_cornernet_lite.show();


    def show_obj_6_cornernet_lite_data_param(self):
        self.obj_6_cornernet_lite_data_param = WindowObj6CornernetLiteDataParam();

        #forward
        self.obj_6_cornernet_lite_data_param.forward_valdata_param.connect(self.show_obj_6_cornernet_lite_valdata_param)

        #backward
        self.obj_6_cornernet_lite_data_param.backward_6_cornernet_lite.connect(self.show_obj_6_cornernet_lite);


        #close other 
        self.close_other_windows(self.obj_6_cornernet_lite_data_param);

        #show_current_window
        self.obj_6_cornernet_lite_data_param.show();


    def show_obj_6_cornernet_lite_valdata_param(self):
        self.obj_6_cornernet_lite_valdata_param = WindowObj6CornernetLiteValDataParam();

        #forward
        self.obj_6_cornernet_lite_valdata_param.forward_model_param.connect(self.show_obj_6_cornernet_lite_model_param)

        #backward
        self.obj_6_cornernet_lite_valdata_param.backward_6_cornernet_lite_data_preproc.connect(self.show_obj_6_cornernet_lite_data_param);


        #close other 
        self.close_other_windows(self.obj_6_cornernet_lite_valdata_param);

        #show_current_window
        self.obj_6_cornernet_lite_valdata_param.show();


    def show_obj_6_cornernet_lite_model_param(self):
        self.obj_6_cornernet_lite_model_param = WindowObj6CornernetLiteModelParam();

        #forward
        self.obj_6_cornernet_lite_model_param.forward_hyper_param.connect(self.show_obj_6_cornernet_lite_hyper_param);

        #backward
        self.obj_6_cornernet_lite_model_param.backward_6_cornernet_lite_valdata_param.connect(self.show_obj_6_cornernet_lite_valdata_param);


        #close other 
        self.close_other_windows(self.obj_6_cornernet_lite_model_param);

        #show_current_window
        self.obj_6_cornernet_lite_model_param.show();


    def show_obj_6_cornernet_lite_hyper_param(self):
        self.obj_6_cornernet_lite_hyper_param = WindowObj6CornernetLiteHyperParam();

        #forward
        self.obj_6_cornernet_lite_hyper_param.forward_train.connect(self.show_obj_6_cornernet_lite_train)

        #backward
        self.obj_6_cornernet_lite_hyper_param.backward_model_param.connect(self.show_obj_6_cornernet_lite_model_param);


        #close other 
        self.close_other_windows(self.obj_6_cornernet_lite_hyper_param);

        #show_current_window
        self.obj_6_cornernet_lite_hyper_param.show();


    def show_obj_6_cornernet_lite_train(self):
        self.obj_6_cornernet_lite_train = WindowObj6CornernetLiteTrain();

        #forward
        self.obj_6_cornernet_lite_train.forward_6_cornernet_lite.connect(self.show_obj_6_cornernet_lite)

        #backward
        self.obj_6_cornernet_lite_train.backward_hyper_param.connect(self.show_obj_6_cornernet_lite_hyper_param);


        #close other 
        self.close_other_windows(self.obj_6_cornernet_lite_train);

        #show_current_window
        self.obj_6_cornernet_lite_train.show();


    def show_obj_6_cornernet_lite_infer(self):
        self.obj_6_cornernet_lite_infer = WindowObj6CornernetLiteInfer();

        #backward
        self.obj_6_cornernet_lite_infer.backward_6_cornernet_lite.connect(self.show_obj_6_cornernet_lite);


        #close other 
        self.close_other_windows(self.obj_6_cornernet_lite_infer);

        #show_current_window
        self.obj_6_cornernet_lite_infer.show();
















    def show_obj_7_yolov3(self):
        self.obj_7_yolov3 = WindowObj7Yolov3();

        #forward
        self.obj_7_yolov3.forward_train.connect(self.show_obj_7_yolov3_data_param)
        self.obj_7_yolov3.forward_infer.connect(self.show_obj_7_yolov3_infer)

        #backward
        self.obj_7_yolov3.backward_obj.connect(self.show_detection_main);

        #close other 
        self.close_other_windows(self.obj_7_yolov3);

        #show_current_window
        self.obj_7_yolov3.show();


    def show_obj_7_yolov3_data_param(self):
        self.obj_7_yolov3_data_param = WindowObj7Yolov3DataParam();

        #forward
        self.obj_7_yolov3_data_param.forward_valdata_param.connect(self.show_obj_7_yolov3_valdata_param)

        #backward
        self.obj_7_yolov3_data_param.backward_7_yolov3.connect(self.show_obj_7_yolov3);


        #close other 
        self.close_other_windows(self.obj_7_yolov3_data_param);

        #show_current_window
        self.obj_7_yolov3_data_param.show();


    def show_obj_7_yolov3_valdata_param(self):
        self.obj_7_yolov3_valdata_param = WindowObj7Yolov3ValDataParam();

        #forward
        self.obj_7_yolov3_valdata_param.forward_model_param.connect(self.show_obj_7_yolov3_model_param)

        #backward
        self.obj_7_yolov3_valdata_param.backward_7_yolov3_data_preproc.connect(self.show_obj_7_yolov3_data_param);


        #close other 
        self.close_other_windows(self.obj_7_yolov3_valdata_param);

        #show_current_window
        self.obj_7_yolov3_valdata_param.show();


    def show_obj_7_yolov3_model_param(self):
        self.obj_7_yolov3_model_param = WindowObj7Yolov3ModelParam();

        #forward
        self.obj_7_yolov3_model_param.forward_hyper_param.connect(self.show_obj_7_yolov3_hyper_param);

        #backward
        self.obj_7_yolov3_model_param.backward_7_yolov3_valdata_param.connect(self.show_obj_7_yolov3_valdata_param);


        #close other 
        self.close_other_windows(self.obj_7_yolov3_model_param);

        #show_current_window
        self.obj_7_yolov3_model_param.show();


    def show_obj_7_yolov3_hyper_param(self):
        self.obj_7_yolov3_hyper_param = WindowObj7Yolov3HyperParam();

        #forward
        self.obj_7_yolov3_hyper_param.forward_train.connect(self.show_obj_7_yolov3_train)

        #backward
        self.obj_7_yolov3_hyper_param.backward_model_param.connect(self.show_obj_7_yolov3_model_param);


        #close other 
        self.close_other_windows(self.obj_7_yolov3_hyper_param);

        #show_current_window
        self.obj_7_yolov3_hyper_param.show();


    def show_obj_7_yolov3_train(self):
        self.obj_7_yolov3_train = WindowObj7Yolov3Train();

        #forward
        self.obj_7_yolov3_train.forward_7_yolov3.connect(self.show_obj_7_yolov3)

        #backward
        self.obj_7_yolov3_train.backward_hyper_param.connect(self.show_obj_7_yolov3_hyper_param);


        #close other 
        self.close_other_windows(self.obj_7_yolov3_train);

        #show_current_window
        self.obj_7_yolov3_train.show();


    def show_obj_7_yolov3_infer(self):
        self.obj_7_yolov3_infer = WindowObj7Yolov3Infer();

        #backward
        self.obj_7_yolov3_infer.backward_7_yolov3.connect(self.show_obj_7_yolov3);


        #close other 
        self.close_other_windows(self.obj_7_yolov3_infer);

        #show_current_window
        self.obj_7_yolov3_infer.show();









    def show_cls_project_main(self):
        self.cls_project_main = WindowClassificationProjectMain();

        #forward
        self.cls_project_main.forward_experiment.connect(self.show_cls_experiment_main);
        self.cls_project_main.forward_compare_current.connect(self.show_cls_comparison_current);
        self.cls_project_main.forward_copy_experiment.connect(self.show_cls_experiment_run_mode);

        #backward
        self.cls_project_main.backward_classification_main.connect(self.show_classification_main);

        #close other 
        self.close_other_windows(self.cls_project_main);

        #show_current_window
        self.cls_project_main.show();


    def show_cls_experiment_main(self):
        self.cls_experiment_main = WindowClassificationExperimentMain();

        #forward
        self.cls_experiment_main.forward_run_mode.connect(self.show_cls_experiment_run_mode)
        self.cls_experiment_main.forward_infer.connect(self.show_cls_quick_infer)
        self.cls_experiment_main.forward_validate.connect(self.show_cls_quick_validate)

        #backward
        self.cls_experiment_main.backward_project_main.connect(self.show_cls_project_main);

        #close other 
        self.close_other_windows(self.cls_experiment_main);

        #show_current_window
        self.cls_experiment_main.show();


    def show_cls_experiment_run_mode(self):
        self.cls_experiment_run_mode = WindowClassificationExperimentRunMode();

        #forward
        self.cls_experiment_run_mode.forward_data_param.connect(self.show_cls_quick_train_data_param)

        #backward
        self.cls_experiment_run_mode.backward_experiment_main.connect(self.show_cls_experiment_main);

        #close other 
        self.close_other_windows(self.cls_experiment_run_mode);

        #show_current_window
        self.cls_experiment_run_mode.show();


    def show_cls_quick_train_data_param(self):
        self.cls_quick_train_data_param = WindowClassificationTrainQuickDataParam();

        #forward
        self.cls_quick_train_data_param.forward_model_param.connect(self.show_cls_quick_train_model_param)

        #backward
        self.cls_quick_train_data_param.backward_experiment_run_mode.connect(self.show_cls_experiment_run_mode);

        #close other 
        self.close_other_windows(self.cls_quick_train_data_param);

        #show_current_window
        self.cls_quick_train_data_param.show();


    def show_cls_quick_train_model_param(self):
        self.cls_quick_train_model_param = WindowClassificationTrainQuickModelParam();

        #forward1
        self.cls_quick_train_model_param.forward_train.connect(self.show_cls_quick_train)

        #forward2
        self.cls_quick_train_model_param.forward_update_data_param.connect(self.show_cls_update_train_data_param)

        #backward
        self.cls_quick_train_model_param.backward_data_param.connect(self.show_cls_quick_train_data_param);

        #close other 
        self.close_other_windows(self.cls_quick_train_model_param);

        #show_current_window
        self.cls_quick_train_model_param.show();


    def show_cls_quick_train(self):
        self.cls_quick_train = WindowClassificationTrainQuickTrain();

        #forward
        self.cls_quick_train.forward_infer.connect(self.show_cls_experiment_main)

        #backward
        self.cls_quick_train.backward_model_param.connect(self.show_cls_quick_train_model_param);

        #close other 
        self.close_other_windows(self.cls_quick_train);

        #show_current_window
        self.cls_quick_train.show();


    def show_cls_update_train_data_param(self):
        self.cls_update_train_data_param = WindowClassificationTrainUpdateDataParam();

        #forward
        self.cls_update_train_data_param.forward_transform_param.connect(self.show_cls_update_train_transform_param)
        self.cls_update_train_data_param.forward_analyse_input_size.connect(self.show_cls_update_train_analyse_input_size)
        self.cls_update_train_data_param.forward_analyse_batch_size.connect(self.show_cls_update_train_analyse_batch_size)
        self.cls_update_train_data_param.forward_analyse_trainval_split.connect(self.show_cls_update_train_analyse_trainval_split)


        #backward
        self.cls_update_train_data_param.backward_model_param.connect(self.show_cls_quick_train_model_param);

        #close other 
        self.close_other_windows(self.cls_update_train_data_param);

        #show_current_window
        self.cls_update_train_data_param.show();

    

    def show_cls_update_train_analyse_input_size(self):
        self.cls_update_train_analyse_input_size = WindowClassificationTrainUpdateAnalysisInputSize();

        #forward
        self.cls_update_train_analyse_input_size.forward_visualize.connect(self.show_cls_update_train_analyse_input_size_visualize);

        #backward
        self.cls_update_train_analyse_input_size.backward_data_param.connect(self.show_cls_update_train_data_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_input_size);

        #show_current_window
        self.cls_update_train_analyse_input_size.show();


    def show_cls_update_train_analyse_input_size_visualize(self):
        self.cls_update_train_analyse_input_size_visualize = WindowClassificationTrainUpdateAnalysisInputSizeVisualize();

        #forward
        #none

        #backward1
        self.cls_update_train_analyse_input_size_visualize.backward_analyse.connect(self.show_cls_update_train_analyse_input_size);

        #backward2
        self.cls_update_train_analyse_input_size_visualize.backward_data_param.connect(self.show_cls_update_train_data_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_input_size_visualize);

        #show_current_window
        self.cls_update_train_analyse_input_size_visualize.show();


    def show_cls_update_train_analyse_batch_size(self):
        self.cls_update_train_analyse_batch_size = WindowClassificationTrainUpdateAnalysisBatchSize();

        #forward
        self.cls_update_train_analyse_batch_size.forward_visualize.connect(self.show_cls_update_train_analyse_batch_size_visualize);

        #backward
        self.cls_update_train_analyse_batch_size.backward_data_param.connect(self.show_cls_update_train_data_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_batch_size);

        #show_current_window
        self.cls_update_train_analyse_batch_size.show();


    def show_cls_update_train_analyse_batch_size_visualize(self):
        self.cls_update_train_analyse_batch_size_visualize = WindowClassificationTrainUpdateAnalysisBatchSizeVisualize();

        #forward
        #none

        #backward1
        self.cls_update_train_analyse_batch_size_visualize.backward_analyse.connect(self.show_cls_update_train_analyse_batch_size);

        #backward2
        self.cls_update_train_analyse_batch_size_visualize.backward_data_param.connect(self.show_cls_update_train_data_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_batch_size_visualize);

        #show_current_window
        self.cls_update_train_analyse_batch_size_visualize.show();


    def show_cls_update_train_analyse_trainval_split(self):
        self.cls_update_train_analyse_trainval_split = WindowClassificationTrainUpdateAnalysisTrainvalSplit();

        #forward
        self.cls_update_train_analyse_trainval_split.forward_visualize.connect(self.show_cls_update_train_analyse_trainval_split_visualize);

        #backward
        self.cls_update_train_analyse_trainval_split.backward_data_param.connect(self.show_cls_update_train_data_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_trainval_split);

        #show_current_window
        self.cls_update_train_analyse_trainval_split.show();


    def show_cls_update_train_analyse_trainval_split_visualize(self):
        self.cls_update_train_analyse_trainval_split_visualize = WindowClassificationTrainUpdateAnalysisTrainvalSplitVisualize();

        #forward
        #none

        #backward1
        self.cls_update_train_analyse_trainval_split_visualize.backward_analyse.connect(self.show_cls_update_train_analyse_trainval_split);

        #backward2
        self.cls_update_train_analyse_trainval_split_visualize.backward_data_param.connect(self.show_cls_update_train_data_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_trainval_split_visualize);

        #show_current_window
        self.cls_update_train_analyse_trainval_split_visualize.show();


    def show_cls_update_train_transform_param(self):
        self.cls_update_train_transform_param = WindowClassificationTrainUpdateTransformParam();

        #forward
        self.cls_update_train_transform_param.forward_model_param.connect(self.show_cls_update_train_model_param)

        #backward
        self.cls_update_train_transform_param.backward_data_param.connect(self.show_cls_update_train_data_param);

        #close other 
        self.close_other_windows(self.cls_update_train_transform_param);

        #show_current_window
        self.cls_update_train_transform_param.show();


    def show_cls_update_train_model_param(self):
        self.cls_update_train_model_param = WindowClassificationTrainUpdateModelParam();

        #forward
        self.cls_update_train_model_param.forward_layer_param.connect(self.show_cls_update_train_layer_param)
        self.cls_update_train_model_param.forward_analyse_model_list.connect(self.show_cls_update_train_analyse_model_list)
        self.cls_update_train_model_param.forward_analyse_use_pretrained.connect(self.show_cls_update_train_analyse_use_pretrained)
        self.cls_update_train_model_param.forward_analyse_freeze_base.connect(self.show_cls_update_train_analyse_freeze_base)
        self.cls_update_train_model_param.forward_analyse_freeze_layer.connect(self.show_cls_update_train_analyse_freeze_layer)


        #backward
        self.cls_update_train_model_param.backward_transform_param.connect(self.show_cls_update_train_transform_param);

        #close other 
        self.close_other_windows(self.cls_update_train_model_param);

        #show_current_window
        self.cls_update_train_model_param.show();

    def show_cls_update_train_analyse_model_list(self):
        self.cls_update_train_analyse_model_list = WindowClassificationTrainUpdateAnalysisModelList();

        #forward
        self.cls_update_train_analyse_model_list.forward_visualize.connect(self.show_cls_update_train_analyse_model_list_visualize);

        #backward
        self.cls_update_train_analyse_model_list.backward_model_param.connect(self.show_cls_update_train_model_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_model_list);

        #show_current_window
        self.cls_update_train_analyse_model_list.show();

    def show_cls_update_train_analyse_model_list_visualize(self):
        self.cls_update_train_analyse_model_list_visualize = WindowClassificationTrainUpdateAnalysisModelListVisualize();

        #forward
        #none

        #backward1
        self.cls_update_train_analyse_model_list_visualize.backward_analyse.connect(self.show_cls_update_train_analyse_model_list);

        #backward2
        self.cls_update_train_analyse_model_list_visualize.backward_model_param.connect(self.show_cls_update_train_model_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_model_list_visualize);

        #show_current_window
        self.cls_update_train_analyse_model_list_visualize.show();

    def show_cls_update_train_analyse_use_pretrained(self):
        self.cls_update_train_analyse_use_pretrained = WindowClassificationTrainUpdateAnalysisUsePretrained();

        #forward
        self.cls_update_train_analyse_use_pretrained.forward_visualize.connect(self.show_cls_update_train_analyse_use_pretrained_visualize);

        #backward
        self.cls_update_train_analyse_use_pretrained.backward_model_param.connect(self.show_cls_update_train_model_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_use_pretrained);

        #show_current_window
        self.cls_update_train_analyse_use_pretrained.show();

    def show_cls_update_train_analyse_use_pretrained_visualize(self):
        self.cls_update_train_analyse_use_pretrained_visualize = WindowClassificationTrainUpdateAnalysisUsePretrainedVisualize();

        #forward
        #none

        #backward1
        self.cls_update_train_analyse_use_pretrained_visualize.backward_analyse.connect(self.show_cls_update_train_analyse_use_pretrained);

        #backward2
        self.cls_update_train_analyse_use_pretrained_visualize.backward_model_param.connect(self.show_cls_update_train_model_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_use_pretrained_visualize);

        #show_current_window
        self.cls_update_train_analyse_use_pretrained_visualize.show();


    def show_cls_update_train_analyse_freeze_base(self):
        self.cls_update_train_analyse_freeze_base = WindowClassificationTrainUpdateAnalysisFreezeBase();

        #forward
        self.cls_update_train_analyse_freeze_base.forward_visualize.connect(self.show_cls_update_train_analyse_freeze_base_visualize);

        #backward
        self.cls_update_train_analyse_freeze_base.backward_model_param.connect(self.show_cls_update_train_model_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_freeze_base);

        #show_current_window
        self.cls_update_train_analyse_freeze_base.show();

    def show_cls_update_train_analyse_freeze_base_visualize(self):
        self.cls_update_train_analyse_freeze_base_visualize = WindowClassificationTrainUpdateAnalysisFreezeBaseVisualize();

        #forward
        #none

        #backward1
        self.cls_update_train_analyse_freeze_base_visualize.backward_analyse.connect(self.show_cls_update_train_analyse_freeze_base);

        #backward2
        self.cls_update_train_analyse_freeze_base_visualize.backward_model_param.connect(self.show_cls_update_train_model_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_freeze_base_visualize);

        #show_current_window
        self.cls_update_train_analyse_freeze_base_visualize.show();


    def show_cls_update_train_analyse_freeze_layer(self):
        self.cls_update_train_analyse_freeze_layer = WindowClassificationTrainUpdateAnalysisFreezeLayer();

        #forward
        self.cls_update_train_analyse_freeze_layer.forward_visualize.connect(self.show_cls_update_train_analyse_freeze_layer_visualize);

        #backward
        self.cls_update_train_analyse_freeze_layer.backward_model_param.connect(self.show_cls_update_train_model_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_freeze_layer);

        #show_current_window
        self.cls_update_train_analyse_freeze_layer.show();

    def show_cls_update_train_analyse_freeze_layer_visualize(self):
        self.cls_update_train_analyse_freeze_layer_visualize = WindowClassificationTrainUpdateAnalysisFreezeLayerVisualize();

        #forward
        #none

        #backward1
        self.cls_update_train_analyse_freeze_layer_visualize.backward_analyse.connect(self.show_cls_update_train_analyse_freeze_layer);

        #backward2
        self.cls_update_train_analyse_freeze_layer_visualize.backward_model_param.connect(self.show_cls_update_train_model_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_freeze_layer_visualize);

        #show_current_window
        self.cls_update_train_analyse_freeze_layer_visualize.show();


    def show_cls_update_train_layer_param(self):
        self.cls_update_train_layer_param = WindowClassificationTrainUpdateLayerParam();

        #forward
        self.cls_update_train_layer_param.forward_train_param.connect(self.show_cls_update_train_train_param)

        #backward
        self.cls_update_train_layer_param.backward_model_param.connect(self.show_cls_update_train_model_param);

        #close other 
        self.close_other_windows(self.cls_update_train_layer_param);

        #show_current_window
        self.cls_update_train_layer_param.show();



    def show_cls_update_train_train_param(self):
        self.cls_update_train_train_param = WindowClassificationTrainUpdateTrainParam();

        #forward
        self.cls_update_train_train_param.forward_optimizer_param.connect(self.show_cls_update_train_optimizer_param)

        #backward
        self.cls_update_train_train_param.backward_layer_param.connect(self.show_cls_update_train_layer_param);

        #close other 
        self.close_other_windows(self.cls_update_train_train_param);

        #show_current_window
        self.cls_update_train_train_param.show();


    def show_cls_update_train_optimizer_param(self):
        self.cls_update_train_optimizer_param = WindowClassificationTrainUpdateOptimizerParam();

        #forward
        self.cls_update_train_optimizer_param.forward_scheduler_param.connect(self.show_cls_update_train_scheduler_param)
        self.cls_update_train_optimizer_param.forward_analyse_optimizer_lr.connect(self.show_cls_update_train_analyse_optimizer_lr)

        #backward
        self.cls_update_train_optimizer_param.backward_train_param.connect(self.show_cls_update_train_train_param);

        #close other 
        self.close_other_windows(self.cls_update_train_optimizer_param);

        #show_current_window
        self.cls_update_train_optimizer_param.show();

    def show_cls_update_train_analyse_optimizer_lr(self):
        self.cls_update_train_analyse_optimizer_lr = WindowClassificationTrainUpdateAnalysisOptimizerLr();

        #forward
        self.cls_update_train_analyse_optimizer_lr.forward_visualize.connect(self.show_cls_update_train_analyse_optimizer_lr_visualize);

        #backward
        self.cls_update_train_analyse_optimizer_lr.backward_optimizer_param.connect(self.show_cls_update_train_optimizer_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_optimizer_lr);

        #show_current_window
        self.cls_update_train_analyse_optimizer_lr.show();

    def show_cls_update_train_analyse_optimizer_lr_visualize(self):
        self.cls_update_train_analyse_optimizer_lr_visualize = WindowClassificationTrainUpdateAnalysisOptimizerLrVisualize();

        #forward
        #none

        #backward1
        self.cls_update_train_analyse_optimizer_lr_visualize.backward_analyse.connect(self.show_cls_update_train_analyse_optimizer_lr);

        #backward2
        self.cls_update_train_analyse_optimizer_lr_visualize.backward_optimizer_param.connect(self.show_cls_update_train_optimizer_param);

        #close other 
        self.close_other_windows(self.cls_update_train_analyse_optimizer_lr_visualize);

        #show_current_window
        self.cls_update_train_analyse_optimizer_lr_visualize.show();


    def show_cls_update_train_scheduler_param(self):
        self.cls_update_train_scheduler_param = WindowClassificationTrainUpdateSchedulerParam();

        #forward
        self.cls_update_train_scheduler_param.forward_loss_param.connect(self.show_cls_update_train_loss_param)

        #backward
        self.cls_update_train_scheduler_param.backward_optimizer_param.connect(self.show_cls_update_train_train_param);

        #close other 
        self.close_other_windows(self.cls_update_train_scheduler_param);

        #show_current_window
        self.cls_update_train_scheduler_param.show();


    def show_cls_update_train_loss_param(self):
        self.cls_update_train_loss_param = WindowClassificationTrainUpdateLossParam();

        #forward
        self.cls_update_train_loss_param.forward_train.connect(self.show_cls_update_train)

        #backward
        self.cls_update_train_loss_param.backward_scheduler_param.connect(self.show_cls_update_train_scheduler_param);

        #close other 
        self.close_other_windows(self.cls_update_train_loss_param);

        #show_current_window
        self.cls_update_train_loss_param.show();


    def show_cls_update_train(self):
        self.cls_update_train = WindowClassificationTrainUpdateTrain();

        #forward
        self.cls_update_train.forward_infer.connect(self.show_cls_experiment_main)

        #backward
        self.cls_update_train.backward_loss_param.connect(self.show_cls_update_train_loss_param);

        #close other 
        self.close_other_windows(self.cls_update_train);

        #show_current_window
        self.cls_update_train.show();


    def show_cls_quick_infer(self):
        self.cls_quick_infer = WindowClassificationInferQuick();

        #backward
        self.cls_quick_infer.backward_exp.connect(self.show_cls_experiment_main);

        #close other 
        self.close_other_windows(self.cls_quick_infer);

        #show_current_window
        self.cls_quick_infer.show();


    def show_cls_update_infer(self):
        self.cls_update_infer = WindowClassificationInferUpdate();

        #backward
        self.cls_update_infer.backward_exp.connect(self.show_cls_experiment_main);

        #close other 
        self.close_other_windows(self.cls_update_infer);

        #show_current_window
        self.cls_update_infer.show();


    def show_cls_quick_validate(self):
        self.cls_quick_validate = WindowClassificationValidateQuick();

        #backward
        self.cls_quick_validate.backward_exp.connect(self.show_cls_experiment_main);

        #close other 
        self.close_other_windows(self.cls_quick_validate);

        #show_current_window
        self.cls_quick_validate.show();

    def show_cls_update_validate(self):
        self.cls_update_validate = WindowClassificationValidateUpdate();

        #backward
        self.cls_update_validate.backward_exp.connect(self.show_cls_experiment_main);

        #close other 
        self.close_other_windows(self.cls_update_validate);

        #show_current_window
        self.cls_update_validate.show();


    def show_cls_comparison_current(self):
        self.cls_comparison_current = WindowClassificationComparisonCurrent();

        #backward
        self.cls_comparison_current.backward_csl_main.connect(self.show_cls_project_main);

        #close other 
        self.close_other_windows(self.cls_comparison_current);

        #show_current_window
        self.cls_comparison_current.show();



app = QtWidgets.QApplication(sys.argv)
controller = Controller()
controller.show_main()
sys.exit(app.exec_())