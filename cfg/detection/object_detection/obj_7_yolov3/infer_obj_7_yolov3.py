import os
import sys
import json
import time


sys.path.append("Monk_Object_Detection/7_yolov3/lib/")


from infer_detector import Infer


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

print("Predicting....")

with open('obj_7_yolov3_infer.json') as json_file:
    system = json.load(json_file)



system["conf_thresh"] = float(system["conf_thresh"])
system["iou_thresh"] = float(system["iou_thresh"])
class_file = system["class_file"];

f = open(class_file, 'r');
class_list = f.readlines();
f.close();
for i in range(len(class_list)):
    class_list[i] = class_list[i][:len(class_list[i])-1]

gtf = Infer();


gtf.Model(system["model"],
            class_list, 
            system["weights"],
            use_gpu=True, 
            input_size=416)


output = gtf.Predict(system["img_file"], 
                        conf_thres=system["conf_thresh"],
                        iou_thres=system["iou_thresh"]);

print("Completed")
