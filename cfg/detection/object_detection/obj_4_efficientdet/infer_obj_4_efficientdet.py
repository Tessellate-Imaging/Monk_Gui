import os
import sys
import json
import time


sys.path.append("Monk_Object_Detection/4_efficientdet/lib/")


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

with open('obj_4_efficientdet_infer.json') as json_file:
    system = json.load(json_file)


if(system["use_gpu"] == "yes"):
    system["use_gpu"] = True;
else:
    system["use_gpu"] = False;

system["conf_thresh"] = float(system["conf_thresh"])
class_file = system["class_file"];

f = open(class_file, 'r');
class_list = f.readlines();
f.close();
for i in range(len(class_list)):
    class_list[i] = class_list[i][:len(class_list[i])-1]

gtf = Infer();

gtf.Model(model_dir=system["weights_dir"])

scores, labels, boxes = gtf.Predict(system["img_file"], 
                                    class_list, 
                                    vis_threshold=system["conf_thresh"]);


print("Completed")
