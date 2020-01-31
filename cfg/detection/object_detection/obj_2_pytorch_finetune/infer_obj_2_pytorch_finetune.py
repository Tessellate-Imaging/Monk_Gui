import os
import sys
import json
import time

sys.path.append("Monk_Object_Detection/2_pytorch_finetune/lib/");

from inference_prototype import Infer



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

with open('obj_2_pytorch_finetune_infer.json') as json_file:
    system = json.load(json_file)


if(system["use_gpu"] == "yes"):
    system["use_gpu"] = True;
else:
    system["use_gpu"] = False;

system["conf_thresh"] = float(system["conf_thresh"])




model_name = system["model"];
params_file = system["weights"];
class_file = system["class_file"];



f = open(class_file, 'r');
class_list = f.readlines();
f.close();
for i in range(len(class_list)):
    class_list[i] = class_list[i][:len(class_list[i])-1]


gtf = Infer(model_name, params_file, class_list, use_gpu=system["use_gpu"]);

img_name = system["img_file"]; 
visualize = False;
thresh = system["conf_thresh"];


output = gtf.run(img_name, thresh=thresh);

print("Completed")
