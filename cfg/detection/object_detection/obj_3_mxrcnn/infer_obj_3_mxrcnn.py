import os
import sys
import json
import time


sys.path.append("Monk_Object_Detection/3_mxrcnn/lib/")
sys.path.append("Monk_Object_Detection/3_mxrcnn/lib/mx-rcnn")


from infer_base import *



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

with open('obj_3_mxrcnn_infer.json') as json_file:
    system = json.load(json_file)


if(system["use_gpu"] == "yes"):
    system["use_gpu"] = True;
else:
    system["use_gpu"] = False;

system["conf_thresh"] = float(system["conf_thresh"])
system["img_short_side"] = int(system["img_short_side"]);
system["img_long_side"] = int(system["img_long_side"]);

tmp = system["mean"].split(",")
system["mean"] = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
tmp = system["std"].split(",")
system["std"] = (float(tmp[0]), float(tmp[1]), float(tmp[2]))




model_name = system["model"];
params_file = system["weights"];
class_file = system["class_file"];



class_file = set_class_list(class_file);
set_model_params(model_name=model_name, model_path=params_file);
set_hyper_params(gpus="0", batch_size=1);
set_img_preproc_params(img_short_side=system["img_short_side"], img_long_side=system["img_long_side"], 
                       mean=system["mean"], std=system["std"]);
initialize_rpn_params();
initialize_rcnn_params();

sym = set_network();
mod = load_model(sym);

set_output_params(vis_thresh=system["conf_thresh"], vis=False)
out = Infer(system["img_file"], mod);




print("Completed")
