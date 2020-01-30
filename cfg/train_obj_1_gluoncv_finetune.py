import os
import sys

import numpy as np
import pandas as pd

import xmltodict
import json
from tqdm.notebook import tqdm

from pycocotools.coco import COCO

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

print("Training....")


with open('obj_1_gluoncv_finetune.json') as json_file:
    system = json.load(json_file)

system["batch_size"] = int(system["batch_size"]);
if(system["use_pretrained"] == "yes"):
    system["use_pretrained"] = True;
else:
    system["use_pretrained"] = False;
if(system["use_gpu"] == "yes"):
    system["use_gpu"] = True;
else:
    system["use_gpu"] = False;
system["lr"] = float(system["lr"])
system["epochs"] = int(system["epochs"]);




sys.path.append("Monk_Object_Detection/1_gluoncv_finetune/lib/");


from detector_prototype import Detector


gtf = Detector();


if(system["anno_type"] == "voc"):
    root_dir = system["root_dir"];
    img_dir = system["img_dir"];
    anno_dir = system["anno_dir"];

    files = os.listdir(root_dir + "/" + anno_dir);

    combined = [];
    for i in tqdm(range(len(files))):
        annoFile = root_dir + "/" + anno_dir + "/" + files[i];
        f = open(annoFile, 'r');
        my_xml = f.read();
        anno = dict(dict(xmltodict.parse(my_xml))["annotation"])
        fname = anno["filename"];
        label_str = "";
        if(type(anno["object"]) == list):
            for j in range(len(anno["object"])):
                obj = dict(anno["object"][j]);
                label = anno["object"][j]["name"];
                bbox = dict(anno["object"][j]["bndbox"])
                x1 = bbox["xmin"];
                y1 = bbox["ymin"];
                x2 = bbox["xmax"];
                y2 = bbox["ymax"];
                if(j == len(anno["object"])-1):
                    label_str += x1 + " " + y1 + " " + x2 + " " + y2 + " " + label;
                else:        
                    label_str += x1 + " " + y1 + " " + x2 + " " + y2 + " " + label + " ";
        else:
            obj = dict(anno["object"]);
            label = anno["object"]["name"];
            bbox = dict(anno["object"]["bndbox"])
            x1 = bbox["xmin"];
            y1 = bbox["ymin"];
            x2 = bbox["xmax"];
            y2 = bbox["ymax"];
            
            label_str += x1 + " " + y1 + " " + x2 + " " + y2 + " " + label;
        
        
        combined.append([fname, label_str])

    df = pd.DataFrame(combined, columns = ['ID', 'Label']);
    df.to_csv(root_dir + "/train_labels.csv", index=False);


    anno_file = "train_labels.csv";

else:

    root_dir = system["root_dir"];
    img_dir = system["img_dir"];
    anno_file = system["anno_file"];

batch_size = system["batch_size"];

gtf.Dataset(root_dir, img_dir, anno_file, batch_size=batch_size, num_workers=4);



pretrained = system["use_pretrained"];         
gpu = system["use_gpu"];
model_name = system["model"];

tmp = system["devices"].split(",");
gpu_devices = [];
for i in range(len(tmp)):
    gpu_devices.append(int(tmp[i]))



gtf.Model(model_name, use_pretrained=pretrained, use_gpu=gpu, gpu_devices=gpu_devices);


lr = system["lr"]
gtf.Set_Learning_Rate(lr);


epochs = system["epochs"];
params_file = system["output_model_name"] + ".params";


gtf.Train(epochs, params_file);

print("Completed");

