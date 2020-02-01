import os
import sys

import numpy as np
import pandas as pd
import cv2

import xmltodict
import json

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
if(isnotebook()):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm as tqdm

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


with open('obj_3_mxrcnn.json') as json_file:
    system = json.load(json_file)

system["batch_size"] = int(system["batch_size"]);
system["img_short_side"] = int(system["img_short_side"]);
system["img_long_side"] = int(system["img_long_side"]);
system["epochs"] = int(system["epochs"]);
system["log_interval"] = int(system["log_interval"]);
system["lr"] = float(system["lr"]);






tmp = system["mean"].split(",")
system["mean"] = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
tmp = system["std"].split(",")
system["std"] = (float(tmp[0]), float(tmp[1]), float(tmp[2]))


if(system["use_pretrained"] == "yes"):
    system["use_pretrained"] = True;
else:
    system["use_pretrained"] = False;
if(system["use_gpu"] == "yes"):
    system["use_gpu"] = True;
else:
    system["use_gpu"] = False;





sys.path.append("Monk_Object_Detection/3_mxrcnn/lib/")
sys.path.append("Monk_Object_Detection/3_mxrcnn/lib/mx-rcnn")


from train_base import *




if(system["anno_type"] == "monk"):
    root_dir = system["monk_root_dir"];
    img_dir = system["monk_img_dir"];
    anno_file = system["monk_anno_file"];

    dataset_path = root_dir; 
    images_folder = root_dir + "/" + img_dir;
    annotations_path = root_dir + "/annotations/";

    if not os.path.isdir(annotations_path):
        os.mkdir(annotations_path)
        
    input_images_folder = images_folder;
    input_annotations_path = root_dir + "/" + anno_file;

    output_dataset_path = root_dir; 
    output_image_folder = input_images_folder;
    output_annotation_folder = annotations_path;

    tmp = img_dir.replace("/", "");
    output_annotation_file = output_annotation_folder + "/instances_" + tmp + ".json";
    output_classes_file = output_annotation_folder + "/classes.txt";


    if not os.path.isdir(output_annotation_folder):
        os.mkdir(output_annotation_folder);

    df = pd.read_csv(input_annotations_path);
    columns = df.columns

    delimiter = " ";

    list_dict = [];
    anno = [];
    for i in range(len(df)):
        img_name = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(delimiter);
        for j in range(len(tmp)//5):
            label = tmp[j*5+4];
            if(label not in anno):
                anno.append(label);
        anno = sorted(anno)
        
    for i in tqdm(range(len(anno))):
        tmp = {};
        tmp["supercategory"] = "master";
        tmp["id"] = i;
        tmp["name"] = anno[i];
        list_dict.append(tmp);

    anno_f = open(output_classes_file, 'w');
    for i in range(len(anno)):
        anno_f.write(anno[i] + "\n");
    anno_f.close();


    coco_data = {};
    coco_data["type"] = "instances";
    coco_data["images"] = [];
    coco_data["annotations"] = [];
    coco_data["categories"] = list_dict;
    image_id = 0;
    annotation_id = 0;


    for i in tqdm(range(len(df))):
        img_name = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(delimiter);
        image_in_path = input_images_folder + "/" + img_name;
        img = cv2.imread(image_in_path, 1);
        h, w, c = img.shape;

        images_tmp = {};
        images_tmp["file_name"] = img_name;
        images_tmp["height"] = h;
        images_tmp["width"] = w;
        images_tmp["id"] = image_id;
        coco_data["images"].append(images_tmp);
        

        for j in range(len(tmp)//5):
            x1 = int(tmp[j*5+0]);
            y1 = int(tmp[j*5+1]);
            x2 = int(tmp[j*5+2]);
            y2 = int(tmp[j*5+3]);
            label = tmp[j*5+4];
            annotations_tmp = {};
            annotations_tmp["id"] = annotation_id;
            annotation_id += 1;
            annotations_tmp["image_id"] = image_id;
            annotations_tmp["segmentation"] = [];
            annotations_tmp["ignore"] = 0;
            annotations_tmp["area"] = (x2-x1)*(y2-y1);
            annotations_tmp["iscrowd"] = 0;
            annotations_tmp["bbox"] = [x1, y1, x2-x1, y2-y1];
            annotations_tmp["category_id"] = anno.index(label);

            coco_data["annotations"].append(annotations_tmp)
        image_id += 1;

    outfile =  open(output_annotation_file, 'w');
    json_str = json.dumps(coco_data, indent=4);
    outfile.write(json_str);
    outfile.close();

    root_dir = system["monk_root_dir"];
    coco_dir = "";
    img_dir = system["monk_img_dir"];


elif(system["anno_type"] == "voc"):
    root_dir = system["voc_root_dir"];
    img_dir = system["voc_img_dir"];
    anno_dir = system["voc_anno_dir"];

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

    dataset_path = root_dir; 
    images_folder = root_dir + "/" + img_dir;
    annotations_path = root_dir + "/annotations/";

    if not os.path.isdir(annotations_path):
        os.mkdir(annotations_path)
        
    input_images_folder = images_folder;
    input_annotations_path = root_dir + "/" + anno_file;

    output_dataset_path = root_dir; 
    output_image_folder = input_images_folder;
    output_annotation_folder = annotations_path;

    tmp = img_dir.replace("/", "");
    output_annotation_file = output_annotation_folder + "/instances_" + tmp + ".json";
    output_classes_file = output_annotation_folder + "/classes.txt";


    if not os.path.isdir(output_annotation_folder):
        os.mkdir(output_annotation_folder);

    df = pd.read_csv(input_annotations_path);
    columns = df.columns

    delimiter = " ";

    list_dict = [];
    anno = [];
    for i in range(len(df)):
        img_name = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(delimiter);
        for j in range(len(tmp)//5):
            label = tmp[j*5+4];
            if(label not in anno):
                anno.append(label);
        anno = sorted(anno)
        
    for i in tqdm(range(len(anno))):
        tmp = {};
        tmp["supercategory"] = "master";
        tmp["id"] = i;
        tmp["name"] = anno[i];
        list_dict.append(tmp);

    anno_f = open(output_classes_file, 'w');
    for i in range(len(anno)):
        anno_f.write(anno[i] + "\n");
    anno_f.close();


    coco_data = {};
    coco_data["type"] = "instances";
    coco_data["images"] = [];
    coco_data["annotations"] = [];
    coco_data["categories"] = list_dict;
    image_id = 0;
    annotation_id = 0;


    for i in tqdm(range(len(df))):
        img_name = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(delimiter);
        image_in_path = input_images_folder + "/" + img_name;
        img = cv2.imread(image_in_path, 1);
        h, w, c = img.shape;

        images_tmp = {};
        images_tmp["file_name"] = img_name;
        images_tmp["height"] = h;
        images_tmp["width"] = w;
        images_tmp["id"] = image_id;
        coco_data["images"].append(images_tmp);
        

        for j in range(len(tmp)//5):
            x1 = int(tmp[j*5+0]);
            y1 = int(tmp[j*5+1]);
            x2 = int(tmp[j*5+2]);
            y2 = int(tmp[j*5+3]);
            label = tmp[j*5+4];
            annotations_tmp = {};
            annotations_tmp["id"] = annotation_id;
            annotation_id += 1;
            annotations_tmp["image_id"] = image_id;
            annotations_tmp["segmentation"] = [];
            annotations_tmp["ignore"] = 0;
            annotations_tmp["area"] = (x2-x1)*(y2-y1);
            annotations_tmp["iscrowd"] = 0;
            annotations_tmp["bbox"] = [x1, y1, x2-x1, y2-y1];
            annotations_tmp["category_id"] = anno.index(label);

            coco_data["annotations"].append(annotations_tmp)
        image_id += 1;

    outfile =  open(output_annotation_file, 'w');
    json_str = json.dumps(coco_data, indent=4);
    outfile.write(json_str);
    outfile.close();

    root_dir = system["voc_root_dir"];
    coco_dir = "";
    img_dir = system["voc_img_dir"];



else:

    root_dir = system["coco_root_dir"];
    coco_dir = system["coco_coco_dir"];
    img_dir = system["coco_img_dir"];





set_dataset_params(root_dir=root_dir, 
                   coco_dir=coco_dir, imageset=img_dir);



model_name = system["model"];
set_model_params(model_name=model_name);


tmp = system["devices"].split(",");
gpu_devices = [];
for i in range(len(tmp)):
    gpu_devices.append(int(tmp[i]))
lr = system["lr"];
lr_decay_epoch = system["lr_decay_epoch"]
epochs = system["epochs"]
log_interval = system["log_interval"]
batch_size = system["batch_size"];
output_model_name = system["output_model_name"];

set_hyper_params(gpus=gpu_devices, lr=lr, lr_decay_epoch=lr_decay_epoch, 
                    epochs=epochs, batch_size=batch_size);

set_output_params(log_interval=log_interval, save_prefix=output_model_name);



img_short_side = system["img_short_side"];
img_long_side = system["img_long_side"];
mean = system["mean"];
std = system["std"];

set_img_preproc_params(img_short_side=img_short_side, img_long_side=img_long_side, 
                       mean=mean, std=std);


initialize_rpn_params();
initialize_rcnn_params();




if os.path.isdir(root_dir + "/cache/"):
    os.system("rm -r " + root_dir + "/cache/")
roidb = set_dataset();


sym = set_network();


train(sym, roidb);

print("Completed");

