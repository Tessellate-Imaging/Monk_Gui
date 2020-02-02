import os
import sys

import numpy as np
import pandas as pd
import cv2

import xmltodict
import json
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO

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



with open('obj_7_yolov3.json') as json_file:
    system = json.load(json_file)


system["batch_size"] = int(system["batch_size"]);
system["img_size"] = int(system["img_size"]);
system["epochs"] = int(system["epochs"]);
system["num_generations"] = int(system["num_generations"]);
system["lr"] = float(system["lr"]);





if(system["use_gpu"] == "yes"):
    system["use_gpu"] = True;
else:
    system["use_gpu"] = False;


if(system["multi_scale"] == "yes"):
    system["multi_scale"] = True;
else:
    system["multi_scale"] = False;


if(system["evolve"] == "yes"):
    system["evolve"] = True;
else:
    system["evolve"] = False;


if(system["mixed_precision"] == "yes"):
    system["mixed_precision"] = True;
else:
    system["mixed_precision"] = False;


if(system["cache_images"] == "yes"):
    system["cache_images"] = True;
else:
    system["cache_images"] = False;




sys.path.append("Monk_Object_Detection/7_yolov3/lib/")


from train_detector import Detector




if(system["anno_type"] == "monk"):
    root_dir = system["monk_root_dir"];
    img_dir = system["monk_img_dir"];
    anno_file = system["monk_anno_file"];


    labels_dir = "labels";
    classes_file = "classes.txt";

    labels_dir_relative = root_dir + "/" + labels_dir
    if(not os.path.isdir(labels_dir_relative)):
        os.mkdir(labels_dir_relative);

    df = pd.read_csv(root_dir + "/" + anno_file);
    
    columns = df.columns
    classes = [];
    for i in range(len(df)):
        img_file = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(" ");
        for j in range(len(tmp)//5):
            label = tmp[j*5 + 4];
            if(label not in classes):
                classes.append(label);
    classes = sorted(classes)

    f = open(root_dir + "/" + classes_file, 'w');
    for i in range(len(classes)):
        f.write(classes[i]);
        f.write("\n");
    f.close();

    for i in tqdm(range(len(df))):
        img_file = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(" ");
        fname = labels_dir_relative + "/" + img_file.split(".")[0] + ".txt";
        img = Image.open(root_dir + "/" + img_dir + "/" + img_file);
        width, height = img.size
    
        f = open(fname, 'w');
        for j in range(len(tmp)//5):
            x1 = float(tmp[j*5 + 0]);
            y1 = float(tmp[j*5 + 1]);
            x2 = float(tmp[j*5 + 2]);
            y2 = float(tmp[j*5 + 3]);
            label = tmp[j*5 + 4];
            
            x_c = str(((x1 + x2)/2)/width);
            y_c = str(((y1 + y2)/2)/height);
            w = str((x2 - x1)/width);
            h = str((y2 - y1)/height);
            index = str(classes.index(label));
            
            f.write(index + " " + x_c + " " + y_c + " " + w + " " + h);
            f.write("\n");
        f.close();

    root_dir = system["monk_root_dir"];
    img_dir = system["monk_img_dir"];
    anno_dir = labels_dir;


elif(system["anno_type"] == "voc"):
    root_dir = system["voc_root_dir"];
    img_dir = system["voc_img_dir"];
    anno_dir = system["voc_anno_dir"];

    files = os.listdir(root_dir + anno_dir);



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

    root_dir = system["voc_root_dir"];
    img_dir = system["voc_img_dir"];
    anno_file = "train_labels.csv";

    labels_dir = "labels";
    classes_file = "classes.txt";

    labels_dir_relative = root_dir + "/" + labels_dir
    if(not os.path.isdir(labels_dir_relative)):
        os.mkdir(labels_dir_relative);

    df = pd.read_csv(root_dir + "/" + anno_file);
    
    columns = df.columns
    classes = [];
    for i in range(len(df)):
        img_file = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(" ");
        for j in range(len(tmp)//5):
            label = tmp[j*5 + 4];
            if(label not in classes):
                classes.append(label);
    classes = sorted(classes)

    f = open(root_dir + "/" + classes_file, 'w');
    for i in range(len(classes)):
        f.write(classes[i]);
        f.write("\n");
    f.close();

    for i in tqdm(range(len(df))):
        img_file = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(" ");
        fname = labels_dir_relative + "/" + img_file.split(".")[0] + ".txt";
        img = Image.open(root_dir + "/" + img_dir + "/" + img_file);
        width, height = img.size
    
        f = open(fname, 'w');
        for j in range(len(tmp)//5):
            x1 = float(tmp[j*5 + 0]);
            y1 = float(tmp[j*5 + 1]);
            x2 = float(tmp[j*5 + 2]);
            y2 = float(tmp[j*5 + 3]);
            label = tmp[j*5 + 4];
            
            x_c = str(((x1 + x2)/2)/width);
            y_c = str(((y1 + y2)/2)/height);
            w = str((x2 - x1)/width);
            h = str((y2 - y1)/height);
            index = str(classes.index(label));
            
            f.write(index + " " + x_c + " " + y_c + " " + w + " " + h);
            f.write("\n");
        f.close();

    root_dir = system["monk_root_dir"];
    img_dir = system["monk_img_dir"];
    anno_dir = labels_dir;


elif(system["anno_type"] == "coco"):
    root_dir = system["coco_root_dir"];
    coco_dir = system["coco_coco_dir"];
    img_dir = system["coco_img_dir"];
    set_dir = system["coco_set_dir"];

    json_path = root_dir + "/" + coco_dir + "/annotations/instances_" + set_dir + ".json"
    json_data = json.load(open(json_path))
    images_info = json_data["images"]
    cls_info = json_data["categories"]
    combined = []

    pbar = tqdm(total=len(images_info));
    for image in images_info:
        pbar.update();
        label_str=""
        for anno in json_data["annotations"]:
            image_id = anno["image_id"]
            cls_id = anno["category_id"]
            filename = None
            cls = None
            for info in images_info:
                if info["id"] == image_id:
                    filename = info["file_name"]
            x1 = int(anno["bbox"][0])
            y1 = int(anno["bbox"][1])
            x2 = int(anno["bbox"][2] + anno["bbox"][0])
            y2 = int(anno["bbox"][3] + anno["bbox"][1])

            for category in cls_info:
                if category["id"] == cls_id:
                    cls = category["name"]

            if image["file_name"] == filename:
                label_str += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + cls + " "

        label_str = label_str.strip()
        combined.append([image["file_name"], label_str])


    df = pd.DataFrame(combined, columns = ['ID', 'Label']);
    df.to_csv(root_dir + "/" + coco_dir + "/train_labels.csv", index=False);


    root_dir = system["coco_root_dir"] + "/" + system["coco_coco_dir"];
    img_dir = system["coco_img_dir"] + "/" + system["coco_set_dir"];
    anno_file = "train_labels.csv";


    labels_dir = "labels";
    classes_file = "classes.txt";

    labels_dir_relative = root_dir + "/" + labels_dir
    if(not os.path.isdir(labels_dir_relative)):
        os.mkdir(labels_dir_relative);

    df = pd.read_csv(root_dir + "/" + anno_file);
    
    columns = df.columns
    classes = [];
    for i in range(len(df)):
        img_file = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(" ");
        for j in range(len(tmp)//5):
            label = tmp[j*5 + 4];
            if(label not in classes):
                classes.append(label);
    classes = sorted(classes)

    f = open(root_dir + "/" + classes_file, 'w');
    for i in range(len(classes)):
        f.write(classes[i]);
        f.write("\n");
    f.close();

    for i in tqdm(range(len(df))):
        img_file = df[columns[0]][i];
        labels = df[columns[1]][i];
        tmp = labels.split(" ");
        fname = labels_dir_relative + "/" + img_file.split(".")[0] + ".txt";
        img = Image.open(root_dir + "/" + img_dir + "/" + img_file);
        width, height = img.size
    
        f = open(fname, 'w');
        for j in range(len(tmp)//5):
            x1 = float(tmp[j*5 + 0]);
            y1 = float(tmp[j*5 + 1]);
            x2 = float(tmp[j*5 + 2]);
            y2 = float(tmp[j*5 + 3]);
            label = tmp[j*5 + 4];
            
            x_c = str(((x1 + x2)/2)/width);
            y_c = str(((y1 + y2)/2)/height);
            w = str((x2 - x1)/width);
            h = str((y2 - y1)/height);
            index = str(classes.index(label));
            
            f.write(index + " " + x_c + " " + y_c + " " + w + " " + h);
            f.write("\n");
        f.close();

    root_dir = system["coco_root_dir"] + "/" + system["coco_coco_dir"];
    img_dir = system["coco_img_dir"] + "/" + system["coco_set_dir"];
    anno_dir = labels_dir;


else:

    root_dir = system["yolo_root_dir"];
    img_dir = system["yolo_img_dir"];
    anno_dir = system["yolo_anno_dir"];
    classes_file = system["yolo_classes_file"]




if(system["val_data"] == "yes"):
    if(system["val_anno_type"] == "monk"):
        val_root_dir = system["val_monk_root_dir"];
        val_img_dir = system["val_monk_img_dir"];
        anno_file = system["val_monk_anno_file"];


        labels_dir = "labels";
        classes_file = "classes.txt";

        labels_dir_relative = val_root_dir + "/" + labels_dir
        if(not os.path.isdir(labels_dir_relative)):
            os.mkdir(labels_dir_relative);

        df = pd.read_csv(val_root_dir + "/" + anno_file);
        
        columns = df.columns
        classes = [];
        for i in range(len(df)):
            img_file = df[columns[0]][i];
            labels = df[columns[1]][i];
            tmp = labels.split(" ");
            for j in range(len(tmp)//5):
                label = tmp[j*5 + 4];
                if(label not in classes):
                    classes.append(label);
        classes = sorted(classes)

        f = open(val_root_dir + "/" + classes_file, 'w');
        for i in range(len(classes)):
            f.write(classes[i]);
            f.write("\n");
        f.close();

        for i in tqdm(range(len(df))):
            img_file = df[columns[0]][i];
            labels = df[columns[1]][i];
            tmp = labels.split(" ");
            fname = labels_dir_relative + "/" + img_file.split(".")[0] + ".txt";
            img = Image.open(val_root_dir + "/" + val_img_dir + "/" + img_file);
            width, height = img.size
        
            f = open(fname, 'w');
            for j in range(len(tmp)//5):
                x1 = float(tmp[j*5 + 0]);
                y1 = float(tmp[j*5 + 1]);
                x2 = float(tmp[j*5 + 2]);
                y2 = float(tmp[j*5 + 3]);
                label = tmp[j*5 + 4];
                
                x_c = str(((x1 + x2)/2)/width);
                y_c = str(((y1 + y2)/2)/height);
                w = str((x2 - x1)/width);
                h = str((y2 - y1)/height);
                index = str(classes.index(label));
                
                f.write(index + " " + x_c + " " + y_c + " " + w + " " + h);
                f.write("\n");
            f.close();

        val_root_dir = system["val_monk_root_dir"];
        val_img_dir = system["val_monk_img_dir"];
        val_anno_dir = labels_dir;

    elif(system["val_anno_type"] == "voc"):
        val_root_dir = system["val_voc_root_dir"];
        val_img_dir = system["val_voc_img_dir"];
        val_anno_dir = system["val_voc_anno_dir"];

        files = os.listdir(val_root_dir + val_anno_dir);



        combined = [];
        for i in tqdm(range(len(files))):
            annoFile = val_root_dir + "/" + val_anno_dir + "/" + files[i];
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
        df.to_csv(val_root_dir + "/train_labels.csv", index=False);

        val_root_dir = system["val_voc_root_dir"];
        val_img_dir = system["val_voc_img_dir"];
        anno_file = "train_labels.csv";

        labels_dir = "labels";
        classes_file = "classes.txt";

        labels_dir_relative = val_root_dir + "/" + labels_dir
        if(not os.path.isdir(labels_dir_relative)):
            os.mkdir(labels_dir_relative);

        df = pd.read_csv(val_root_dir + "/" + anno_file);
        
        columns = df.columns
        classes = [];
        for i in range(len(df)):
            img_file = df[columns[0]][i];
            labels = df[columns[1]][i];
            tmp = labels.split(" ");
            for j in range(len(tmp)//5):
                label = tmp[j*5 + 4];
                if(label not in classes):
                    classes.append(label);
        classes = sorted(classes)

        f = open(val_root_dir + "/" + classes_file, 'w');
        for i in range(len(classes)):
            f.write(classes[i]);
            f.write("\n");
        f.close();

        for i in tqdm(range(len(df))):
            img_file = df[columns[0]][i];
            labels = df[columns[1]][i];
            tmp = labels.split(" ");
            fname = labels_dir_relative + "/" + img_file.split(".")[0] + ".txt";
            img = Image.open(val_root_dir + "/" + val_img_dir + "/" + img_file);
            width, height = img.size
        
            f = open(fname, 'w');
            for j in range(len(tmp)//5):
                x1 = float(tmp[j*5 + 0]);
                y1 = float(tmp[j*5 + 1]);
                x2 = float(tmp[j*5 + 2]);
                y2 = float(tmp[j*5 + 3]);
                label = tmp[j*5 + 4];
                
                x_c = str(((x1 + x2)/2)/width);
                y_c = str(((y1 + y2)/2)/height);
                w = str((x2 - x1)/width);
                h = str((y2 - y1)/height);
                index = str(classes.index(label));
                
                f.write(index + " " + x_c + " " + y_c + " " + w + " " + h);
                f.write("\n");
            f.close();

        val_root_dir = system["val_monk_root_dir"];
        val_img_dir = system["val_monk_img_dir"];
        val_anno_dir = labels_dir;


    elif(system["val_anno_type"] == "coco"):
        val_root_dir = system["val_coco_root_dir"];
        coco_dir = system["val_coco_coco_dir"];
        val_img_dir = system["val_coco_img_dir"];
        set_dir = system["val_coco_set_dir"];

        json_path = val_root_dir + "/" + coco_dir + "/annotations/instances_" + set_dir + ".json"
        json_data = json.load(open(json_path))
        images_info = json_data["images"]
        cls_info = json_data["categories"]
        combined = []

        pbar = tqdm(total=len(images_info));
        for image in images_info:
            pbar.update();
            label_str=""
            for anno in json_data["annotations"]:
                image_id = anno["image_id"]
                cls_id = anno["category_id"]
                filename = None
                cls = None
                for info in images_info:
                    if info["id"] == image_id:
                        filename = info["file_name"]
                x1 = int(anno["bbox"][0])
                y1 = int(anno["bbox"][1])
                x2 = int(anno["bbox"][2] + anno["bbox"][0])
                y2 = int(anno["bbox"][3] + anno["bbox"][1])

                for category in cls_info:
                    if category["id"] == cls_id:
                        cls = category["name"]

                if image["file_name"] == filename:
                    label_str += str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + cls + " "

            label_str = label_str.strip()
            combined.append([image["file_name"], label_str])


        df = pd.DataFrame(combined, columns = ['ID', 'Label']);
        df.to_csv(val_root_dir + "/" + coco_dir + "/train_labels.csv", index=False);


        val_root_dir = system["val_coco_root_dir"] + "/" + system["val_coco_coco_dir"];
        val_img_dir = system["val_coco_img_dir"] + "/" + system["val_coco_set_dir"];
        anno_file = "train_labels.csv";


        labels_dir = "labels";
        classes_file = "classes.txt";

        labels_dir_relative = val_root_dir + "/" + labels_dir
        if(not os.path.isdir(labels_dir_relative)):
            os.mkdir(labels_dir_relative);

        df = pd.read_csv(val_root_dir + "/" + anno_file);
        
        columns = df.columns
        classes = [];
        for i in range(len(df)):
            img_file = df[columns[0]][i];
            labels = df[columns[1]][i];
            tmp = labels.split(" ");
            for j in range(len(tmp)//5):
                label = tmp[j*5 + 4];
                if(label not in classes):
                    classes.append(label);
        classes = sorted(classes)

        f = open(val_root_dir + "/" + classes_file, 'w');
        for i in range(len(classes)):
            f.write(classes[i]);
            f.write("\n");
        f.close();

        for i in tqdm(range(len(df))):
            img_file = df[columns[0]][i];
            labels = df[columns[1]][i];
            tmp = labels.split(" ");
            fname = labels_dir_relative + "/" + img_file.split(".")[0] + ".txt";
            img = Image.open(val_root_dir + "/" + val_img_dir + "/" + img_file);
            width, height = img.size
        
            f = open(fname, 'w');
            for j in range(len(tmp)//5):
                x1 = float(tmp[j*5 + 0]);
                y1 = float(tmp[j*5 + 1]);
                x2 = float(tmp[j*5 + 2]);
                y2 = float(tmp[j*5 + 3]);
                label = tmp[j*5 + 4];
                
                x_c = str(((x1 + x2)/2)/width);
                y_c = str(((y1 + y2)/2)/height);
                w = str((x2 - x1)/width);
                h = str((y2 - y1)/height);
                index = str(classes.index(label));
                
                f.write(index + " " + x_c + " " + y_c + " " + w + " " + h);
                f.write("\n");
            f.close();

        val_root_dir = system["val_coco_root_dir"] + "/" + system["val_coco_coco_dir"];
        val_img_dir = system["val_coco_img_dir"] + "/" + system["val_coco_set_dir"];
        val_anno_dir = labels_dir;


    else:

        val_root_dir = system["val_yolo_root_dir"];
        val_img_dir = system["val_yolo_img_dir"];
        val_anno_dir = system["val_yolo_anno_dir"];
        val_classes_file = system["val_yolo_classes_file"]



from train_detector import Detector

gtf = Detector();



gtf.set_train_dataset(root_dir + "/" + img_dir, root_dir + "/" + anno_dir, root_dir + "/" + classes_file, 
                        batch_size=system["batch_size"],
                        img_size=system["img_size"], 
                        cache_images=system["cache_images"])

if(system["val_data"] == "yes"):
    gtf.set_val_dataset(val_root_dir + "/" + val_img_dir, val_root_dir + "/" + val_anno_dir)


gtf.set_model(model_name=system["model"]);


gtf.set_hyperparams(optimizer=system["optimizer"], 
                    lr=system["lr"], 
                    multi_scale=system["multi_scale"], 
                    evolve=system["evolve"], 
                    num_generations=system["num_generations"], 
                    mixed_precision=system["mixed_precision"], 
                    gpu_devices=system["devices"]);

gtf.Train(num_epochs=system["epochs"]);


print("Completed");

