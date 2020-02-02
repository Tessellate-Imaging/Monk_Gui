import os
import sys
import json
import time


sys.path.append("Monk_Object_Detection/6_cornernet_lite/lib/")


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

with open('obj_6_cornernet_lite_infer.json') as json_file:
    system = json.load(json_file)



system["conf_thresh"] = float(system["conf_thresh"])
class_file = system["class_file"];

f = open(class_file, 'r');
class_list = f.readlines();
f.close();
for i in range(len(class_list)):
    class_list[i] = class_list[i][:len(class_list[i])-1]

gtf = Infer();


gtf.Model(class_list, 
          base=system["model"], 
          model_path=system["weights"])


output = gtf.Predict(system["img_file"], 
                        vis_thresh=system["conf_thresh"],
                        output_img="output.jpg");



print("Completed")
