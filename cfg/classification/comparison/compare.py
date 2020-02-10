import os
import sys
import json

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

print("Compraring (Wait)....")


with open('base_classification.json') as json_file:
    system = json.load(json_file)


sys.path.append("monk_v1/monk/");


from compare_prototype import compare

ctf = compare(verbose=1);
ctf.Comparison("Sample_Comparison");


for i in range(len(system["compare"]["experiments"])):
    ctf.Add_Experiment(system["compare"]["project"], system["compare"]["experiments"][i]);


ctf.Generate_Statistics();

print("Completed");
