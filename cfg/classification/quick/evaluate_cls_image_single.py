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

print("Evaluating....")


with open('base_classification.json') as json_file:
    system = json.load(json_file)


sys.path.append("monk_v1/monk/");

with open('workspace/' + system["project"] + '/' + system["experiment"] + '/experiment_state.json') as json_file:
    data = json.load(json_file)


if(data["library"] == "Mxnet"):
    from gluon_prototype import prototype

elif(data["library"] == "Pytorch"):
    from pytorch_prototype import prototype        

elif(data["library"] == "Keras"):
    from keras_prototype import prototype





ptf = prototype(verbose=1);
ptf.Prototype(system["project"], system["experiment"], eval_infer=True);



if(system["structuretype"] == "foldered"):
    ptf.Dataset_Params(dataset_path=system["evaluate"]["dir"]);
else:
    ptf.Dataset_Params(dataset_path=system["evaluate"]["cdir"], path_to_csv=[system["evaluate"]["csv"]]);

ptf.Dataset();
accuracy, class_based_accuracy = ptf.Evaluate();

wr = "";
wr += "Total Accuracy - {}\n\n".format(accuracy);
wr += "Class based accuracies\n";
keys = list(class_based_accuracy.keys());
for i in range(len(keys)):
    wr += "Class - {}\n".format(keys[i]);
    wr += "    Num Images - {}\n".format(class_based_accuracy[keys[i]]["num_images"])
    wr += "    Num Correct - {}\n".format(class_based_accuracy[keys[i]]["num_correct"])
    wr += "    Accuracy(%) - {}\n\n".format(class_based_accuracy[keys[i]]["accuracy(%)"])



f = open("results.txt", 'w');
f.write(wr);
f.close();

print("Completed");
