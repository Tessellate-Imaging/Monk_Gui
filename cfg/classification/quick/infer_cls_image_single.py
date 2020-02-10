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

print("Predicting....")


with open('base_classification.json') as json_file:
    system = json.load(json_file)


system["epochs"] = int(system["epochs"]);


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

if(system["test_data"] == "Single Image"):
    img_file = system["img_file"];
    predictions = ptf.Infer(img_name=img_file, return_raw=False);
    predictions["score"] = str(predictions["score"])
    with open('output.json', 'w') as f:
        json.dump(predictions, f)
else:
    img_folder = system["img_folder"];
    predictions = ptf.Infer(img_dir=img_folder, return_raw=False);
    preds = {};
    for i in range(len(predictions)):
        predictions[i]["score"] = str(predictions[i]["score"])
        preds[str(i)] = predictions[i];
    with open('output.json', 'w') as f:
        json.dump(preds, f)


print("Completed");
