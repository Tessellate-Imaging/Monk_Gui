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

print("Analysing....")


with open('base_classification.json') as json_file:
    system = json.load(json_file)


system["epochs"] = int(system["epochs"]);

if(system["freeze_base_model"] == "yes"):
    system["freeze_base_model"] = True;
else:
    system["freeze_base_model"] = False;

if(system["val"] == "yes"):
    system["val"] = True;
else:
    system["val"] = False;



sys.path.append("monk_v1/monk/");

if(system["backend"] == "Mxnet-1.5.1"):
    from gluon_prototype import prototype

elif(system["backend"] == "Pytorch-1.3.1"):
    from pytorch_prototype import prototype        

elif(system["backend"] == "Keras-2.2.5_Tensorflow-1"):
    from keras_prototype import prototype




ptf = prototype(verbose=1);
ptf.Prototype(system["project"], system["experiment"]);



if(system["structuretype"] == "foldered"):
    if(system["val"]):
        ptf.Default(dataset_path=[system["traindata"]["dir"], system["valdata"]["dir"]], 
                    model_name=system["model"], 
                    freeze_base_network=system["freeze_base_model"], 
                    num_epochs=system["epochs"]);
    else:
        ptf.Default(dataset_path=system["traindata"]["dir"], 
                    model_name=system["model"], 
                    freeze_base_network=system["freeze_base_model"], 
                    num_epochs=system["epochs"]);
else:
    if(system["val"]):
        ptf.Default(dataset_path=[system["traindata"]["cdir"], system["valdata"]["cdir"]], 
                    path_to_csv=[system["traindata"]["csv"], system["valdata"]["csv"]],
                    model_name=system["model"], 
                    freeze_base_network=system["freeze_base_model"], 
                    num_epochs=system["epochs"]);
    else:
        ptf.Default(dataset_path=system["traindata"]["cdir"], 
                    path_to_csv=system["traindata"]["csv"],
                    model_name=system["model"], 
                    freeze_base_network=system["freeze_base_model"], 
                    num_epochs=system["epochs"]);


dataset_reload_status = False;

if(system["update"]["input_size"]["active"]):
    dataset_reload_status = True;
    ptf.update_input_size(int(system["update"]["input_size"]["value"]));


if(system["update"]["batch_size"]["active"]):
    dataset_reload_status = True;
    ptf.update_batch_size(int(system["update"]["batch_size"]["value"]));


if(system["update"]["shuffle_data"]["active"]):
    dataset_reload_status = True;
    if(system["update"]["shuffle_data"]["value"] == "True"):
        ptf.update_shuffle_data(True);    
    else:
        ptf.update_shuffle_data(False);


if(system["update"]["num_processors"]["active"]):
    dataset_reload_status = True;
    ptf.update_num_processors(int(system["update"]["num_processors"]["value"]));



if(system["update"]["trainval_split"]["active"]):
    dataset_reload_status = True;
    ptf.update_trainval_split(float(system["update"]["trainval_split"]["value"]));



if(system["update"]["transforms"]["active"]):
    dataset_reload_status = True;
    ptf.reset_transforms();
    ptf.reset_transforms(test=True);

    if(system["backend"] == "Mxnet-1.5.1"):
        for i in range(len(system["update"]["transforms"]["value"])):
            name = system["update"]["transforms"]["value"][i]["name"];
            params = system["update"]["transforms"]["value"][i]["params"];
            if(params["train"] == "True"):
                train = True;
            else:
                train = False;
            if(params["val"] == "True"):
                val = True;
            else:
                val = False;
            if(params["test"] == "True"):
                test = True;
            else:
                test = False;


            if(name == "apply_random_resized_crop"):
                ptf.apply_random_resized_crop(
                    int(params["input_size"]), 
                    scale=(float(params["scale"][0]), float(params["scale"][1])), 
                    ratio=(float(params["ratio"][0]), float(params["ratio"][0])), 
                    train=train, val=val, test=test);
            
            elif(name == "apply_center_crop"):
                ptf.apply_center_crop(
                    int(params["input_size"]),  
                    train=train, val=val, test=test);

            elif(name == "apply_color_jitter"):
                ptf.apply_color_jitter(
                    brightness=float(params["brightness"]), 
                    contrast=float(params["contrast"]), 
                    saturation=float(params["saturation"]), 
                    hue=float(params["hue"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_horizontal_flip"):
                ptf.apply_random_horizontal_flip(
                    probability=float(params["probability"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_vertical_flip"):
                ptf.apply_random_vertical_flip(
                    probability=float(params["probability"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_lighting"):
                ptf.apply_random_lighting(
                    alpha=float(params["alpha"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_resize"):
                ptf.apply_resize(
                    int(params["input_size"]),  
                    train=train, val=val, test=test);

            elif(name == "apply_normalize"):
                ptf.apply_normalize(
                    mean=[float(params["mean"][0]),float(params["mean"][1]), float(params["mean"][2])], 
                    std=[float(params["std"][0]),float(params["std"][1]), float(params["std"][2])], 
                    train=train, val=val, test=test);



    elif(system["backend"] == "Pytorch-1.3.1"):
        for i in range(len(system["update"]["transforms"]["value"])):
            name = system["update"]["transforms"]["value"][i]["name"];
            params = system["update"]["transforms"]["value"][i]["params"];
            if(params["train"] == "True"):
                train = True;
            else:
                train = False;
            if(params["val"] == "True"):
                val = True;
            else:
                val = False;
            if(params["test"] == "True"):
                test = True;
            else:
                test = False;

            if(name == "apply_center_crop"):
                ptf.apply_center_crop(
                    int(params["input_size"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_color_jitter"):
                ptf.apply_color_jitter(
                    brightness=float(params["brightness"]), 
                    contrast=float(params["contrast"]), 
                    saturation=float(params["saturation"]), 
                    hue=float(params["hue"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_affine"):
                ptf.apply_random_affine(
                    float(params["degrees"]), 
                    translate=(float(params["translate"][0]), float(params["translate"][1])), 
                    scale=(float(params["scale"][0]), float(params["scale"][1])), 
                    shear=(float(params["shear"][0]), float(params["shear"][1])), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_crop"):
                ptf.apply_random_crop(
                    int(params["input_size"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_horizontal_flip"):
                ptf.apply_random_horizontal_flip(
                    probability=float(params["probability"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_perspective"):
                ptf.apply_random_perspective(
                    distortion_scale=float(params["distortion_scale"]), 
                    probability=float(params["probability"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_resized_crop"):
                ptf.apply_random_resized_crop(
                    int(params["input_size"]), 
                    scale=(float(params["scale"][0]), float(params["scale"][1])), 
                    ratio=(float(params["ratio"][0]), float(params["ratio"][0])), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_rotation"):
                ptf.apply_random_rotation(
                    float(params["degrees"]), 
                        train=train, val=val, test=test);

            elif(name == "apply_random_vertical_flip"):
                ptf.apply_random_vertical_flip(
                    probability=float(params["probability"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_resize"):
                ptf.apply_resize(
                    int(params["input_size"]),  
                    train=train, val=val, test=test);

            elif(name == "apply_normalize"):
                ptf.apply_normalize(
                    mean=[float(params["mean"][0]),float(params["mean"][1]), float(params["mean"][2])], 
                    std=[float(params["std"][0]),float(params["std"][1]), float(params["std"][2])], 
                    train=train, val=val, test=test);



    elif(system["backend"] == "Keras-2.2.5_Tensorflow-1"):
        for i in range(len(system["update"]["transforms"]["value"])):
            name = system["update"]["transforms"]["value"][i]["name"];
            params = system["update"]["transforms"]["value"][i]["params"];
            if(params["train"] == "True"):
                train = True;
            else:
                train = False;
            if(params["val"] == "True"):
                val = True;
            else:
                val = False;
            if(params["test"] == "True"):
                test = True;
            else:
                test = False;

            if(name == "apply_color_jitter"):
                ptf.apply_color_jitter(
                    brightness=float(params["brightness"]), 
                    contrast=float(params["contrast"]), 
                    saturation=float(params["saturation"]), 
                    hue=float(params["hue"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_affine"):
                ptf.apply_random_affine(
                    float(params["degrees"]), 
                    translate=(float(params["translate"][0]), float(params["translate"][1])), 
                    scale=(float(params["scale"][0]), float(params["scale"][1])), 
                    shear=(float(params["shear"][0]), float(params["shear"][1])), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_horizontal_flip"):
                ptf.apply_random_horizontal_flip(
                    probability=float(params["probability"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_vertical_flip"):
                ptf.apply_random_vertical_flip(
                    probability=float(params["probability"]), 
                    train=train, val=val, test=test);

            elif(name == "apply_random_rotation"):
                ptf.apply_random_rotation(
                    float(params["degrees"]), 
                        train=train, val=val, test=test);

            elif(name == "apply_mean_subtraction"):
                ptf.apply_mean_subtraction(
                    mean=[float(params["mean"][0]),float(params["mean"][1]), float(params["mean"][2])],
                    train=train, val=val, test=test);

            elif(name == "apply_normalize"):
                ptf.apply_normalize(
                    mean=[float(params["mean"][0]),float(params["mean"][1]), float(params["mean"][2])], 
                    std=[float(params["std"][0]),float(params["std"][1]), float(params["std"][2])], 
                    train=train, val=val, test=test);



if(dataset_reload_status):
    ptf.Reload();







model_reload_status = False;


if(system["update"]["model_name"]["active"]):
    model_reload_status = True;
    ptf.update_model_name(system["update"]["model_name"]["value"]);


if(system["update"]["use_gpu"]["active"]):
    model_reload_status = True;
    if(system["update"]["use_gpu"]["value"] == "True"):
        ptf.update_use_gpu(True);
    else:
        ptf.update_use_gpu(False);


if(system["update"]["use_pretrained"]["active"]):
    model_reload_status = True;
    if(system["update"]["use_pretrained"]["value"] == "True"):
        ptf.update_use_pretrained(True);
    else:
        ptf.update_use_pretrained(False);


if(system["update"]["freeze_base_network"]["active"]):
    model_reload_status = True;
    if(system["update"]["freeze_base_network"]["value"] == "True"):
        ptf.update_freeze_base_network(True);
    else:
        ptf.update_freeze_base_network(False);


if(system["update"]["freeze_layers"]["active"]):
    model_reload_status = True;
    ptf.update_freeze_layers(int(system["update"]["freeze_layers"]["value"]));




if(system["update"]["layers"]["active"]):
    model_reload_status = True;

    if(system["backend"] == "Mxnet-1.5.1"):
        for i in range(len(system["update"]["layers"]["value"])):
            name = system["update"]["layers"]["value"][i]["name"];
            params = system["update"]["layers"]["value"][i]["params"];

            if(params["final"] == "Yes"):
                final_layer = True;
            else:
                final_layer = False;


            if(name == "append_linear"):
                if(final_layer):
                    ptf.append_linear( 
                        final_layer=final_layer);
                else:
                    ptf.append_linear(
                        num_neurons=int(params["neurons"]), 
                        final_layer=final_layer);

            elif(name == "append_dropout"):
                ptf.append_dropout(
                    probability=float(params["probability"]), 
                    final_layer=final_layer);

            elif(name == "relu"):
                ptf.append_relu(
                    final_layer=final_layer);

            elif(name == "sigmoid"):
                ptf.append_sigmoid(
                    final_layer=final_layer);

            elif(name == "tanh"):
                ptf.append_tanh(
                    final_layer=final_layer);

            elif(name == "softpllus"):
                ptf.append_softplus(
                    beta=float(params["beta"]), 
                    threshold=float(params["threshold"]), 
                    final_layer=final_layer);

            elif(name == "softsign"):
                ptf.append_softsign(
                    final_layer=final_layer);

            elif(name == "elu"):
                ptf.append_elu(
                    alpha=float(params["alpha"]), 
                    final_layer=final_layer);

            elif(name == "leaky_relu"):
                ptf.append_leakyrelu(
                    negative_slope=float(params["negative_slope"]), 
                    final_layer=final_layer);

            elif(name == "prelu"):
                ptf.append_prelu(
                    num_parameters=1, 
                    init=float(params["init"]), 
                    final_layer=final_layer);

            elif(name == "selu"):
                ptf.append_selu(
                    final_layer=final_layer);

            elif(name == "swish"):
                ptf.append_swish(
                    beta=float(params["beta"]), 
                    final_layer=final_layer);



    elif(system["backend"] == "Pytorch-1.3.1"):
        for i in range(len(system["update"]["layers"]["value"])):
            name = system["update"]["layers"]["value"][i]["name"];
            params = system["update"]["layers"]["value"][i]["params"];

            if(params["final"] == "Yes"):
                final_layer = True;
            else:
                final_layer = False;


            if(name == "append_linear"):
                if(final_layer):
                    ptf.append_linear( 
                        final_layer=final_layer);
                else:
                    ptf.append_linear(
                        num_neurons=int(params["neurons"]), 
                        final_layer=final_layer);

            elif(name == "append_dropout"):
                ptf.append_dropout(
                    probability=float(params["probability"]), 
                    final_layer=final_layer);

            elif(name == "relu"):
                ptf.append_relu(
                    final_layer=final_layer);

            elif(name == "sigmoid"):
                ptf.append_sigmoid(
                    final_layer=final_layer);

            elif(name == "tanh"):
                ptf.append_tanh(
                    final_layer=final_layer);

            elif(name == "softplus"):
                ptf.append_softplus(
                    beta=float(params["beta"]), 
                    threshold=float(params["threshold"]), 
                    final_layer=final_layer);

            elif(name == "softsign"):
                ptf.append_softsign(
                    final_layer=final_layer);

            elif(name == "elu"):
                ptf.append_elu(
                    alpha=float(params[alpha]),
                    final_layer=final_layer);

            elif(name == "leaky_relu"):
                ptf.append_leakyrelu(
                    negative_slope=float(params["negative_slope"]), 
                    final_layer=final_layer);

            elif(name == "prelu"):
                ptf.append_prelu(
                    num_parameters=1, 
                    init=float(params["init"]), 
                    final_layer=final_layer);

            elif(name == "selu"):
                ptf.append_selu(
                    final_layer=final_layer);

            elif(name == "hardshrink"):
                ptf.append_hardshrink(
                    lambd=float(params["lambd"]), 
                    final_layer=final_layer);

            elif(name == "hardtanh"):
                ptf.append_hardtanh(
                    min_val=float(params["min_val"]), 
                    max_val=float(params["max_val"]), 
                    final_layer=final_layer);

            elif(name == "logsigmoid"):
                ptf.append_logsigmoid(
                    final_layer=final_layer);

            elif(name == "relu6"):
                ptf.append_relu6(
                    final_layer=final_layer);

            elif(name == "rrelu"):
                ptf.append_rrelu(
                    lower=float(params["lower"]), 
                    upper=float(params["upper"]), 
                    final_layer=final_layer);

            elif(name == "celu"):
                ptf.append_celu(
                    alpha=float(params["alpha"]), 
                    final_layer=final_layer);

            elif(name == "softshrink"):
                ptf.append_softshrink(
                    lambd=float(params["lambd"]), 
                    final_layer=final_layer);

            elif(name == "tanhshrink"):
                ptf.append_tanhshrink(
                    final_layer=final_layer);

            elif(name == "threshold"):
                ptf.append_threshold(
                    float(params["threshold"]), 
                    float(params["value"]),
                    final_layer=final_layer)

            elif(name == "softmin"):
                ptf.append_softmin(
                    final_layer=final_layer);

            elif(name == "softmax"):
                ptf.append_softmax(
                    final_layer=final_layer);

            elif(name == "logsoftmax"):
                ptf.append_logsoftmax(
                    final_layer=final_layer);


    elif(system["backend"] == "Keras-2.2.5_Tensorflow-1"):
        for i in range(len(system["update"]["layers"]["value"])):
            name = system["update"]["layers"]["value"][i]["name"];
            params = system["update"]["layers"]["value"][i]["params"];

            if(params["final"] == "Yes"):
                final_layer = True;
            else:
                final_layer = False;


            if(name == "append_linear"):
                if(final_layer):
                    ptf.append_linear( 
                        final_layer=final_layer);
                else:
                    ptf.append_linear(
                        num_neurons=int(params["neurons"]), 
                        final_layer=final_layer);

            elif(name == "append_dropout"):
                ptf.append_dropout(
                    probability=float(params["probability"]), 
                    final_layer=final_layer);

            elif(name == "relu"):
                ptf.append_relu(
                    final_layer=final_layer);

            elif(name == "elu"):
                ptf.append_elu(
                    alpha=float(params[alpha]),
                    final_layer=final_layer);

            elif(name == "leaky_relu"):
                ptf.append_leakyrelu(
                    negative_slope=float(params["negative_slope"]), 
                    final_layer=final_layer);

            elif(name == "prelu"):
                ptf.append_prelu(
                    num_parameters=1, 
                    init=float(params["init"]), 
                    final_layer=final_layer);

            elif(name == "threshold"):
                ptf.append_threshold(
                    float(params["threshold"]), 
                    float(params["value"]),
                    final_layer=final_layer);

            elif(name == "softmax"):
                ptf.append_softmax(
                    final_layer=final_layer);

            elif(name == "selu"):
                ptf.append_selu(
                    final_layer=final_layer);

            elif(name == "softplus"):
                ptf.append_softplus(
                    beta=float(params["beta"]), 
                    threshold=float(params["threshold"]), 
                    final_layer=final_layer);

            elif(name == "softsign"):
                ptf.append_softsign(
                    final_layer=final_layer);

            elif(name == "tanh"):
                ptf.append_tanh(
                    final_layer=final_layer);

            elif(name == "sigmoid"):
                ptf.append_sigmoid(
                    final_layer=final_layer);



if(model_reload_status):
    ptf.Reload();






train_reload_status = False;

if(system["update"]["realtime_progress"]["active"]):
    train_reload_status = True;
    if(system["update"]["realtime_progress"]["value"] == "True"):
        ptf.update_display_progress_realtime(True);
    else:
        ptf.update_display_progress_realtime(False);


if(system["update"]["progress"]["active"]):
    train_reload_status = True;
    if(system["update"]["progress"]["value"] == "True"):
        ptf.update_display_progress(True);
    else:
        ptf.update_display_progress(False);


if(system["update"]["save_intermediate"]["active"]):
    train_reload_status = True;
    if(system["update"]["save_intermediate"]["value"] == "True"):
        ptf.update_save_intermediate_models(True);
    else:
        ptf.update_save_intermediate_models(False);


if(system["update"]["save_logs"]["active"]):
    train_reload_status = True;
    if(system["update"]["save_logs"]["value"] == "True"):
        ptf.update_save_training_logs(True);
    else:
        ptf.update_save_training_logs(False);


if(train_reload_status):
    ptf.Reload();





optimizer_reload_status = False;
if(system["update"]["optimizers"]["active"]):
    optimizer_reload_status = True;

    if(system["backend"] == "Mxnet-1.5.1"):
        
        name = system["update"]["optimizers"]["value"]["name"];
        params = system["update"]["optimizers"]["value"]["params"];

        if(name == "optimizer_sgd"):
            ptf.optimizer_sgd(
                float(params["learning_rate"]), 
                momentum=float(params["momentum"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_nesterov_sgd"):
            ptf.optimizer_nesterov_sgd(
                float(params["learning_rate"]), 
                momentum=float(params["momentum"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_rmsprop"):
            ptf.optimizer_rmsprop(
                float(params["learning_rate"]), 
                decay_rate=float(params["decay_rate"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]));     

        elif(name == "optimizer_momentum_rmsprop"):
            ptf.optimizer_momentum_rmsprop(
                float(params["learning_rate"]), 
                momentum=float(params["momentum"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_adam"):
            ptf.optimizer_adam(
                float(params["learning_rate"]), 
                beta1=float(params["beta1"]), 
                beta2=float(params["beta2"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_adamax"):
            ptf.optimizer_adamax(
                float(params["learning_rate"]), 
                beta1=float(params["beta1"]), 
                beta2=float(params["beta2"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]))

        elif(name == "optimizer_nesterov_adam"):
            if(params["amsgrad"] == "Yes"):
                amsgrad = True;
            else:
                amsgrad = False;
            ptf.optimizer_nesterov_adam(
                float(params["learning_rate"]), 
                beta1=float(params["beta1"]), 
                beta2=float(params["beta2"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]), 
                amsgrad=amsgrad)

        elif(name == "optimizer_adagrad"):
            ptf.optimizer_adagrad(
                float(params["learning_rate"]), 
                learning_rate_decay=float(params["learning_rate_decay"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]))

        elif(name == "optimizer_adadelta"):
            ptf.optimizer_adadelta(
                float(params["learning_rate"]), 
                rho=float(params["rho"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"])); 



    elif(system["backend"] == "Pytorch-1.3.1"):
        name = system["update"]["optimizers"]["value"]["name"];
        params = system["update"]["optimizers"]["value"]["params"];

        if(name == "optimizer_sgd"):
            ptf.optimizer_sgd(
                float(params["learning_rate"]), 
                momentum=float(params["momentum"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_nesterov_sgd"):
            ptf.optimizer_nesterov_sgd(
                float(params["learning_rate"]), 
                momentum=float(params["momentum"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_rmsprop"):
            ptf.optimizer_rmsprop(
                float(params["learning_rate"]), 
                decay_rate=float(params["decay_rate"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]));     

        elif(name == "optimizer_momentum_rmsprop"):
            ptf.optimizer_momentum_rmsprop(
                float(params["learning_rate"]), 
                momentum=float(params["momentum"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_adam"):
            ptf.optimizer_adam(
                float(params["learning_rate"]), 
                beta1=float(params["beta1"]), 
                beta2=float(params["beta2"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_adamax"):
            ptf.optimizer_adamax(
                float(params["learning_rate"]), 
                beta1=float(params["beta1"]), 
                beta2=float(params["beta2"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]))

        elif(name == "optimizer_adamw"):
            if(params["amsgrad"] == "Yes"):
                amsgrad = True;
            else:
                amsgrad = False;
            ptf.optimizer_adamw(
                float(params["learning_rate"]), 
                beta1=float(params["beta1"]), 
                beta2=float(params["beta2"]), 
                weight_decay=float(params["weight_decay"]), 
                amsgrad=amsgrad)

        elif(name == "optimizer_adagrad"):
            ptf.optimizer_adagrad(
                float(params["learning_rate"]), 
                learning_rate_decay=float(params["learning_rate_decay"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]))

        elif(name == "optimizer_adadelta"):
            ptf.optimizer_adadelta(
                float(params["learning_rate"]), 
                rho=float(params["rho"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"])); 



    elif(system["backend"] == "Keras-2.2.5_Tensorflow-1"):
        name = system["update"]["optimizers"]["value"]["name"];
        params = system["update"]["optimizers"]["value"]["params"];

        if(name == "optimizer_sgd"):
            ptf.optimizer_sgd(
                float(params["learning_rate"]), 
                momentum=float(params["momentum"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_nesterov_sgd"):
            ptf.optimizer_nesterov_sgd(
                float(params["learning_rate"]), 
                momentum=float(params["momentum"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_rmsprop"):
            ptf.optimizer_rmsprop(
                float(params["learning_rate"]), 
                decay_rate=float(params["decay_rate"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_adam"):
            ptf.optimizer_adam(
                float(params["learning_rate"]), 
                beta1=float(params["beta1"]), 
                beta2=float(params["beta2"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]));

        elif(name == "optimizer_nesterov_adam"):
            if(params["amsgrad"] == "Yes"):
                amsgrad = True;
            else:
                amsgrad = False;
            ptf.optimizer_nesterov_adam(
                float(params["learning_rate"]), 
                beta1=float(params["beta1"]), 
                beta2=float(params["beta2"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]), 
                amsgrad=amsgrad)

        elif(name == "optimizer_adamax"):
            ptf.optimizer_adamax(
                float(params["learning_rate"]), 
                beta1=float(params["beta1"]), 
                beta2=float(params["beta2"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]))

        elif(name == "optimizer_adagrad"):
            ptf.optimizer_adagrad(
                float(params["learning_rate"]), 
                learning_rate_decay=float(params["learning_rate_decay"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"]))

        elif(name == "optimizer_adadelta"):
            ptf.optimizer_adadelta(
                float(params["learning_rate"]), 
                rho=float(params["rho"]), 
                epsilon=float(params["epsilon"]), 
                weight_decay=float(params["weight_decay"])); 



if(optimizer_reload_status):
    ptf.Reload();











scheduler_reload_status = False;
if(system["update"]["schedulers"]["active"]):
    scheduler_reload_status = True;

    if(system["backend"] == "Mxnet-1.5.1"):
        
        name = system["update"]["schedulers"]["value"]["name"];
        params = system["update"]["schedulers"]["value"]["params"];

        if(name == "lr_fixed"):
            ptf.lr_fixed();

        elif(name == "lr_step_decrease"):
            ptf.lr_step_decrease(
                params["step_size"], 
                gamma=float(params["gamma"]));

        elif(name == "lr_multistep_decrease"):
            milestones = params["milestones"].split(",");
            for i in range(len(milestones)):
                milestones[i] = int(milestones[i]);
            ptf.lr_multistep_decrease(
                milestones, 
                gamma=float(params["gamma"]));
        


    elif(system["backend"] == "Pytorch-1.3.1"):
        name = system["update"]["schedulers"]["value"]["name"];
        params = system["update"]["schedulers"]["value"]["params"];

        if(name == "lr_fixed"):
            ptf.lr_fixed();

        elif(name == "lr_step_decrease"):
            ptf.lr_step_decrease(
                params["step_size"], 
                gamma=float(params["gamma"]));

        elif(name == "lr_multistep_decrease"):
            milestones = params["milestones"].split(",");
            for i in range(len(milestones)):
                milestones[i] = int(milestones[i]);
            ptf.lr_multistep_decrease(
                milestones, 
                gamma=float(params["gamma"]));

        elif(name == "lr_exponential_decrease"):
            ptf.lr_exponential_decrease(
                float(params["gamma"]));

        elif(name == "lr_plateau_decrease"):
            lr_plateau_decrease(
                mode=params["mode"].lower(), 
                factor=float(params["factor"]), 
                patience=int(params["patience"]), 
                verbose=True, 
                threshold=float(params["threshold"]), 
                min_lr=float(params["min_lr"]));

        


    elif(system["backend"] == "Keras-2.2.5_Tensorflow-1"):
        name = system["update"]["schedulers"]["value"]["name"];
        params = system["update"]["schedulers"]["value"]["params"];

        if(name == "lr_fixed"):
            ptf.lr_fixed();

        elif(name == "lr_step_decrease"):
            ptf.lr_step_decrease(
                params["step_size"], 
                gamma=float(params["gamma"]));

        elif(name == "lr_exponential_decrease"):
            ptf.lr_exponential_decrease(
                float(params["gamma"]));

        elif(name == "lr_plateau_decrease"):
            ptf.lr_plateau_decrease(
                mode=params["mode"].lower(), 
                factor=float(params["factor"]), 
                patience=int(params["patience"]), 
                verbose=True, 
                threshold=float(params["threshold"]), 
                min_lr=float(params["min_lr"]));
        


if(scheduler_reload_status):
    ptf.Reload();





loss_reload_status = False;
if(system["update"]["losses"]["active"]):
    scheduler_reload_status = True;

    if(system["backend"] == "Mxnet-1.5.1"):
        
        name = system["update"]["losses"]["value"]["name"];
        params = system["update"]["losses"]["value"]["params"];

        if(name == "loss_l1"):
            ptf.loss_l1(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_l2"):
            ptf.loss_l2(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_softmax_crossentropy"):
            ptf.loss_softmax_crossentropy(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_crossentropy"):
            ptf.loss_crossentropy(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_sigmoid_binary_crossentropy"):
            ptf.loss_sigmoid_binary_crossentropy(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_binary_crossentropy"):
            ptf.loss_binary_crossentropy(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_kldiv"):
            if(params["log_pre_applied"] == "Yes"):
                log_pre_applied = True;
            else:
                log_pre_applied = False;
            ptf.loss_kldiv(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                log_pre_applied=log_pre_applied);

        elif(name == "loss_poisson_nll"):
            if(params["log_pre_applied"] == "Yes"):
                log_pre_applied = True;
            else:
                log_pre_applied = False;
            ptf.loss_poisson_nll(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                log_pre_applied=log_pre_applied);

        elif(name == "loss_huber"):
            ptf.loss_huber(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                threshold_for_mean_estimator=float(params["threshold_for_mean_estimator"]));

        elif(name == "loss_hinge"):
            ptf.loss_hinge(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                margin=float(params["margin"]));

        elif(name == "loss_squared_hinge"):
            ptf.loss_squared_hinge(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                margin=float(params["margin"]));




    elif(system["backend"] == "Pytorch-1.3.1"):
        name = system["update"]["losses"]["value"]["name"];
        params = system["update"]["losses"]["value"]["params"];

        if(name == "loss_l1"):
            ptf.loss_l1(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_l2"):
            ptf.loss_l2(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_softmax_crossentropy"):
            ptf.loss_softmax_crossentropy(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_crossentropy"):
            ptf.loss_crossentropy(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_sigmoid_binary_crossentropy"):
            ptf.loss_sigmoid_binary_crossentropy(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_binary_crossentropy"):
            ptf.loss_binary_crossentropy(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_kldiv"):
            if(params["log_pre_applied"] == "Yes"):
                log_pre_applied = True;
            else:
                log_pre_applied = False;
            ptf.loss_kldiv(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                log_pre_applied=log_pre_applied);

        elif(name == "loss_poisson_nll"):
            if(params["log_pre_applied"] == "Yes"):
                log_pre_applied = True;
            else:
                log_pre_applied = False;
            ptf.loss_poisson_nll(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                log_pre_applied=log_pre_applied);

        elif(name == "loss_huber"):
            ptf.loss_huber(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                threshold_for_mean_estimator=float(params["threshold_for_mean_estimator"]));

        elif(name == "loss_hinge"):
            ptf.loss_hinge(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                margin=float(params["margin"]));

        elif(name == "loss_squared_hinge"):
            ptf.loss_squared_hinge(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                margin=float(params["margin"]));

        elif(name == "loss_multimargin"):
            ptf.loss_multimargin(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_squared_multimargin"):
            ptf.loss_squared_multimargin(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_multilabel_margin"):
            ptf.loss_multilabel_margin(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_multilabel_softmargin"):
            ptf.loss_multilabel_softmargin(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));            

        

    elif(system["backend"] == "Keras-2.2.5_Tensorflow-1"):
        name = system["update"]["losses"]["value"]["name"];
        params = system["update"]["losses"]["value"]["params"];

        if(name == "loss_l1"):
            ptf.loss_l1(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_l2"):
            ptf.loss_l2(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_crossentropy"):
            ptf.loss_crossentropy(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_binary_crossentropy"):
            ptf.loss_binary_crossentropy(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]));

        elif(name == "loss_kldiv"):
            if(params["log_pre_applied"] == "Yes"):
                log_pre_applied = True;
            else:
                log_pre_applied = False;
            ptf.loss_kldiv(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                log_pre_applied=log_pre_applied);

        elif(name == "loss_hinge"):
            ptf.loss_hinge(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                margin=float(params["margin"]));

        elif(name == "loss_squared_hinge"):
            ptf.loss_squared_hinge(
                weight=float(params["weight"]), 
                batch_axis=int(parmas["batch_axis"]),
                margin=float(params["margin"]));            
        
        


if(loss_reload_status):
    ptf.Reload();







# Analysis Project Name
analysis_name = system["analysis"]["input_size"]["analysis_name"];

# Input sizes to explore    
input_sizes = system["analysis"]["input_size"]["list"].split(",");
input_sizes = list(map(int, input_sizes))

# Num epochs for each experiment to run 
epochs=int(system["analysis"]["input_size"]["epochs"]);

# Percentage of original dataset to take in for experimentation
percent_data=int(system["analysis"]["input_size"]["percent"]);

# "keep_all" - Keep all the sub experiments created
# "keep_non" - Delete all sub experiments created   
analysis = ptf.Analyse_Input_Sizes(analysis_name, 
                                    input_sizes, 
                                    percent_data, 
                                    num_epochs=epochs, 
                                    state="keep_none"); 




print("Completed");