#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
workon obj_1_gluoncv_finetune && python train_obj_1_gluoncv_finetune.py