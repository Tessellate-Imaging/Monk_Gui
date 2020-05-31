#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
workon monk_cls && python evaluate_cls_image_single.py