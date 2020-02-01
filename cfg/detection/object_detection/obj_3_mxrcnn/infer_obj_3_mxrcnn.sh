#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh


workon obj_3_mxrcnn

python infer_obj_3_mxrcnn.py