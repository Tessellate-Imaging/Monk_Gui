#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh


mkvirtualenv -p /usr/bin/python3.6 obj_1_gluoncv_finetune

workon obj_1_gluoncv_finetune && cat Monk_Object_Detection/1_gluoncv_finetune/installation/requirements_cuda10.1.txt | xargs -n 1 -L 1 pip install 


echo "Completed"