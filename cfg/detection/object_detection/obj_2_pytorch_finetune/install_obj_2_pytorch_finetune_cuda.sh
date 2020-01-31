#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh


mkvirtualenv -p /usr/bin/python3.6 obj_2_pytorch_finetune

workon obj_2_pytorch_finetune && cat Monk_Object_Detection/2_pytorch_finetune/installation/requirements.txt | xargs -n 1 -L 1 pip install 

echo "Completed"