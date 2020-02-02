#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh


mkvirtualenv -p /usr/bin/python3.6 obj_5_pytorch_retinanet

workon obj_5_pytorch_retinanet && cat Monk_Object_Detection/5_pytorch_retinanet/installation/requirements.txt | xargs -n 1 -L 1 pip install 

echo "Completed"