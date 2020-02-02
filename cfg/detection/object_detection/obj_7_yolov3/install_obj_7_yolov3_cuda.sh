#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh


mkvirtualenv -p /usr/bin/python3.6 obj_7_yolov3

workon obj_7_yolov3 && cat Monk_Object_Detection/7_yolov3/installation/requirements.txt | xargs -n 1 -L 1 pip install 

echo "Completed"