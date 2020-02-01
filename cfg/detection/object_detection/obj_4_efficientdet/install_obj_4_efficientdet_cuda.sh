#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh


mkvirtualenv -p /usr/bin/python3.6 obj_4_efficientdet

workon obj_4_efficientdet && cat Monk_Object_Detection/4_efficientdet/installation/requirements.txt | xargs -n 1 -L 1 pip install 

echo "Completed"