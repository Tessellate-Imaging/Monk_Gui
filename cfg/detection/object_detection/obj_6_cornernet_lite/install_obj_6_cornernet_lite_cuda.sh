#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh


mkvirtualenv -p /usr/bin/python3.6 obj_6_cornernet_lite

workon obj_6_cornernet_lite && cat Monk_Object_Detection/6_cornernet_lite/installation/requirements.txt | xargs -n 1 -L 1 pip install 

cd Monk_Object_Detection/6_cornernet_lite/lib/core/models/py_utils/_cpools/ && pip install -e .

cd ../../../external && make

echo "Completed"