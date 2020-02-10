#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh


mkvirtualenv -p /usr/bin/python3.6 monk_cls

workon monk_cls && cat monk_v1/installation/requirements_cpu.txt | xargs -n 1 -L 1 pip install 

echo "Completed"