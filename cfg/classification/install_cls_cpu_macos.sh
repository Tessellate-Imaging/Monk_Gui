#
#!/bin/bash

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.6
export WORKON_HOME=$HOME/.virtualenvs
. $HOME/.local/bin/virtualenvwrapper.sh
. /usr/local/bin/virtualenvwrapper.sh

mkvirtualenv -p /usr/bin/python3.6 monk_cls

workon monk_cls && pip install monk-cpu

echo "Completed"