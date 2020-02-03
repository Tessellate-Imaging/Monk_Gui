# Monk_Gui [![HitCount](http://hits.dwyl.io/Tessellate-Imaging/Monk_Gui.svg)](http://hits.dwyl.io/Tessellate-Imaging/Monk_Gui)

A Graphical user Interface for deep learning and computer vision over Monk Libraries
<br />
<br />
<br />

![Alt Text](complete.gif)

<br />
<br />

# Backend Libraries

- A) Monk - https://github.com/Tessellate-Imaging/monk_v1 
    - Monk is a low code Deep Learning tool and a unified wrapper for Computer Vision. 
- B) Monk Object Detection - https://github.com/Tessellate-Imaging/Monk_Object_Detection
    - A one-stop repository for low-code easily-installable object detection pipelines. 
    - Check licenses of each pipeline before using 
<br />
<br />
<br />


# Installation

` $ sudo apt-get install python3.6 python3.6-dev python3.7 python3.7-dev python3-pip`

` $ sudo pip install virtualenv virtualenvwrapper`

` $ $ echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc`

` $ echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc`

` $ echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc`

` $ echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc`

` $ source ~/.bashrc`

` $ mkvirtualenv -p /usr/bin/python3.6 monk_gui`

` $ workon monk_gui && pip install numpy pyqt5 tqdm`
<br />
<br />
<br />

# Running GUI

` workon monk_gui`

` python gui.py`


## Author
Tessellate Imaging - https://www.tessellateimaging.com/
   
Check out Monk AI - (https://github.com/Tessellate-Imaging/monk_v1)
    
    Monk features
        - low-code
        - unified wrapper over major deep learning framework - keras, pytorch, gluoncv
        - syntax invariant wrapper

    Enables developers
        - to create, manage and version control deep learning experiments
        - to compare experiments across training metrics
        - to quickly find best hyper-parameters

To contribute to Monk AI or Monk Object Detection repository raise an issue in the git-repo or dm us on linkedin 
   - Abhishek - https://www.linkedin.com/in/abhishek-kumar-annamraju/
   - Akash - https://www.linkedin.com/in/akashdeepsingh01/
<br />
<br />
<br />


## Copyright

Copyright 2019 onwards, Tessellate Imaging Private Limited Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.



