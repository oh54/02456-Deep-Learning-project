#!/bin/bash

source activate tensorflow_p36

python3.6 DEC_tryout2.py 2>&1 > some_log


# use 'nohup ./run.sh &' to run this file and
# tail -f some_log to see scrolling log view 


