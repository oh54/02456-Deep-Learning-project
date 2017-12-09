#!/bin/bash

source activate tensorflow_p36

python3.6 DEC_aws.py 2>&1 > some_log

#python3.6 DEC_local.py 2>&1 > some_log


# use 'nohup ./run_aws.sh &' to run this file and
# tail -f some_log to see scrolling log view 


