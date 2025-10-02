#!/bin/bash

python controlnets/run_single_prep_compare.py 0 > model_0.log
python controlnets/run_single_prep_compare.py 1 > model_1.log
python controlnets/run_single_prep_compare.py 2 > model_2.log
