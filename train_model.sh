#!/bin/bash

conda activate medsam 

python 03_finetune.py --axis 0
python 03_finetune.py --axis 1
python 03_finetune.py --axis 2