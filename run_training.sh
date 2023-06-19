#!/bin/bash
conda activate medsam

# loop over the three axes
for AXIS in 1 2
do
    python 03_finetune.py --axis $AXIS
done