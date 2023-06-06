#!/bin/bash

# the path to your python script
SCRIPT="02_create_embeddings.py"

# the input and output directories
INP_PATH="data/mni_transform"
NPZ_PATH="data/medsam/embeddings"

# Model checkpoint to use
CHECKPOINT="work_dir/MedSAM/medsam_20230423_vit_b_0.0.1.pth"

# loop over the three axes
for AXIS in 0 1 2
do
    PREFIX="axis_${AXIS}"
    python $SCRIPT -i $INP_PATH -o $NPZ_PATH -p $PREFIX -a $AXIS --checkpoint $CHECKPOINT
done