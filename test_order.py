from os.path import join
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

AXIS = 0
DATA_LOCATION = "data/medsam"
IMAGES_FOLDER = join(DATA_LOCATION, "images")
MASKS_FOLDER = join(DATA_LOCATION, "masks")
EMBEDDINGS_FOLDER = join(DATA_LOCATION, "embeddings", f"axis_{AXIS}")
DATA_SPLIT = 0.2
MODEL_VERSION = "0.0.1"
MODEL_TASK = "brainstrip"
MODEL_SAVE_PATH = f"models/{MODEL_TASK}/{MODEL_VERSION}/{AXIS}"

# Create folder for save path
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


embedding_ids = [i for i in os.listdir(EMBEDDINGS_FOLDER) if i.endswith(".npz")]
# Train test split
train_ids, test_ids = train_test_split(embedding_ids, test_size=DATA_SPLIT, random_state=2023)
train_filenames = [join(EMBEDDINGS_FOLDER, i) for i in train_ids]
test_filenames = [join(EMBEDDINGS_FOLDER, i) for i in test_ids]
print(train_filenames)
print(test_filenames)