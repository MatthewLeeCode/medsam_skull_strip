from os.path import join
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

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


# Create a dictionary to store the data
data = {
    "image_id": [],
    "image_location": [],
    "mask_location": [],
    "dataset_group": [],
    "random_state": []
}

# Populate the dictionary with the data
for i, filename in enumerate(embedding_ids):
    image_id = filename.split(".")[0]
    image_location = join(IMAGES_FOLDER, f"{image_id}.nii.gz")
    mask_location = join(MASKS_FOLDER, f"{image_id}_mask.nii.gz")
    embeddings_location = join(EMBEDDINGS_FOLDER, filename)
    dataset_group = "train" if i < len(train_ids) else "test"
    random_state = 2023
    
    data["image_id"].append(image_id)
    data["image_location"].append(image_location)
    data["mask_location"].append(mask_location)
    data["dataset_group"].append(dataset_group)
    data["random_state"].append(random_state)

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("data/mni_transform.csv", index=False)
