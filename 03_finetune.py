import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
from sklearn.model_selection import train_test_split
import radvis as rv
import monai

torch.manual_seed(2023)
np.random.seed(2023)

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

print(f"{len(train_ids)} training samples")
print(f"{len(test_ids)} testing samples")


class NpzDataset(Dataset):
    def __init__(self, data_filenames):
        self.npz_files = data_filenames
        self.all_data = self.load_all_data()

    def load_all_data(self):
        all_data = []
        for f in self.npz_files:
            data = np.load(f)
            all_data.append({'gts': data['gts'], 'img_embeddings': data['img_embeddings']})
        return all_data

    def __len__(self):
        return sum([data['gts'].shape[0] for data in self.all_data])

    def __getitem__(self, index):
        for data in self.all_data:
            if index < data['gts'].shape[0]:
                ori_gt = data['gts'][index]
                img_embedding = data['img_embeddings'][index]
                break
            else:
                index -= data['gts'].shape[0]

        y_indices, x_indices = np.where(ori_gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = ori_gt.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        bboxes = np.array([x_min, y_min, x_max, y_max])

        # convert img embedding, mask, bounding box to torch tensor
        return (torch.tensor(img_embedding).float(), torch.tensor(ori_gt[None, :,:]).long(), torch.tensor(bboxes).float())


demo_dataset = NpzDataset(train_filenames)
demo_dataloader = DataLoader(demo_dataset, batch_size=8, shuffle=False)
for img_embed, gt2D, bboxes in demo_dataloader:
    # img_embed: (B, 256, 64, 64), gt2D: (B, 1, 256, 256), bboxes: (B, 4)
    print(f"{img_embed.shape=}, {gt2D.shape=}, {bboxes.shape=}")
    break

# prepare SAM model
model_type = 'vit_b'
checkpoint = 'work_dir/MedSAM/medsam_20230423_vit_b_0.0.1.pth'
device = 'cuda:0'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
sam_model.train()

# Set up the optimizer, hyperparameter tuning will improve performance here
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


from tqdm import tqdm

num_epochs = 100
losses = []
best_loss = 1e10
train_dataset = NpzDataset(train_filenames)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0
    step = 0
    # train
    for image_embedding, gt2D, boxes in tqdm(train_dataloader):
        # do not compute gradients for image encoder and prompt encoder
        with torch.no_grad():
            # convert box to 1024x1024 grid
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)
            # get prompt embeddings 
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        # predicted masks
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          )

        loss = seg_loss(mask_predictions, gt2D.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step += 1

    losses.append(epoch_loss)
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
    # save the latest model checkpoint
    torch.save(sam_model.state_dict(), join(MODEL_SAVE_PATH, 'sam_model_latest.pth'))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(sam_model.state_dict(), join(MODEL_SAVE_PATH, 'sam_model_best.pth'))