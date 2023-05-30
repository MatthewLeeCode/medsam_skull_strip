#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 6 19:25:36 2023

convert nonCT nii image to npz files, including input image, image embeddings, and ground truth
Compared to pre_CT.py, the main difference is the image intensity normalization method (see line 66-72)

@author: jma
"""
#%% import packages
import numpy as np
import SimpleITK as sitk
import os
join = os.path.join 
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import radvis

# set up the parser
parser = argparse.ArgumentParser(description='preprocess non-CT images')
parser.add_argument('-i', '--nii_path', type=str, default='data/FLARE22Train/images', help='path to the nii images')
parser.add_argument('-gt', '--gt_path', type=str, default='data/FLARE22Train/labels', help='path to the ground truth',)
parser.add_argument('-o', '--npz_path', type=str, default='data/Npz_files', help='path to save the npz files')

parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--modality', type=str, default='CT', help='modality')
parser.add_argument('--anatomy', type=str, default='Abd-Gallbladder', help='anatomy')
parser.add_argument('--img_name_suffix', type=str, default='_0000.nii.gz', help='image name suffix')
parser.add_argument('--label_id', type=int, default=9, help='label id')
parser.add_argument('--prefix', type=str, default='CT_Abd-Gallbladder_', help='prefix')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='work_dir/SAM/sam_vit_b_01ec64.pth', help='checkpoint')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
# seed
parser.add_argument('--seed', type=int, default=2023, help='random seed')
args = parser.parse_args()

prefix = args.modality + '_' + args.anatomy
names = sorted(os.listdir(args.gt_path))
names = [name for name in names if not os.path.exists(join(args.npz_path, prefix + '_' + name.split('.nii.gz')[0]+'.npz'))]
names = [name for name in names if os.path.exists(join(args.nii_path, name))]

# split names into training and testing
np.random.seed(args.seed)
np.random.shuffle(names)
train_names = sorted(names[:int(len(names)*0.8)])
test_names = sorted(names[int(len(names)*0.8):])

# def preprocessing function

# This function preprocesses non-CT images by loading the ground truth and image data, resizing them, and applying a model if provided.
# It takes in the ground truth path, nii path, ground truth name, image name, label id, image size, sam model, and device as inputs.
def preprocess_nonct(gt_path, nii_path, gt_name, image_name, label_id, image_size, sam_model, device):
    # Load the ground truth image and convert it to binary
    gt_sitk = radvis.load_image(join(gt_path, gt_name))
    gt_data = gt_sitk.image_data
    gt_data = np.uint8(gt_data==label_id)
    
    # If the ground truth has more than 1000 pixels, preprocess the image
    if np.sum(gt_data)>1000:
        imgs = []
        gts =  []
        img_embeddings = []
        
        # Check that the ground truth is binary
        assert np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2, 'ground truth should be binary'
        
        # Load the image data and preprocess it
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre[image_data==0] = 0
        image_data_pre = np.uint8(image_data_pre)
        
        # Get the z index of the ground truth and resize the slices
        z_index, _, _ = np.where(gt_data>0)
        z_min, z_max = np.min(z_index), np.max(z_index)
        # Go through all the indices between z_min and z_max
        for i in range(z_min, z_max):
            gt_slice_i = gt_data[i,:,:]
            gt_slice_i = transform.resize(gt_slice_i, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
            
            # If the ground truth has more than 100 pixels, resize the image slice and convert it to three channels
            if np.sum(gt_slice_i)>100:
                img_slice_i = transform.resize(image_data_pre[i,:,:], (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
                img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))
                
                # Check that the image is 3 channels and has the same size as the ground truth
                assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
                assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
                
                # Append the image slice and ground truth slice to their respective lists
                imgs.append(img_slice_i)
                assert np.sum(gt_slice_i)>100, 'ground truth should have more than 100 pixels'
                gts.append(gt_slice_i)
                
                # If a sam model is provided, apply it to the image slice and append the embedding to the list
                if sam_model is not None:
                    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                    resize_img = sam_transform.apply_image(img_slice_i)
                    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:])
                    assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                    with torch.no_grad():
                        embedding = sam_model.image_encoder(input_image)
                        img_embeddings.append(embedding.cpu().numpy()[0])
    
    # If a sam model is provided, return the image slices, ground truth slices, and embeddings
    if sam_model is not None:
        return imgs, gts, img_embeddings
    # Otherwise, return the image slices and ground truth slices
    else:
        return imgs, gts


#%% prepare the save path
save_path_tr = join(args.npz_path, prefix, 'train')
save_path_ts = join(args.npz_path, prefix, 'test')
os.makedirs(save_path_tr, exist_ok=True)
os.makedirs(save_path_ts, exist_ok=True)

#%% set up the model
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

for name in tqdm(train_names):
    image_name = name
    gt_name = name 
    imgs, gts, img_embeddings = preprocess_nonct(args.gt_path, args.nii_path, gt_name, image_name, args.label_id, args.image_size, sam_model, args.device)
    #%% save to npz file
    # stack the list to array
    if len(imgs)>1:
        imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
        gts = np.stack(gts, axis=0) # (n, 256, 256)
        img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64)
        np.savez_compressed(join(save_path_tr, prefix + '_' + gt_name.split('.nii.gz')[0]+'.npz'), imgs=imgs, gts=gts, img_embeddings=img_embeddings)
        # save an example image for sanity check
        idx = np.random.randint(0, imgs.shape[0])
        img_idx = imgs[idx,:,:,:]
        gt_idx = gts[idx,:,:]
        bd = segmentation.find_boundaries(gt_idx, mode='inner')
        img_idx[bd, :] = [255, 0, 0]
        io.imsave(save_path_tr + '.png', img_idx, check_contrast=False)

# save testing data
for name in tqdm(test_names):
    image_name = name
    gt_name = name 
    imgs, gts = preprocess_nonct(args.gt_path, args.nii_path, gt_name, image_name, args.label_id, args.image_size, sam_model=None, device=args.device)
    #%% save to npz file
    if len(imgs)>1:
        imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
        gts = np.stack(gts, axis=0) # (n, 256, 256)
        img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64)
        np.savez_compressed(join(save_path_ts, prefix + '_' + gt_name.split('.nii.gz')[0]+'.npz'), imgs=imgs, gts=gts)
        # save an example image for sanity check
        idx = np.random.randint(0, imgs.shape[0])
        img_idx = imgs[idx,:,:,:]
        gt_idx = gts[idx,:,:]
        bd = segmentation.find_boundaries(gt_idx, mode='inner')
        img_idx[bd, :] = [255, 0, 0]
        io.imsave(save_path_ts + '.png', img_idx, check_contrast=False)