import numpy as np
import radvis as rv
import os
from os.path import join
from typing import Tuple

import torch
from segment_anything import sam_model_registry, SamPredictor
import argparse
from typing import Optional
from skimage import transform, io, segmentation
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling import Sam
from tqdm import tqdm


def define_parser() -> argparse.ArgumentParser:
    """ Defines the parser for the preprocess script

    :return: the parser
    """
    parser = argparse.ArgumentParser(description='Process MRI Images')
    parser.add_argument('-i', '--inp_path', type=str, default='data/', help='path to the folders "images" and "masks"')
    parser.add_argument('-o', '--npz_path', type=str, default='data/', help='path to save the npz files')
    parser.add_argument('-p', '--prefix', type=str, help='Prefix to save')

    parser.add_argument('--filetype', type=str, default='nii.gz', help='filetype of the images')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--label_id', type=int, default=1, help='label id for the mask')
    parser.add_argument('--model_type', type=str, default='vit_b', help='MedSAM model type')
    parser.add_argument('--checkpoint', type=str, default='work_dir/SAM/sam_vit_b_01ec64.pth', help='checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    return parser


def preprocess_image(image: rv.RadImage, mask: rv.RadImage) -> Tuple[rv.RadImage, rv.RadImage]:
    """ Performs preprocessing for medsam

    1. Clip image between 0.5 and 99.5 percentile
    2. Normalize image to 0-255
    3. Add padding on each axis to meet 256x256x256

    :param image: the image to preprocess
    :param mask: the mask to preprocess

    :return: the preprocessed image and mask
    """
    image = rv.percentile_clipping(image, 0.5, 99.5)

    image = rv.normalization(
        image, 
         min_val=image.image_data.min(), 
         max_val=image.image_data.max()
    )
    image.image_data *= 255

    image = rv.add_padding(image, target_shape=(256, 256, 256))
    mask = rv.add_padding(mask, target_shape=(256, 256, 256))

    return image, mask

def load_image_and_groundtruth(file_path: str, image_id: str, label_id: int, filetype: str="nii.gz") -> Tuple[Optional[rv.RadImage], Optional[rv.RadImage]]:
    """ Loads the image and the ground truth mask
    
    :param file_path: the path to the image and ground truth
    :param image_id: the image id that matches both the image and the ground truth
    :param label_id: the label id of the mask (The groundtruth could have multiple masks)

    :return: the image and the ground truth mask
    """
    image_file = join(file_path, "images", image_id + "." + filetype)
    ground_truth_file = join(file_path, "masks", image_id + "." + filetype)
    try:
        image = rv.load_image(image_file)
        ground_truth = rv.load_image(ground_truth_file)

        ground_truth.image_data = np.where(ground_truth.image_data == label_id, 1, 0)
        
        return image, ground_truth
    except Exception as e:
        
        print(f"Error loading image [{image_file}]: {e}")
        return None, None


def get_z_range(mask: rv.RadImage) -> Tuple[int, int]:
    """ Retrieves the minimum and maximum z indices that contain the mask

    :param mask: the mask of binary 0-1 values

    :return: the minimum and maximum z indices
    """
    z_index = np.where(mask.image_data > 0)[0]
    z_min = int(np.min(z_index))
    z_max = int(np.max(z_index))
    return z_min, z_max


def run_sam_model(sam_model: Sam, device: str, img_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Runs the sam model on the slices

    :param model: the sam model
    :param slices: the slices to run through the model

    :return: the embeddings and the attention maps
    """
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size) # type: ignore
    resize_img = sam_transform.apply_image(img_slice)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # type: ignore
    assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024' # type: ignore
    with torch.no_grad():
        embedding = sam_model.image_encoder(input_image) # type: ignore
    return embedding.cpu().numpy()[0]


def main():
    """ Main function for the preprocess script

    Performs the following steps:
    1. Loads the param arguments
    2. Loops through all the image ids
    3. Loads the image and ground truth
    4. Preprocesses the image and ground truth
    5. Identify the min_z and max_z indices where the mask is present
    6. Get the slices of the image and ground truth based on the min_z and max_z indices
    7. Run the slices through the encoder
    8. Save the npz file
    
    """
    # Load the arguments
    parser = define_parser()
    args = parser.parse_args()

    # Load SAM model
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

    # Loop through all the image ids
    image_ids = [f.split(".")[0] for f in os.listdir(join(args.inp_path, "images"))]
    
    for image_id in tqdm(image_ids):
        # Load the image and ground truth
        image, ground_truth = load_image_and_groundtruth(args.inp_path, image_id, args.label_id, args.filetype)
        
        if image is None or ground_truth is None:
            continue

        # Preprocess the image and ground truth
        image, ground_truth = preprocess_image(image, ground_truth)

        # Identify the min_z and max_z indices where the mask is present
        min_z, max_z = get_z_range(ground_truth)
        
        # Run the slices through the encoder
        embeddings = []
        images = []
        groundtruths = []

        ground_truth_slice = np.array([])
        for i in range(min_z, max_z):
            ground_truth_slice:np.ndarray = ground_truth.image_data[i]
            
            ground_truth_slice = transform.resize(
                ground_truth_slice, 
                (args.image_size, args.image_size), 
                order=0, 
                preserve_range=True, 
                mode='constant', 
                anti_aliasing=True
            ) # Resize to the image size passed
            ground_truth_slice = ground_truth_slice.astype(np.uint8)

            # Check if the mask has <= 100 values. If so, skip this slice
            if np.sum(ground_truth_slice) <= 100:
                continue

            image_slice = image.image_data[i]
            image_slice = transform.resize(
                image_slice, 
                (args.image_size, args.image_size), 
                order=3, 
                preserve_range=True, 
                mode='constant', 
                anti_aliasing=True
            ) # Resize to the image size passed
            # SAM model expects 3 channels
            image_slice = np.repeat(image_slice[:,:,None], 3, axis=-1)
            image_slice = image_slice.astype(np.uint8)

            # Check that the image is setup correctly
            assert len(image_slice.shape)==3 and image_slice.shape[2]==3, 'image should be 3 channels'
            assert image_slice.shape[0]==ground_truth_slice.shape[0] and image_slice.shape[1]==ground_truth_slice.shape[1], 'image and ground truth should have the same size'

            # Save the image and ground truth
            images.append(image_slice)
            groundtruths.append(ground_truth_slice)

            # Run the model
            embedding = run_sam_model(sam_model, args.device, image_slice)
            embeddings.append(embedding)
    
        # stack the list to array
        if len(images) > 1:
            images = np.stack(images, axis=0) # (n, 256, 256, 3)
            groundtruths = np.stack(groundtruths, axis=0) # (n, 256, 256)
            embeddings = np.stack(embeddings, axis=0) # (n, 1, 256, 64, 64)
            # Convert to float16 to save space
            embeddings = embeddings.astype(np.float16)
            np.savez_compressed(join(args.npz_path, args.prefix + '_' + image_id + '.npz'), imgs=images, gts=groundtruths, img_embeddings=embeddings)
            
            # save an example image for sanity check
            idx = np.random.randint(0, images.shape[0])
            img_idx = images[idx,:,:,:]
            gt_idx = groundtruths[idx,:,:]
            bd = segmentation.find_boundaries(gt_idx, mode='inner')
            img_idx[bd, :] = [255, 0, 0]
            io.imsave(join(args.npz_path, args.prefix + '_' + image_id + '.png'), img_idx, check_contrast=False)


if __name__ == "__main__":
    main()