import argparse
import numpy as np
import radvis as rv
from nipype.interfaces import fsl
from nipype.interfaces.fsl import FLIRT, ConvertXFM
from os.path import join
import os
from tqdm import tqdm
from glob import glob


def define_parser() -> argparse.ArgumentParser:
    """ Defines the parser for the preprocess script """
    parser = argparse.ArgumentParser(description="Align Brain images with MNI152")
    parser.add_argument('-i', '--inp_path', type=str, default='data/', help='path to the folders "images" and "masks"')
    parser.add_argument('-o', '--out_path', type=str, default='data/', help='path to save the npz files')
    parser.add_argument('-p', '--prefix', type=str, help='Prefix to save')

    parser.add_argument('--filetype', type=str, default='nii.gz', help='filetype of the images')
    return parser


def convert_image_to_mni(image_path:str, out_location:str):
    # File paths
    standard_img = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
    
    # Get the ID of the image from the path
    image_id = image_path.split('/')[-1].split('.')[0]
    transform_mat = join(out_location, "transforms", f"{image_id}.mat")  

    # Register image
    flirt_img = FLIRT()
    flirt_img.inputs.in_file = image_path
    flirt_img.inputs.reference = standard_img
    flirt_img.inputs.out_file = join(out_location, "images", f"{image_id}_mni.nii.gz")
    flirt_img.inputs.out_matrix_file = transform_mat
    flirt_img.run()

    return transform_mat


def convert_mask_to_original_space(mask_path:str, out_location:str, transform_mat:str):
    # File paths
    original_mask = mask_path  # Use original mask as reference 

    # Get the ID of the mask from the path
    mask_id = mask_path.split('/')[-1].split('.')[0]
    
    # Create output file path
    out_file = join(out_location, "masks", f"{mask_id}_original.nii.gz")
    
    # Get inverse of the transformation matrix
    invt = ConvertXFM()
    invt.inputs.in_file = transform_mat
    invt.inputs.out_file = join(out_location, "transforms", f"{mask_id}_inverse.mat")
    invt.run()

    # Register mask back to original space using inverse transformation matrix
    flirt_mask = FLIRT()
    flirt_mask.inputs.in_file = join(out_location, "masks", f"{mask_id}_mni.nii.gz")
    flirt_mask.inputs.reference = original_mask
    flirt_mask.inputs.out_file = out_file
    flirt_mask.inputs.apply_xfm = True
    flirt_mask.inputs.in_matrix_file = invt.inputs.out_file
    flirt_mask.inputs.interp = 'nearestneighbour'
    flirt_mask.run()

    return out_file


def convert_mask_to_mni(mask_path:str, out_location:str, transform_mat:str):
    # File paths
    standard_img = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')

    # Get the ID of the mask from the path
    mask_id = mask_path.split('/')[-1].split('.')[0]

    # Register mask
    flirt_mask = FLIRT()
    flirt_mask.inputs.in_file = mask_path
    flirt_mask.inputs.reference = standard_img
    flirt_mask.inputs.out_file = join(out_location, "masks", f"{mask_id}_mni.nii.gz")
    flirt_mask.inputs.apply_xfm = True
    flirt_mask.inputs.in_matrix_file = transform_mat
    flirt_mask.inputs.interp = 'nearestneighbour'
    flirt_mask.run()


def convert_to_mni(image_path:str, mask_path:str, out_location:str):
    # Assert the image and mask IDs match
    image_id = image_path.split('/')[-1].split('.')[0]
    mask_id = mask_path.split('/')[-1].split('.')[0]
    assert image_id == mask_id, "Image and mask ID's do not match"

    # Convert image and mask to MNI space
    transform_mat = convert_image_to_mni(image_path, out_location)
    convert_mask_to_mni(mask_path, out_location, transform_mat)


def load_images(id:str, path:str) -> rv.RadImage:
    """
    Gets the image and the mask from the images and masks folder respectively. 
    Filenames are '{path}/{id}.nii.gz'
    """
    image_path = f'{path}/images/{id}.nii.gz'
    mask_path = f'{path}/masks/{id}.nii.gz'
    return rv.load_image(image_path), rv.load_image(mask_path)


def main() -> None:
    parser = define_parser()
    args = parser.parse_args()

    # Check if subfolders exist in the output directory, if not, create them
    os.makedirs(join(args.out_path, args.prefix, "images"), exist_ok=True)
    os.makedirs(join(args.out_path, args.prefix, "masks"), exist_ok=True)
    os.makedirs(join(args.out_path, args.prefix, "transforms"), exist_ok=True)

    # Find all images / masks
    image_paths = glob(join(args.inp_path, "images", "*." + args.filetype))
    mask_paths = glob(join(args.inp_path, "masks", "*." + args.filetype))

    assert len(image_paths) == len(mask_paths), "The number of images and masks do not match."

    # Loop through them
    for image_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        # Ensure that the image and mask IDs match
        assert image_path.split('/')[-1] == mask_path.split('/')[-1], "Image and mask ID's do not match"

        # Convert them to MNI
        convert_to_mni(image_path, mask_path, join(args.out_path, args.prefix))


if __name__ == "__main__":
    main()
