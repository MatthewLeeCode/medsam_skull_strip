import argparse
import numpy as np
import radvis as rv
from nipype.interfaces import fsl
from nipype.interfaces.fsl import FLIRT, ConvertXFM
from os.path import join
import os
from tqdm import tqdm
from glob import glob
import radvis as rv


def define_parser() -> argparse.ArgumentParser:
    """ Defines the parser for the preprocess script """
    parser = argparse.ArgumentParser(description="Align Brain images with MNI152")
    parser.add_argument('-i', '--mask_path', type=str, default='data/medsam/', help='path to the masks')
    parser.add_argument('-t', '--transform_path', type=str, default='data/mni_transform/transforms', help='path to the transforms')
    parser.add_argument('--image_path', type=str, default='data/mni_transform/images', help='path to the images')
    parser.add_argument('-o', '--output', type=str, default='data/', help='path to save the npz files')
    parser.add_argument('-a', '--axis', type=int, default=0, help='axis of the images')
    parser.add_argument('-r', '--reference_path', type=str, default='/mnt/c/Users/matth/Data/brainseg/nfbs_dataset_preprocessed/images', help='path to the reference image')
    parser.add_argument('--version', type=str, default='0.0.1', help='version of the dataset')

    return parser


def convert_mask_to_original_space(mask_path:str, out_location:str, transform_mat:str, reference_image:str):
    """
    Convert mask back to the original space using an inverse transformation matrix.

    Args:
        mask_path (str): Path to the mask file in NIfTI format.
        out_location (str): Path to the output location.
        transform_mat (str): Path to the transformation matrix file.
        reference_image (str): Path to the reference image file.
        
    Returns:
        str: Path to the output file.
    """
    # Define an instance of the FSL FLIRT interface
    flirt = FLIRT()

    # Set the input image (mask) and the reference image
    flirt.inputs.in_file = mask_path
    flirt.inputs.reference = reference_image

    # Set the output file name
    output_file = os.path.join(out_location, os.path.basename(mask_path))
    flirt.inputs.out_file = output_file

    # Set the transformation matrix file
    flirt.inputs.in_matrix_file = transform_mat

    # Apply the transformation
    flirt.inputs.apply_xfm = True

    # Run the FLIRT interface
    flirt.run()

    return output_file

def main():
    args = define_parser().parse_args()
    image_path = join(args.mask_path, args.version, "results", f"axis_{args.axis}")

    image_ids = [x.split(".")[0].split("_")[0] for x in os.listdir(join(image_path)) if x.find("segmentation") == -1]
    image_ids = list(set(image_ids))

    for image_id in tqdm(image_ids):
        mask_path = join(image_path, f"{image_id}_segmentation.nii.gz")
        transform = join(args.transform_path, f"{image_id}.mat")
        reference_image = join(args.reference_path, f"{image_id}.nii.gz")

        output_mask = convert_mask_to_original_space(mask_path, args.output, transform, reference_image)

        img = rv.load_image(reference_image)
        mask = rv.load_image(output_mask)
        slicer = rv.RadSlicer(img, axis=0)
        slicer.add_mask(mask)
        slicer.display()


if __name__ == "__main__":
    main()