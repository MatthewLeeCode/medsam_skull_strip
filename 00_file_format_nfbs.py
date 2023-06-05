import os
import shutil
import argparse

def define_parser() -> argparse.ArgumentParser:
    """ Defines the parser for the script """
    parser = argparse.ArgumentParser(description="Organize Brain images and masks")
    parser.add_argument('-i', '--inp_path', type=str, default='NFBS_Dataset', help='path to the source folders "images" and "masks"')
    parser.add_argument('-o', '--out_path', type=str, default='nfbs_dataset_preprocessed', help='path to save the organized images and masks')
    parser.add_argument('--filetype', type=str, default='nii.gz', help='filetype of the images')
    return parser

def organize_images(args):
    """ Organize images and masks into separate directories """

    # create output directories if they don't exist
    image_dir = os.path.join(args.out_path, 'images')
    mask_dir = os.path.join(args.out_path, 'masks')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # iterate through files in the source directory
    for root, dirs, files in os.walk(args.inp_path):
        for file in files:
            if file.endswith(args.filetype):
                id = file.split('-')[1].split("_")[0][1:]
                source_file_path = os.path.join(root, file)
                if '_brainmask' in file:
                    target_file_path = os.path.join(mask_dir, id + '.' + args.filetype)
                    shutil.copy2(source_file_path, target_file_path)
                elif '_brain' not in file:
                    target_file_path = os.path.join(image_dir, id + '.' + args.filetype)
                    shutil.copy2(source_file_path, target_file_path)

def main():
    parser = define_parser()
    args = parser.parse_args()
    organize_images(args)

if __name__ == "__main__":
    main()
