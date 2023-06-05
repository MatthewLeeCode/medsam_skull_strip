import argparse


def define_parser() -> argparse.ArgumentParser:
    """ Defines the parser for the preprocess script """
    parser = argparse.ArgumentParser(description="Align Brain images with MNI152")
    parser.add_argument('-i', '--inp_path', type=str, default='data/', help='path to the folders "images" and "masks"')
    parser.add_argument('-o', '--npz_path', type=str, default='data/', help='path to save the npz files')
    parser.add_argument('-p', '--prefix', type=str, help='Prefix to save')

    parser.add_argument('--filetype', type=str, default='nii.gz', help='filetype of the images')
    return parser


def main() -> None:
    pass


if __name__ == "__main__":
    main()
