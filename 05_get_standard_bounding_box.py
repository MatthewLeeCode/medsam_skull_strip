import argparse
import pandas as pd
import numpy as np


def create_parser() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", type=int, default=0, help="Axis to get standard bounding box from")
    parser.add_argument("--data_location", type=str, default="data/medsam/embeddings", help="Location of data")
    parser.add_argument("--metadata_location", type=str, default="data/train_test.csv", help="Location of metadata")
    parser.add_argument("--buffer", type=int, default=2, help="Buffer to add to bounding box")
    args = parser.parse_args()
    return args


def get_mask_min_max(mask_location: str, axis:int, index:int) -> np.ndarray:
    """ Gets the min and max of the mask along the specified axis at the specified index

    Args:
        mask_location (str): Location of mask
        axis (int): Axis to get min and max of
        index (int): Index of mask to get min and max of

    Returns:
        np.ndarray: Min and max of mask along specified axis
    """
    pass


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    AXIS = args.axis
    DATA_LOCATION = args.data_location
    BUFFER = args.buffer
    METADATA_LOCATION = args.metadata_location

    # Headings: image_id, image_location, mask_location, dataset_group, random_state
    metadata = pd.read_csv(METADATA_LOCATION)
    
    # Get all train image ids
    train_ids = metadata[metadata['dataset_group'] == 'train']['image_id'].values.tolist()

if __name__ == "__main__":
    main()