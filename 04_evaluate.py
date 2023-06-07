"""
Run the model on the test set and save the results
"""
import argparse
import pandas as pd
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from os.path import join
import numpy as np
import torch


def create_parser() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="data/train_test.csv", help="Location of metadata")
    parser.add_argument("--model_version", type=str, default="0.0.1", help="Version of model")
    parser.add_argument("--model_location", type=str, default="models/brainstrip", help="Location of model")
    parser.add_argument("--sam_model_type", type=str, default="vit_b", help="SAM model type")
    parser.add_argument("--axis", type=int, default=0, help="Axis to perform segmentation on")
    parser.add_argument("--save_location", type=str, default="data/medsam/embeddings/results", help="Location to save segmentation results and metadata")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on")
    return parser


def load_embedding_and_groundtruth(image_location: str, axis: int, image_id: str) -> np.ndarray:
    """ Loads the embedding and groundtruth of an image

    Args:
        image_location (str): Location of image
        axis (int): Axis to perform segmentation on
        image_id (str): Image id

    Returns:
        np.ndarray: Embedding of image
        np.ndarray: image
        np.ndarray: Groundtruth of image
    """
    filename = join(image_location, f"axis_{axis}", f"{image_id}.npz")
    embeddings = np.load(filename)

    image = embeddings['imgs']
    embedding = embeddings['img_embeddings']
    groundtruth = embeddings['gts']

    assert image.shape[:2] == groundtruth.shape[:2], f"Image and groundtruth shapes do not match: {image.shape} != {groundtruth.shape}"
    return embedding, image, groundtruth


def create_bounding_box(groundtruth: np.ndarray, index: int, buffer: int = 1) -> np.ndarray:
    """ Creates the x_min, x_max, y_min, y_max of the bounding box at the specified index

    Args:
        groundtruth (np.ndarray): Groundtruth of image
        index (int): Index of bounding box
        buffer (int, optional): Buffer to add to bounding box. Defaults to 20.

    Returns:
        np.ndarray: Bounding box - np.array([x_min, y_min, x_max, y_max])
    """

    # Extract the specified slice
    slice_groundtruth = groundtruth[index]

    # Get the indices of the non-zero elements
    y_indices, x_indices = np.where(slice_groundtruth > 0)

    # If the slice is empty (no non-zero elements), return None
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    # Calculate min and max coordinates
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Add buffer to the coordinates
    H, W = slice_groundtruth.shape
    x_min = max(0, x_min - buffer)
    x_max = min(W, x_max + buffer)
    y_min = max(0, y_min - buffer)
    y_max = min(H, y_max + buffer)

    # Create bounding box
    bbox = np.array([x_min, y_min, x_max, y_max])

    return bbox


def load_model(model_type:str, model_location: str, model_version: str, axis: int, device: str):
    """ Loads the model

    Args:
        model_type (str): Model type ('vit_b')
        model_location (str): Location of model
        model_version (str): Version of model
        device (str): Device to run model on

    Returns:
        sam_model_registry.SAMModel: Model
    """
    # Load model
    checkpoint = join(model_location, model_version, f"{axis}", "sam_model_best.pth")
    model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return model


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    
    # Headings: image_id, image_location, mask_location, dataset_group, random_state
    metadata = pd.read_csv(args.metadata)

    # Get all test image ids
    test_ids = metadata[metadata['dataset_group'] == 'test']['image_id'].values.tolist()
    
    embedding, image, groundtruth = load_embedding_and_groundtruth(image_location="data/medsam/embeddings", axis=args.axis, image_id=test_ids[0])

    print(embedding.shape, image.shape, groundtruth.shape)

    bounding_box = create_bounding_box(groundtruth, index=10)

    model = load_model(model_type=args.sam_model_type, model_location=args.model_location, model_version=args.model_version, axis=args.axis, device=args.device)

    sam_trans = ResizeLongestSide(model.image_encoder.img_size)
    box = sam_trans.apply_boxes(bounding_box, (groundtruth.shape[-2], groundtruth.shape[-1]))
    box_torch = torch.as_tensor(box, dtype=torch.float, device=args.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

    # Generate prompt embeddings
    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

        embedding = torch.tensor(embedding, dtype=torch.float32, device=args.device)
        groundtruth = torch.tensor(groundtruth, dtype=torch.float32, device=args.device)

        # Run the embeddings through the mask_decoder part of the model
        output = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        medsam_seg_prob = torch.sigmoid(output[0])

        print(len(output))
        for i in range(len(output)):
            print(output[i].shape)


if __name__ == "__main__":
    main()