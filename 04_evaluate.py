"""
Run the model on the test set and save the results
"""
import argparse
import os
from typing import Tuple
import pandas as pd
from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling import Sam
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


def create_bounding_box(groundtruth: np.ndarray, index: int, image_size: tuple, device:str, buffer: int = 1) -> np.ndarray:
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

    # Check if bounding box is valid
    sam_trans = ResizeLongestSide(image_size)
    box = sam_trans.apply_boxes(bbox, (groundtruth.shape[-2], groundtruth.shape[-1]))
    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

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


def save_output(output: torch.Tensor, medsam_seg_prob: torch.Tensor, image_id: str, axis: int, save_location: str) -> None:
    """ Saves the output of the model

    Args:
        output (torch.Tensor): Output of model
        medsam_seg_prob (torch.Tensor): Segmentation probability
        image_id (str): Image id
        axis (int): Axis to perform segmentation on
        save_location (str): Location to save segmentation results and metadata
    """
    save_location_folder = join(save_location, f"axis_{axis}")
    if not os.path.exists(save_location_folder):
        os.makedirs(save_location_folder)

    masks = output[0].cpu().numpy()
    mask_probabilities = output[1].cpu().numpy()
    medsam_seg_prob = medsam_seg_prob.cpu().numpy()

    # Save output
    np.savez_compressed(
        join(save_location_folder, f"{image_id}.npz"),
        masks=masks,
        mask_probabilities=mask_probabilities,
        mask_value_probabilities=medsam_seg_prob,
    )


def run_model(model: Sam, embedding: np.ndarray, groundtruth: np.ndarray, box: np.ndarray, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Runs the model on the embedding

    Args:
        model (Sam): Model
        embedding (np.ndarray): Embedding of image
        groundtruth (np.ndarray): Groundtruth of image
        box (np.ndarray): Bounding box
        device (str): Device to run model on

    Returns:
        torch.Tensor: Output of model
        torch.Tensor: Segmentation probability
    """
    # Generate prompt embeddings
    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=box,
            masks=None,
        )

        embedding = torch.tensor(embedding, dtype=torch.float32, device=device)
        groundtruth = torch.tensor(groundtruth, dtype=torch.float32, device=device)

        # Run the embeddings through the mask_decoder part of the model
        output = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        medsam_seg_prob = torch.sigmoid(output[0])

    return output, medsam_seg_prob


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    
    # Headings: image_id, image_location, mask_location, dataset_group, random_state
    metadata = pd.read_csv(args.metadata)

    # Get all test image ids
    test_ids = metadata[metadata['dataset_group'] == 'test']['image_id'].values.tolist()

    model = load_model(model_type=args.sam_model_type, model_location=args.model_location, model_version=args.model_version, axis=args.axis, device=args.device)


    for image_id in tqdm(test_ids):
        # Load image, embedding and groundtruth
        embedding, image, groundtruth = load_embedding_and_groundtruth(image_location="data/medsam/embeddings", axis=args.axis, image_id=image_id)

        # Create bounding box
        bboxes = []
        for index in range(groundtruth.shape[0]):
            bounding_box = create_bounding_box(groundtruth, index=index, device=args.device, image_size=model.image_encoder.img_size)
            bboxes.append(bounding_box)
        bboxes = np.array(bboxes)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float, device=args.device)

        # Run model
        output, medsam_seg_prob = run_model(model, embedding, groundtruth, bboxes, args.device)

        # Save output
        save_output(output, medsam_seg_prob, image_id, args.axis, args.save_location)
        

if __name__ == "__main__":
    main()