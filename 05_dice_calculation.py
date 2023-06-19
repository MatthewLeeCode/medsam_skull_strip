from typing import Tuple
import radvis as rv
import numpy as np
from os.path import join
from utils.SurfaceDice import compute_dice_coefficient
import nibabel as nib
import os
import pandas as pd
from tqdm import tqdm


def align_segmentation_voxels(axis_0_seg: np.ndarray, axis_1_seg: np.ndarray, axis_2_seg: np.ndarray) -> np.ndarray:
    """
    MEDSAM is applied to all 3 axis. This function aligns each segmentation map to the same voxel space so that the same voxels overlap.

    Args:
        axis_0_seg: segmentation map from axis 0
        axis_1_seg: segmentation map from axis 1
        axis_2_seg: segmentation map from axis 2

    Returns:
        the aligned segmentation maps stacked along the channel dimension
    """

    # Ensure all segmentations have the same shape
    if axis_0_seg.shape != axis_1_seg.shape or axis_0_seg.shape != axis_2_seg.shape:
        raise ValueError("All input segmentation maps must have the same shape.")

    # Align the segmentation maps to the same voxel space
    axis_0_seg_aligned = np.transpose(axis_0_seg, (2, 0, 1))  # sagittal to axial
    axis_1_seg_aligned = np.transpose(axis_1_seg, (2, 1, 0))  # coronal to axial
    axis_2_seg_aligned = axis_2_seg  # axial is already aligned

    # Stack the segmentation maps along the channel dimension
    stacked_segmentations = np.stack([axis_0_seg_aligned, axis_1_seg_aligned, axis_2_seg_aligned], axis=-1)

    return stacked_segmentations


def add_padding(image: np.ndarray, min_index: int, original_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Adds the padding back after cropping by using the min and max index

    Args:
        image: the image to add padding to
        min_index: the minimum index of the image
        max_index: the maximum index of the image
        original_shape: the original shape of the mask before cropping

    Returns:
        the image with padding added
    """
    image_shape = image.shape

    # Calculate the difference between the original shape and the cropped shape
    diff = np.array(original_shape) - np.array(image_shape)
    pad_amount = (min_index, diff[0] - min_index)

    # Pad the image
    padded_image = np.pad(image, ((int(pad_amount[0]), pad_amount[1]), (0, 0), (0, 0)), mode="constant")

    return padded_image


def get_dice_scores(image_id: str, results_location: str, image_location: str, suffix: str = "") -> dict:
    """
    Calculates the dice scores for each axis
    """
    segmentation_probs = []
    segmentation_masks = []
    groundtruth_masks = []
    results = {} 
    for axis in [0,1,2]:
        # Load the result
        result = np.load(join(results_location, f"axis_{axis}", f"{image_id}{suffix}.npz"))
        image_data = np.load(join(image_location, f"axis_{axis}", f"{image_id}{suffix}.npz"))

        # Image data. Contains 'imgs', 'img_embeddings', 'gts'
        image = image_data['imgs']
        embedding = image_data['img_embeddings']
        groundtruth = image_data['gts']

        # Contains 'masks', 'mask_probabilities', 'mask_value_probabilities'
        masks = result['masks']
        segmentation = result['segmentation']
        mask_probabilities = result['mask_probabilities']
        mask_value_probabilities = result['mask_value_probabilities']

        # Get the first mask [slice, mask, height, width]
        mask = mask_value_probabilities[:, :, :]

        r_image = rv.from_numpy(image[:, :, :, 0])
        slicer_medsam = rv.RadSlicer(r_image, title="MEDSAM")
        slicer_medsam.add_mask(segmentation, color="green", alpha=0.5)

        slicer_gt = rv.RadSlicer(r_image, title="Groundtruth")
        slicer_gt.add_mask(groundtruth, color="red", alpha=0.5)

        # Add a mask which is the XOR of the segmentation and groundtruth
        slicer = rv.RadSlicer(r_image, title="Difference")

        xor_mask = np.logical_xor(segmentation, groundtruth)
        # Calculates which pixels of the segmentation are wrong
        xor_seg_mask = np.logical_and(xor_mask, segmentation)

        # Calculates which pixels of the groundtruth are wrong
        xor_gt_mask = np.logical_and(xor_mask, groundtruth)

        results[f"axis_{axis}_diff"] = np.sum(xor_mask)
        results[f"axis_{axis}_gt_diff"] = np.sum(xor_gt_mask)
        results[f"axis_{axis}_seg_diff"] = np.sum(xor_seg_mask)

        mask_probabilities = mask_probabilities.reshape((mask_probabilities.shape[0], 1, 1))
        mask_value_probabilities = mask_probabilities * mask_value_probabilities
        segmentation_masks.append(segmentation)
        segmentation_probs.append(mask_value_probabilities)
        groundtruth_masks.append(groundtruth)

        dice = compute_dice_coefficient(segmentation, groundtruth)
        results["axis_{}".format(axis)] = dice

    # Loads the min and max index for each axis
    minmax_0 = np.load(join(image_location, f"axis_0", f"{image_id}{suffix}_minmax.npz"))
    minmax_1 = np.load(join(image_location, f"axis_1", f"{image_id}{suffix}_minmax.npz"))
    minmax_2 = np.load(join(image_location, f"axis_2", f"{image_id}{suffix}_minmax.npz"))
    min_0, max_0 = minmax_0['min_ind'], minmax_0['max_ind']
    min_1, max_1 = minmax_1['min_ind'], minmax_1['max_ind']
    min_2, max_2 = minmax_2['min_ind'], minmax_2['max_ind']

    # Add padding to the segmentation masks
    original_shape = (256, 256, 256)
    seg_0, seg_1, seg_2 = segmentation_masks
    probs_0, probs_1, probs_2 = segmentation_probs

    seg_0 = add_padding(seg_0, min_0, original_shape=original_shape)
    seg_1 = add_padding(seg_1, min_1, original_shape=original_shape)
    seg_2 = add_padding(seg_2, min_2, original_shape=original_shape)

    probs_0 = add_padding(probs_0, min_0, original_shape=original_shape)
    probs_1 = add_padding(probs_1, min_1, original_shape=original_shape)
    probs_2 = add_padding(probs_2, min_2, original_shape=original_shape)

    r_image.image_data = add_padding(r_image.image_data, min_2, original_shape=original_shape)
    groundtruth = add_padding(groundtruth, min_2, original_shape=original_shape)

    # Align the segmentation maps to the same voxel space
    segmentation_ensemble = align_segmentation_voxels(seg_0, seg_1, seg_2)
    probabilities_ensemble = align_segmentation_voxels(probs_0, probs_1, probs_2)

    # Probabilities ensemble has 3 channels for each pixel with every channel having a value from 0 - 1
    # We want to add up the channels and then convert the values from 0 - 3 to 0 - 1
    probabilities_ensemble = np.sum(probabilities_ensemble, axis=-1) / 3

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        probs_threshold = probabilities_ensemble > threshold
        dice = compute_dice_coefficient(probs_threshold, groundtruth)
        results[f"threshold_{threshold}"] = dice
        nib.save(nib.Nifti1Image(probs_threshold.astype(np.uint8), np.eye(4)), join(results_location, f"thresholds", f"{image_id}_threshold_{threshold}.nii.gz"))
    
    # Save the segmentation_ensemble channels as a nifti file
    nib.save(nib.Nifti1Image(segmentation_ensemble[:, :, :, 0].astype(np.uint8), np.eye(4)), join(results_location, f"axis_0", f"{image_id}_segmentation.nii.gz"))
    nib.save(nib.Nifti1Image(segmentation_ensemble[:, :, :, 1].astype(np.uint8), np.eye(4)), join(results_location, f"axis_1", f"{image_id}_segmentation.nii.gz"))
    nib.save(nib.Nifti1Image(segmentation_ensemble[:, :, :, 2].astype(np.uint8), np.eye(4)), join(results_location, f"axis_2", f"{image_id}_segmentation.nii.gz"))
    return results 

def main():
    version = "0.0.2"
    results_location = f"data/medsam/{version}/results"
    image_location = f"data/medsam/"

    image_filenames = os.listdir(join(results_location, "axis_0"))
    image_ids = list(set([filename.split("_")[0] for filename in image_filenames]))
    suffix = "_mni"

    results = []
    for image_id in tqdm(image_ids):
        row = get_dice_scores(image_id, results_location, image_location, suffix)
        results.append(row)

    results = pd.DataFrame(results)
    results.to_csv(join(results_location, "results.csv"))
    # Calculate the amount of voxels which have all 3 segmentation channels as '1'
    # This is the amount of voxels which are segmented correctly by all 3 axes
    '''
    all_vote_voxels = np.sum(segmentation_ensemble, axis=-1) == 3
    two_vote_voxels = np.sum(segmentation_ensemble, axis=-1) >= 2
    one_vote_voxels = np.sum(segmentation_ensemble, axis=-1) >= 1

    all_vote_dice = compute_dice_coefficient(all_vote_voxels, groundtruth)
    two_vote_dice = compute_dice_coefficient(two_vote_voxels, groundtruth)
    one_vote_dice = compute_dice_coefficient(one_vote_voxels, groundtruth)

    print(f"Dice coefficient for all vote voxels: {all_vote_dice}")
    print(f"Dice coefficient for two vote voxels: {two_vote_dice}")
    print(f"Dice coefficient for one vote voxels: {one_vote_dice}")

    #seg_1_num = rv.from_numpy(seg_1)
    ensemble_slicer = rv.RadSlicer(r_image, axis=0, title="Ensemble")
    ensemble_slicer.add_mask(all_vote_voxels, color="green", alpha=0.5)
    ensemble_slicer.add_mask(two_vote_voxels, color="blue", alpha=0.5)
    ensemble_slicer.add_mask(one_vote_voxels, color="red", alpha=0.5)
    #ensemble_slicer.add_mask(seg_0, color="green", alpha=0.5)
    #ensemble_slicer.add_mask(seg_1, color="blue", alpha=0.5)
    #ensemble_slicer.add_mask(seg_2, color="red", alpha=0.5)
    ensemble_slicer.display()
    '''

if __name__ == "__main__":
    main()