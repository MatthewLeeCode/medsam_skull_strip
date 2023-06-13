import radvis as rv
import numpy as np
from os.path import join
from utils.SurfaceDice import compute_dice_coefficient


def main():
    results_location = "data/medsam/embeddings/results"
    image_location = "data/medsam/embeddings"

    image_id = "00060516"
    suffix = "_mni"


    segmentation_masks = []
    groundtruth_masks = []
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

        slicer_med_diff = rv.RadSlicer(r_image, title="MEDSAM Difference")
        slicer_med_diff.add_mask(xor_seg_mask, color="green", alpha=0.5)

        slicer_gt_diff = rv.RadSlicer(r_image, title="Groundtruth Difference")
        slicer_gt_diff.add_mask(xor_gt_mask, color="red", alpha=0.5)

        # Animates the difference 
        #slicer_medsam.save_animation(f"medsam_{axis}.gif", fps=10)
        #slicer_gt.save_animation(f"gt_{axis}.gif", fps=10)
        #slicer_med_diff.save_animation(f"medsam_diff_{axis}.gif", fps=10)
        #slicer_gt_diff.save_animation(f"gt_diff_{axis}.gif", fps=10)

        segmentation_masks.append(segmentation)
        groundtruth_masks.append(groundtruth)

        dice = compute_dice_coefficient(segmentation, groundtruth)
        print(f"Dice coefficient for axis {axis}: {dice}")

    [print(x.shape) for x in segmentation_masks]


if __name__ == "__main__":
    main()