# Required Libraries
import argparse
import numpy as np
import SimpleITK as sitk
import os
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

def create_arg_parser() -> argparse.ArgumentParser:
    """
    Creates and returns an ArgumentParser to read arguments from command line.
    """
    parser = argparse.ArgumentParser(description='Preprocess non-CT images')
    parser.add_argument('-i', '--nii_path', type=str, default='data/FLARE22Train/images', help='Path to the nii images')
    parser.add_argument('-gt', '--gt_path', type=str, default='data/FLARE22Train/labels', help='Path to the ground truth')
    parser.add_argument('-o', '--npz_path', type=str, default='data/Npz_files', help='Path to save the npz files')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--modality', type=str, default='CT', help='Modality')
    parser.add_argument('--anatomy', type=str, default='Abd-Gallbladder', help='Anatomy')
    parser.add_argument('--img_name_suffix', type=str, default='_0000.nii.gz', help='Image name suffix')
    parser.add_argument('--label_id', type=int, default=9, help='Label id')
    parser.add_argument('--prefix', type=str, default='CT_Abd-Gallbladder_', help='Prefix')
    parser.add_argument('--model_type', type=str, default='vit_b', help='Model type')
    parser.add_argument('--checkpoint', type=str, default='work_dir/SAM/sam_vit_b_01ec64.pth', help='Checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    return parser


def filter_and_shuffle_names(gt_path: str, nii_path: str, img_name_suffix: str, seed: int = 2023) -> tuple:
    """
    Filters file names and returns shuffled train and test file names.
    """
    names = sorted(os.listdir(gt_path))
    names = [name for name in names if not os.path.exists(os.path.join(nii_path, name.split('.nii.gz')[0] + img_name_suffix))]
    np.random.seed(seed)
    np.random.shuffle(names)
    train_names = sorted(names[:int(len(names) * 0.8)])
    test_names = sorted(names[int(len(names) * 0.8):])
    return train_names, test_names


def preprocess_nonct(gt_path: str, nii_path: str, gt_name: str, image_name: str, label_id: int, image_size: int, sam_model: SamPredictor, device: str):
    """
    Preprocess nonCT images: perform image intensity normalization, resize, calculate embeddings.
    """
    gt_sitk = sitk.ReadImage(os.path.join(gt_path, gt_name))
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    gt_data = np.uint8(gt_data == label_id)

    # Images, ground truths and embeddings
    imgs = []
    gts =  []
    img_embeddings = []
    if np.sum(gt_data) > 1000:
        img_sitk = sitk.ReadImage(os.path.join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        image_data_pre = preprocess_image_data(image_data)

        z_index, _, _ = np.where(gt_data > 0)
        z_min, z_max = np.min(z_index), np.max(z_index)

        for i in range(z_min, z_max):
            gt_slice_i, img_slice_i = preprocess_slices(gt_data, image_data_pre, i, image_size)

            # Calculate image embeddings
            if sam_model is not None:
                embedding = compute_image_embedding(sam_model, device, img_slice_i)
                img_embeddings.append(embedding)
    
    return imgs, gts, img_embeddings if sam_model is not None else imgs, gts


def preprocess_image_data(image_data: np.ndarray) -> np.ndarray:
    """
    Preprocess image data: normalize intensity, clip to range, scale to 255, and convert to uint8.
    """
    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
    image_data_pre[image_data == 0] = 0
    return np.uint8(image_data_pre)


def preprocess_slices(gt_data: np.ndarray, image_data_pre: np.ndarray, i: int, image_size: int) -> tuple:
    """
    Preprocess slices: resize, convert to three channels.
    """
    gt_slice_i = transform.resize(gt_data[i, :, :], (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
    img_slice_i = transform.resize(image_data_pre[i, :, :], (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
    img_slice_i = np.uint8(np.repeat(img_slice_i[:, :, None], 3, axis=-1))
    return gt_slice_i, img_slice_i


def compute_image_embedding(sam_model: SamPredictor, device: str, img_slice_i: np.ndarray) -> np.ndarray:
    """
    Compute image embeddings using SAM model.
    """
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    resize_img = sam_transform.apply_image(img_slice_i)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])

    with torch.no_grad():
        return sam_model.image_encoder(input_image).cpu().numpy()[0]


def save_to_npz(save_path: str, prefix: str, gt_name: str, imgs: list, gts: list, img_embeddings: list = None) -> None:
    """
    Save images, ground truths, and embeddings to compressed npz file.
    """
    if len(imgs) > 1:
        imgs = np.stack(imgs, axis=0)
        gts = np.stack(gts, axis=0)

        if img_embeddings is not None:
            img_embeddings = np.stack(img_embeddings, axis=0)
            np.savez_compressed(os.path.join(save_path, prefix + '_' + gt_name.split('.nii.gz')[0] + '.npz'), imgs=imgs, gts=gts, img_embeddings=img_embeddings)
        else:
            np.savez_compressed(os.path.join(save_path, prefix + '_' + gt_name.split('.nii.gz')[0] + '.npz'), imgs=imgs, gts=gts)

        # Save an example image for sanity check
        save_example_image(save_path, imgs, gts)


def save_example_image(save_path: str, imgs: np.ndarray, gts: np.ndarray) -> None:
    """
    Save an example image for sanity check.
    """
    idx = np.random.randint(0, imgs.shape[0])
    img_idx = imgs[idx, :, :, :]
    gt_idx = gts[idx, :, :]
    bd = segmentation.find_boundaries(gt_idx, mode='inner')
    img_idx[bd, :] = [255, 0, 0]
    io.imsave(save_path + '.png', img_idx, check_contrast=False)


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the script.
    """
    prefix = args.modality + '_' + args.anatomy
    names = get_names(args, prefix)

    np.random.seed(args.seed)
    np.random.shuffle(names)
    train_names = sorted(names[:int(len(names)*0.8)])
    test_names = sorted(names[int(len(names)*0.8):])

    # Prepare the save path
    save_path_tr = os.path.join(args.npz_path, prefix, 'train')
    save_path_ts = os.path.join(args.npz_path, prefix, 'test')
    os.makedirs(save_path_tr, exist_ok=True)
    os.makedirs(save_path_ts, exist_ok=True)

    # Set up the model
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

    # Process and save training data
    for name in tqdm(train_names):
        process_and_save(name, args, sam_model, save_path_tr)

    # Process and save testing data (without calculating embeddings)
    for name in tqdm(test_names):
        process_and_save(name, args, None, save_path_ts)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
