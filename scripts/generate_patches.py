from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='/data/fcy/Datasets/Underwater/LSUI/test', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='/data/fcy/Datasets/Underwater/LSUI_AVG_PATCH/test',type=str, help='Directory for image patches')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=300, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=128, type=int, help='Number of CPU Cores')
parser.add_argument('--stride', default=128, type=int, help='Stride for sliding window')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
STRIDE = args.stride
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores

noisy_patchDir = os.path.join(tar, 'input')
clean_patchDir = os.path.join(tar, 'gt')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(noisy_patchDir)
os.makedirs(clean_patchDir)

#get sorted folders

noisy_files = natsorted(glob(os.path.join(src, 'input', '*.png')) + glob(os.path.join(src, 'input', '*.jpg')))
clean_files = natsorted(glob(os.path.join(src, 'gt', '*.png')) + glob(os.path.join(src, 'gt', '*.jpg')))

def patch_files(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)

    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]

        cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(i+1,j+1)), noisy_patch)
        cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(i+1,j+1)), clean_patch)


def sliding_window(image):
    window_size = (PS, PS)
    stride = STRIDE
    height, width = image.shape[:2]
    patches = []
    x_start_positions = [x for x in range(0, width - window_size[1] + 1, stride)]
    y_start_positions = [y for y in range(0, height - window_size[0] + 1, stride)]
    x_start_positions.append(width - window_size[1])
    y_start_positions.append(height - window_size[0])

    for y in y_start_positions:
        for x in x_start_positions:
            patch = image[y:y + window_size[1], x:x + window_size[0]]
            patches.append(patch)

    return patches

def avg_patch_files(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)

    # padding to minimal size can be divided by PS
    H = noisy_img.shape[0]
    W = noisy_img.shape[1]

    H_pad = int(np.ceil(H / 128) * PS)
    W_pad = int(np.ceil(W / 128) * PS)

    # sliding window
    noisy_patches = sliding_window(noisy_img)
    clean_patches = sliding_window(clean_img)

    # save patches
    for j in range(len(noisy_patches)):
        cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(i+1,j+1)), noisy_patches[j])
        cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(i+1,j+1)), clean_patches[j])

def pad_image_to_patch_size(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)
    # if image size is not divisible by 8, pad it minimal to be divisible by 8
    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    pad_H = int(np.ceil(H / 8) * 8)
    pad_W = int(np.ceil(W / 8) * 8)
    if pad_W != 0:
        noisy_img = cv2.copyMakeBorder(noisy_img, 0, 0, 0, pad_W - W, cv2.BORDER_REFLECT)
        clean_img = cv2.copyMakeBorder(clean_img, 0, 0, 0, pad_W - W, cv2.BORDER_REFLECT)
    if pad_H != 0:
        noisy_img = cv2.copyMakeBorder(noisy_img, 0, pad_H - H, 0, 0, cv2.BORDER_REFLECT)
        clean_img = cv2.copyMakeBorder(clean_img, 0, pad_H - H, 0, 0, cv2.BORDER_REFLECT)
    cv2.imwrite(os.path.join(noisy_patchDir, '{}.png'.format(i+1)), noisy_img)
    cv2.imwrite(os.path.join(clean_patchDir, '{}.png'.format(i+1)), clean_img)

Parallel(n_jobs=NUM_CORES)(delayed(pad_image_to_patch_size)(i) for i in tqdm(range(len(noisy_files))))
