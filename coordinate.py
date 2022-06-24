# python3.7
"""Utility functions to help define the region coordinates within an image."""

import os
from glob import glob
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from utils.parsing_utils import parse_index


def get_mask_by_coordinates(image_size, coordinate):
    """Get mask using the provided coordinates."""
    mask = np.zeros([image_size, image_size], dtype=np.float32)
    center_x, center_y = coordinate[0], coordinate[1]
    crop_x, crop_y = coordinate[2], coordinate[3]
    xx = center_x - crop_x // 2
    yy = center_y - crop_y // 2
    mask[xx:xx + crop_x, yy:yy + crop_y] = 1.
    return mask


def get_mask_by_segmentation(seg_mask, label):
    """Get the mask using the segmentation array and labels."""
    zeros = np.zeros_like(seg_mask)
    ones = np.ones_like(seg_mask)
    mask = np.where(seg_mask == label, ones, zeros)
    return mask


def get_mask(image_size, coordinate=None, seg_mask=None, labels='1'):
    """Get mask using either the coordinate or the segmentation array."""
    if coordinate is not None:
        print('Using coordinate to get mask!')
        mask = get_mask_by_coordinates(image_size, coordinate)
    else:
        print('Using segmentation to get the mask!')
        print(f'Using label {labels}')
        mask = np.zeros_like(seg_mask)
        for label_ in labels:
            mask += get_mask_by_segmentation(seg_mask, int(label_))
        mask = np.clip(mask, a_min=0, a_max=1)

    return mask


# For FFHQ [center_x, center_y, height, width]
# Those coordinates are suitable for both ffhq and metface.
COORDINATE_ffhq = {'left_eye': [120, 95, 20, 38],
                   'right_eye': [120, 159, 20, 38],
                   'eyes': [120, 128, 20, 115],
                   'nose': [142, 131, 40, 46],
                   'mouth': [184, 127, 30, 70],
                   'chin': [217, 130, 42, 110],
                   'eyebrow': [126, 105, 15, 118],
                   }


# For FFHQ unaligned
COORDINATE_ffhqu = {'eyesr2': [134, 116, 30, 115],
                    'eyesr3': [64, 128, 26, 115],
                    'eyest0': [70, 88, 30, 115],
                    'eyest3': [108, 142, 26, 115],
                    }

# [center_x, center_y, height, width]
COORDINATE_biggan = {'center0': [120, 120, 80, 80],
                     'center1': [120, 120, 130, 130],
                     'center2': [120, 120,  200, 200],
                     'left_side': [128, 64, 256, 128],
                     'top_side': [64, 128, 128, 256],
                     'head0': [89, 115, 49, 70],
                     'head1': [93, 110, 48, 70]}


COORDINATES = {'ffhq': COORDINATE_ffhq,
               'ffhqu': COORDINATE_ffhqu,
               'biggan': COORDINATE_biggan
               }


def parse_args():
    """Parses arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='',
                        help='The path to the image.')
    parser.add_argument('--mask_path', type=str, default='',
                        help='The path to the mask.')
    parser.add_argument('--save_dir', type=str, default='',
                        help='The path to the image.')
    parser.add_argument('--label', type=str, default=None,
                        help='The label number in the mask.')
    parser.add_argument('--data', type=str, default='ffhq',
                        help='The name of the dataset to test.')
    parser.add_argument('--num', type=int, default=0,
                        help='number of image to display.')
    parser.add_argument('--img_type', type=str, default='jpeg',
                        help='Format of the image.')

    return parser.parse_args()


def main():
    """Main function to show an image with masks"""
    args = parse_args()
    save_dir = args.save_dir or './temp_mask'
    os.makedirs(save_dir, exist_ok=True)
    images = sorted(glob(f'{args.image_path}/*.{args.img_type}'))[args.num:]
    label_files = sorted(glob(f'{args.mask_path}/*.npy'))[args.num:]
    COORDINATE = COORDINATES[args.data]
    for i, image in tqdm(enumerate(images)):
        img = cv2.imread(image)
        im_name = image.split('/')[-1].split('.')[0]
        if args.label is None:
            for name, coord in COORDINATE.items():
                if len(coord) == 0:
                    continue
                mask = np.zeros(img.shape, dtype=np.float32)
                center_x, center_y = coord[0], coord[1]
                crop_x, crop_y = coord[2], coord[3]
                xx = center_x - crop_x // 2
                yy = center_y - crop_y // 2
                mask[xx:xx + crop_x, yy:yy + crop_y, :] = 1.
                img_ = img * mask
                cv2.imwrite(f'{save_dir}/{im_name}_{name}.png', img_)
        else:
            print('Using segmentation to get the mask!')
            seg_mask = np.load(label_files[i])
            labels = parse_index(args.label)
            print(f'Using label {labels}')
            mask = np.zeros_like(seg_mask)
            for label_ in labels:
                mask += get_mask_by_segmentation(seg_mask, int(label_))
            mask = np.clip(mask, a_min=0, a_max=1)
            img_ = img * mask[:, :, np.newaxis]
            cv2.imwrite(f'{save_dir}/{im_name}_{args.label}.png', img_)


if __name__ == '__main__':
    main()
