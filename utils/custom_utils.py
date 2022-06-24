# python3.7
"""Utility functions for image editing."""

import numpy as np
import cv2
import torch


__all__ = ['to_numpy', 'linear_interpolate', 'make_transform',
           'get_ind', 'mask2image']


def to_numpy(data):
    """Converts the input data to `numpy.ndarray`."""
    if isinstance(data, (int, float)):
        return np.array(data)
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    raise TypeError(f'Not supported data type `{type(data)}` for '
                    f'converting to `numpy.ndarray`!')


def linear_interpolate(latent_code,
                       boundary,
                       layer_index=None,
                       start_distance=-10.0,
                       end_distance=10.0,
                       steps=21):
    """Interpolate between the latent code and boundary."""
    assert (len(latent_code.shape) == 3 and len(boundary.shape) == 3 and
            latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
            latent_code.shape[1] == boundary.shape[1])
    linspace = np.linspace(start_distance, end_distance, steps)
    linspace = linspace.reshape([-1, 1, 1]).astype(np.float32)
    inter_code = linspace * boundary
    is_manipulatable = np.zeros(inter_code.shape, dtype=bool)
    is_manipulatable[:, layer_index, :] = True
    mani_code = np.where(is_manipulatable, latent_code+inter_code, latent_code)
    return mani_code


def make_transform(tx, ty, angle):
    """Transform the input feature maps with given
    coordinates and rotation angle.

    cos(theta) -sin(theta) tx
    sin(theta)  cos(theta) ty
        0          0        1

    """
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = tx
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = ty
    return m


def get_ind(seg_mask, label):
    """Get the index of the masked and unmasked region."""
    mask = np.where(seg_mask == label,
                    np.ones_like(seg_mask),
                    np.zeros_like(seg_mask))
    f_ind = np.where(mask == 1)
    b_ind = np.where((1 - mask) == 1)
    return f_ind, b_ind, mask


def mask2image(image, mask, r=3, g=255, b=118):
    """Show the mask on the given image."""
    assert image.shape[0] == image.shape[1]
    r_c = np.ones([256, 256, 1]) * r
    g_c = np.ones([256, 256, 1]) * g
    b_c = np.ones([256, 256, 1]) * b
    img1 = np.concatenate([r_c, g_c, b_c], axis=2).astype(np.uint8)
    mask = np.expand_dims(mask, axis=2).astype(np.uint8)
    img1 = img1 * mask
    image = cv2.addWeighted(image, 0.4, img1, 0.6, 0)
    mask_i = np.tile(mask, [1, 1, 3]) * 255
    return image, mask_i
