# python3.7
"""Computes the semantic directions regarding a specific image region."""

import os
import argparse
import numpy as np
from tqdm import tqdm

from coordinate import COORDINATES
from coordinate import get_mask
from utils.image_utils import save_image


def parse_args():
    """Parses arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('jaco_path', type=str,
                        help='Path to jacobian matrix.')
    parser.add_argument('--region', type=str, default='eyes',
                        help='The region to be used to compute jacobian.')
    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory to save the results. If not specified,'
                             'the results will be saved to '
                            '`work_dirs/{TASK_SPECIFIC}/` by default')
    parser.add_argument('--job', type=str, default='directions',
                        help='Name for the job (default: directions)')
    parser.add_argument('--name', type=str, default='resefa',
                        help='Name of help save the results.')
    parser.add_argument('--data_name', type=str, default='ffhq',
                        help='Name of the dataset.')
    parser.add_argument('--full_rank', action='store_true',
                        help='Whether or not to full rank background'
                             ' (default: False).')
    parser.add_argument('--tao', type=float, default=1e-3,
                        help='Coefficient to the identity matrix '
                             '(default: 1e-3).')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    assert os.path.exists(args.jaco_path)
    Jacobians = np.load(args.jaco_path)
    image_size = Jacobians.shape[2]
    w_dim = Jacobians.shape[-1]
    coord_dict = COORDINATES[args.data_name]
    assert args.region in coord_dict, \
        f'{args.region} coordinate is not defined in ' \
        f'COORDINATE_{args.data_name}. Please define this region first!'
    coords = coord_dict[args.region]
    mask = get_mask(image_size, coordinate=coords)
    foreground_ind = np.where(mask == 1)
    background_ind = np.where((1 - mask) == 1)
    temp_dir = f'./work_dirs/{args.job}/{args.data_name}/{args.region}'
    save_dir = args.save_dir or temp_dir
    os.makedirs(save_dir, exist_ok=True)
    for ind in tqdm(range(Jacobians.shape[0])):
        Jacobian = Jacobians[ind]
        if len(Jacobian.shape) == 4:  # [H, W, 1, latent_dim]
            Jaco_fore = Jacobian[foreground_ind[0], foreground_ind[1], 0]
            Jaco_back = Jacobian[background_ind[0], background_ind[1], 0]
        elif len(Jacobian.shape) == 5:  # [channel, H, W, 1, latent_dim]
            Jaco_fore = Jacobian[:, foreground_ind[0], foreground_ind[1], 0]
            Jaco_back = Jacobian[:, background_ind[0], background_ind[1], 0]
        else:
            raise ValueError('Shape of the Jacobian is not correct!')
        Jaco_fore = np.reshape(Jaco_fore, [-1, w_dim])
        Jaco_back = np.reshape(Jaco_back, [-1, w_dim])
        coef_f = 1 / Jaco_fore.shape[0]
        coef_b = 1 / Jaco_back.shape[0]
        M_fore = coef_f * Jaco_fore.T.dot(Jaco_fore)
        M_back = coef_b * Jaco_back.T.dot(Jaco_back)
        if args.full_rank:
            # J = J_b^TJ_b
            # J = (J + tao * trace(J) * I)
            print('Using full rank')
            coef = args.tao * np.trace(M_back)
            M_back = M_back + coef * np.identity(M_back.shape[0])
        # inv(B) * A = lambda x
        temp = np.linalg.inv(M_back).dot(M_fore)
        eig_val, eig_vec = np.linalg.eig(temp)
        eig_val = np.real(eig_val)
        eig_vec = np.real(eig_vec)
        directions = eig_vec.T
        directions = directions[np.argsort(-eig_val)]
        save_name = f'{save_dir}/image_{ind:02d}_region_{args.region}' \
                    f'_name_{args.name}'
        np.save(f'{save_name}.npy', directions)
        mask_i = np.tile(mask[:, :, np.newaxis], [1, 1, 3]) * 255
        save_image(f'{save_name}_mask.png', mask_i.astype(np.uint8))


if __name__ == '__main__':
    main()
