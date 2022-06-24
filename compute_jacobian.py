# python3.7
"""Functions to compute Jacobian based on pre-trained GAN generator.

Support StyleGAN2 or StyleGAN3
"""

import os
import argparse
import warnings
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from models import build_model
from utils.image_utils import save_image
from utils.image_utils import postprocess_image
from utils.custom_utils import to_numpy


warnings.filterwarnings(action='ignore', category=UserWarning)


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('General options.')
    group.add_argument('weight_path', type=str,
                       help='Weight path to the pre-trained model.')
    group.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save the results. If not specified, '
                            'the results will be saved to '
                            '`work_dirs/{TASK_SPECIFIC}/` by default.')
    group.add_argument('--job', type=str, default='jacobians',
                       help='Name for the job (default: jacobians)')
    group.add_argument('--seed', type=int, default=4,
                       help='Seed for sampling. (default: 4)')
    group.add_argument('--nums', type=int, default=5,
                       help='Number of samples to synthesized. (default: 5)')
    group.add_argument('--img_size', type=int, default=1024,
                       help='Size of the synthesized images. (default: 1024)')
    group.add_argument('--w_dim', type=int, default=512,
                       help='Dimension of the latent w. (default: 512)')
    group.add_argument('--save_jpg', action='store_false',
                       help='Whether to save the images used to compute '
                            'jacobians. (default: True)')
    group.add_argument('-d', '--data_name', type=str, default='ffhq',
                       help='Name of the datasets. (default: ffhq)')
    group.add_argument('--latent_path', type=str, default='',
                       help='Path to the given latent codes. (default: None)')

    group = parser.add_argument_group('StyleGAN2')
    group.add_argument('--stylegan2', action='store_true',
                       help='Whether or not using StyleGAN2. (default: False)')
    group.add_argument('--scale_stylegan2', type=float, default=1.0,
                       help='Scale for the number of channel fro stylegan2.')
    group.add_argument('--randomize_noise', type=str, default='const',
                       help='Noise type when computing. (const or random)')

    group = parser.add_argument_group('StyleGAN3')
    group.add_argument('--stylegan3', action='store_true',
                       help='Whether or not using StyleGAN3. (default: False)')
    group.add_argument('--cfg', type=str, default='T',
                       help='Config of the stylegan3 (T/R).')
    group.add_argument('--scale_stylegan3r', type=float, default=2.0,
                       help='Scale for the number of channel for stylegan3 R.')
    group.add_argument('--scale_stylegan3t', type=float, default=1.0,
                       help='Scale for the number of channel for stylegan3 T.')
    group.add_argument('--tx', type=float, default=0,
                       help='Translate X-coordinate. (default: 0.0)')
    group.add_argument('--ty', type=float, default=0,
                       help='Translate Y-coordinate. (default: 0.0)')
    group.add_argument('--rotate', type=float, default=0,
                       help='Rotation angle in degrees. (default: 0)')

    group = parser.add_argument_group('Jacobians')
    group.add_argument('--b', type=float, default=1e-3,
                       help='Constant when computing jacobians fast.')
    group.add_argument('--batch_size', type=int, default=4,
                       help='Batch size. (default: 4)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    # Parse model configuration.
    assert (args.stylegan2 and not args.stylegan3) or \
           (not args.stylegan2 and args.stylegan3)
    job_disc = ''
    if args.stylegan2:
        config = dict(model_type='StyleGAN2Generator',
                      resolution=args.img_size,
                      w_dim=args.w_dim,
                      fmaps_base=int(args.scale_stylegan2 * (32 << 10)),
                      fmaps_max=512,)
        job_disc += 'stylegan2'
    else:
        if args.stylegan3 and args.cfg == 'R':
            config = dict(model_type='StyleGAN3Generator',
                          resolution=args.img_size,
                          w_dim=args.w_dim,
                          fmaps_base=int(args.scale_stylegan3r * (32 << 10)),
                          fmaps_max=1024,
                          use_radial_filter=True,)
            job_disc += 'stylegan3r'
        elif args.stylegan3 and args.cfg == 'T':
            config = dict(model_type='StyleGAN3Generator',
                          resolution=args.img_size,
                          w_dim=args.w_dim,
                          fmaps_base=int(args.scale_stylegan3t * (32 << 10)),
                          fmaps_max=512,
                          use_radial_filter=False,
                          kernel_size=3,)
            job_disc += 'stylegan3t'
        else:
            raise TypeError(f'StyleGAN3 config type error, need `R/T`,'
                            f' but got {args.cfg}')
    job_name = f'seed_{args.seed}_num_{args.nums}_{job_disc}'
    temp_dir = f'work_dirs/{args.job}/{args.data_name}/{job_name}'
    save_dir = args.save_dir or temp_dir
    os.makedirs(save_dir, exist_ok=True)
    if args.save_jpg:
        os.makedirs(f'{save_dir}/images', exist_ok=True)

    print('Building generator...')
    generator = build_model(**config)
    checkpoint_path = args.weight_path
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['models']
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.eval().cuda()
    print('Finish loading checkpoint.')

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if os.path.exists(args.latent_path):
        latent_zs = np.load(args.latent_path)
        latent_zs = latent_zs[:args.nums]
    else:
        latent_zs = np.random.randn(args.nums, generator.z_dim)
    latent_zs = torch.from_numpy(latent_zs.astype(np.float32))
    latent_zs = latent_zs.cuda()
    with torch.no_grad():
        latent_ws = generator.mapping(latent_zs)['w']
    print(f'Shape of the latent w: {latent_ws.shape}')

    def syn2jaco(w):
        """Wrap the synthesized function to compute the Jacobian easily.

        Basically, this function defines a generator that takes the input
        from the W space and then synthesizes an image. If the image is
        larger than 256, it will be resized to 256 to save the time and
        storage.

        Args:
            w: latent code from the W space

        Returns:
            An image with the size of [1, 256, 256]
        """
        wp = w.unsqueeze(1).repeat((1, generator.num_layers, 1))
        image = generator.synthesis(wp)['image']
        if image.shape[-1] > 256:
            scale = 256 / image.shape[-1]
            image = F.interpolate(image, scale_factor=scale)
            image = torch.sum(image, dim=1)
        return image

    jacobians = []
    for idx in tqdm(range(latent_zs.shape[0])):
        latent_w = latent_ws[idx:idx+1]
        jac_i = jacobian(func=syn2jaco,
                            inputs=latent_w,
                            create_graph=False,
                            strict=False)
        jacobians.append(jac_i)
        if args.save_jpg:
            wp = latent_w.unsqueeze(1).repeat((1, generator.num_layers, 1))
            syn_outputs = generator.synthesis(wp)['image']
            syn_outputs = to_numpy(syn_outputs)
            images = postprocess_image(syn_outputs)
            save_path = f'{save_dir}/images/{idx:06d}.jpg'
            save_image(save_path, images[0])
    jacobians = torch.cat(jacobians, dim=0)
    jacobians = to_numpy(jacobians)
    print(f'shape of the jacobian: {jacobians.shape}')
    latent_ws = to_numpy(latent_ws)
    np.save(f'{save_dir}/latent_codes.npy', latent_ws)
    np.save(f'{save_dir}/jacobians_w.npy', jacobians)
    print(f'Finish computing {args.nums} jacobians.')


if __name__ == '__main__':
    main()
