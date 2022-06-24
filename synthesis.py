# python3.7
"""Script that synthesizes images with pre-trained models.

Support StyleGAN2 and StyleGAN3.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from models import build_model
from utils.visualizers.html_visualizer import HtmlVisualizer
from utils.image_utils import save_image, resize_image
from utils.image_utils import postprocess_image
from utils.custom_utils import to_numpy


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
    group.add_argument('--job', type=str, default='synthesize',
                       help='Name for the job. (default: synthesize)')
    group.add_argument('--seed', type=int, default=4,
                       help='Seed for sampling. (default: 4)')
    group.add_argument('--nums', type=int, default=100,
                       help='Number of samples to synthesized. (default: 100)')
    group.add_argument('--img_size', type=int, default=1024,
                       help='Size of the synthesized images. (default: 1024)')
    group.add_argument('--vis_size', type=int, default=256,
                       help='Size of the visualize images. (default: 256)')
    group.add_argument('--w_dim', type=int, default=512,
                       help='Dimension of the latent w. (default: 512)')
    group.add_argument('--batch_size', type=int, default=4,
                       help='Batch size. (default: 4)')
    group.add_argument('--save_jpg', action='store_true', default=False,
                       help='Whether to save raw image. (default: False)')
    group.add_argument('-d', '--data_name', type=str, default='ffhq',
                       help='Name of the datasets. (default: ffhq)')
    group.add_argument('--latent_path', type=str, default='',
                       help='Path to the given latent codes. (default: None)')
    group.add_argument('--trunc_psi', type=float, default=0.7,
                       help='Psi factor used for truncation. (default: 0.7)')
    group.add_argument('--trunc_layers', type=int, default=8,
                       help='Number of layers to perform truncation.'
                            ' (default: 8)')

    group = parser.add_argument_group('StyleGAN2')
    group.add_argument('--stylegan2', action='store_true',
                       help='Whether or not using StyleGAN2. (default: False)')
    group.add_argument('--scale_stylegan2', type=float, default=1.0,
                       help='Scale for the number of channel fro stylegan2.')
    group.add_argument('--randomize_noise', type=str, default='const',
                       help='Noise type when synthesizing. (const or random)')

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
                            f' but got {args.cfg} instead.')

    # Get work directory and job name.
    save_dir = args.save_dir or f'work_dirs/{args.job}/{args.data_name}'
    os.makedirs(save_dir, exist_ok=True)
    job_name = f'seed_{args.seed}_num_{args.nums}_{job_disc}'
    os.makedirs(f'{save_dir}/{job_name}', exist_ok=True)

    # Build generation and get synthesis kwargs.
    print('Building generator...')
    generator = build_model(**config)
    synthesis_kwargs = dict(trunc_psi=args.trunc_psi,
                            trunc_layers=args.trunc_layers,)
    # Load pre-trained weights.
    checkpoint_path = args.weight_path
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['models']
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.eval().cuda()
    print('Finish loading checkpoint.')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if os.path.exists(args.latent_path):
        latent_zs = np.load(args.latent_path)
        latent_zs = latent_zs[:args.nums]
    else:
        latent_zs = np.random.randn(args.nums, generator.z_dim)
    num_images = latent_zs.shape[0]
    latent_zs = torch.from_numpy(latent_zs.astype(np.float32))
    html = HtmlVisualizer(grid_size=num_images)
    print(f'Synthesizing {num_images} images ...')
    latent_ws = []
    for batch_idx in tqdm(range(0, num_images, args.batch_size)):
        latent_z = latent_zs[batch_idx:batch_idx + args.batch_size]
        latent_z = latent_z.cuda()
        with torch.no_grad():
            g_outputs = generator(latent_z, **synthesis_kwargs)
            g_image = to_numpy(g_outputs['image'])
            images = postprocess_image(g_image)
        for idx in range(images.shape[0]):
            sub_idx = batch_idx + idx
            img = images[idx]
            row_idx, col_idx = divmod(sub_idx, html.num_cols)
            image = resize_image(img, (args.vis_size, args.vis_size))
            html.set_cell(row_idx, col_idx, image=image,
                          text=f'Sample {sub_idx:06d}')
            if args.save_jpg:
                save_path = f'{save_dir}/{job_name}/{sub_idx:06d}.jpg'
                save_image(save_path, img)
        latent_ws.append(to_numpy(g_outputs['wp']))
    latent_ws = np.concatenate(latent_ws, axis=0)
    print(f'shape of the latent code: {latent_ws.shape}')
    np.save(f'{save_dir}/{job_name}/latent_codes.npy', latent_ws)
    html.save(f'{save_dir}/{job_name}.html')
    print(f'Finish synthesizing {num_images} samples.')


if __name__ == '__main__':
    main()
