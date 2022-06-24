# python3.7
"""Manipulates synthesized or real images with existing boundary.

Support StyleGAN2 and StyleGAN3.
"""

import os.path
import argparse
import numpy as np
from tqdm import tqdm
import torch

from models import build_model
from utils.visualizers.html_visualizer import HtmlVisualizer
from utils.image_utils import save_image
from utils.parsing_utils import parse_index
from utils.image_utils import postprocess_image
from utils.custom_utils import to_numpy, linear_interpolate
from utils.custom_utils import make_transform


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('General options.')
    group.add_argument('weight_path', type=str,
                       help='Weight path to the pre-trained model.')
    group.add_argument('boundary_path', type=str,
                       help='Path to the attribute vectors.')
    group.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save the results. If not specified, '
                            'the results will be saved to '
                            '`work_dirs/{TASK_SPECIFIC}/` by default.')
    group.add_argument('--job', type=str, default='manipulations',
                       help='Name for the job. (default: manipulations)')
    group.add_argument('--seed', type=int, default=4,
                       help='Seed for sampling. (default: 4)')
    group.add_argument('--nums', type=int, default=10,
                       help='Number of samples to synthesized. (default: 10)')
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
    group.add_argument('--name', type=str, default='resefa',
                       help='Name of help save the results.')

    group = parser.add_argument_group('StyleGAN2')
    group.add_argument('--stylegan2', action='store_true',
                       help='Whether or not using StyleGAN2. (default: False)')
    group.add_argument('--scale_stylegan2', type=float, default=1.0,
                       help='Scale for the number of channel fro stylegan2.')
    group.add_argument('--randomize_noise', type=str, default='const',
                       help='Noise type when editing. (const or random)')

    group = parser.add_argument_group('StyleGAN3')
    group.add_argument('--stylegan3', action='store_true',
                       help='Whether or not using StyleGAN3. (default: False)')
    group.add_argument('--cfg', type=str, default='T',
                       help='Config of the stylegan3 (T/R)')
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

    group = parser.add_argument_group('Manipulation')
    group.add_argument('--mani_layers', type=str, default='4,5,6,7',
                       help='The layers will be manipulated.'
                            '(default: 4,5,6,7). For the eyebrow and lipstick,'
                            'using [8-11] layers instead.')
    group.add_argument('--step', type=int, default=7,
                       help='Number of manipulation steps. (default: 7)')
    group.add_argument('--start', type=int, default=0,
                       help='The start index of the manipulation directions.')
    group.add_argument('--end', type=int, default=1,
                       help='The end index of the manipulation directions.')
    group.add_argument('--start_distance', type=float, default=-10.0,
                       help='Start distance for manipulation. (default: -10.0)')
    group.add_argument('--end_distance', type=float, default=10.0,
                       help='End distance for manipulation. (default: 10.0)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    # Parse model configuration.
    assert (args.stylegan2 and not args.stylegan3) or \
           (not args.stylegan2 and args.stylegan3)
    checkpoint_path = args.weight_path
    boundary_path = args.boundary_path
    assert os.path.exists(checkpoint_path)
    assert os.path.exists(boundary_path)
    boundary_name = os.path.splitext(os.path.basename(boundary_path))[0]
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
    job_name = f'seed_{args.seed}_num_{args.nums}_{job_disc}_{boundary_name}'
    os.makedirs(f'{save_dir}/{job_name}', exist_ok=True)

    print('Building generator...')
    generator = build_model(**config)
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['models']
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.eval().cuda()
    print('Finish loading checkpoint.')
    if args.stylegan3 and hasattr(generator.synthesis, 'early_layer'):
        m = make_transform(args.tx, args.ty, args.rotate)
        m = np.linalg.inv(m)
        generator.synthesis.early_layer.transform.copy_(torch.from_numpy(m))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if os.path.exists(args.latent_path):
        print(f'Load latent codes from {args.latent_path}')
        latent_zs = np.load(args.latent_path)
        latent_zs = latent_zs[:args.nums]
    else:
        print('Sampling latent code randomly')
        latent_zs = np.random.randn(args.nums, generator.z_dim)
    latent_zs = torch.from_numpy(latent_zs.astype(np.float32))
    latent_zs = latent_zs.cuda()
    num_images = latent_zs.shape[0]
    wp = []
    for idx in range(0, num_images, args.batch_size):
        latent_z = latent_zs[idx:idx+args.batch_size]
        latent_w_ = generator.mapping(latent_z, None)['wp']
        wp.append(latent_w_)
    wp = torch.cat(wp, dim=0)
    trunc_psi = args.trunc_psi
    trunc_layers = args.trunc_layers
    if trunc_psi < 1.0 and trunc_layers > 0:
        w_avg = generator.w_avg
        w_avg = w_avg.reshape(1, -1, generator.w_dim)[:, :trunc_layers]
        wp[:, :trunc_layers] = w_avg.lerp(wp[:, :trunc_layers], trunc_psi)
    print(f'Shape of the latent ws: {wp.shape}')
    image_list = []
    for i in range(num_images):
        image_list.append(f'{i:06d}')

    print('Loading boundary.')
    directions = np.load(boundary_path)
    layer_index = parse_index(args.mani_layers)
    if not layer_index:
        layer_index = list(range(generator.num_layers - 1))
    print(f'Manipulating on layers `{layer_index}`.')

    vis_size = None if args.vis_size == 0 else args.vis_size
    delta_num = args.end - args.start
    visualizer = HtmlVisualizer(num_rows=num_images * delta_num,
                                num_cols=args.step + 2,
                                image_size=vis_size)
    visualizer.set_headers(
        ['Name', 'Origin'] +
        [f'Step {i:02d}' for i in range(1, args.step + 1)]
    )
    # Manipulate images.
    print('Start manipulation.')
    for row in tqdm(range(num_images)):
        latent_w = wp[row:row+1]
        images_ori = generator.synthesis(latent_w)['image']
        images_ori = postprocess_image(to_numpy(images_ori))
        if args.save_jpg:
            save_image(f'{save_dir}/{job_name}/{row:06d}_orin.jpg',
                       images_ori[0])
        for num_direc in range(args.start, args.end):
            html_row = num_direc - args.start
            direction = directions[num_direc:num_direc+1]
            direction = np.tile(direction, [1, generator.num_layers, 1])
            visualizer.set_cell(row * delta_num + html_row, 0,
                                text=f'{image_list[row]}_{num_direc:03d}')
            visualizer.set_cell(row * delta_num + html_row, 1,
                                image=images_ori[0])
            mani_codes = linear_interpolate(latent_code=to_numpy(latent_w),
                                            boundary=direction,
                                            layer_index=layer_index,
                                            start_distance=args.start_distance,
                                            end_distance=args.end_distance,
                                            steps=args.step)
            mani_codes = torch.from_numpy(mani_codes.astype(np.float32)).cuda()
            for idx in range(0, mani_codes.shape[0], args.batch_size):
                codes_ = mani_codes[idx:idx+args.batch_size]
                images_ = generator.synthesis(codes_)['image']
                images_ = postprocess_image(to_numpy(images_))
                for i in range(images_.shape[0]):
                    visualizer.set_cell(row * delta_num + html_row, idx+i+2,
                                        image=images_[i])
                    if args.save_jpg:
                        save_image(f'{save_dir}/{job_name}/{row:06d}_ind_'
                                   f'{num_direc:06d}_mani_{idx+i:06d}.jpg',
                                   images_[i])
    # Save results.
    np.save(f'{save_dir}/{job_name}/latent_codes.npy', to_numpy(wp))
    visualizer.save(f'{save_dir}/{job_name}_{args.name}.html')


if __name__ == '__main__':
    main()
