import os
import sys
import re
import glob
import click
import pickle
from PIL import Image, ImageEnhance
from tqdm import tqdm
from tqdm.contrib import tzip

import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F

import dnnlib
from training import dataset as ds
from torch_utils import misc
from torch_utils import distributed as dist


def refdiff_sampler(
    net, latents, ref_images, mask_images, class_labels=None, randn_like=torch.randn_like,
    num_steps=256, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=40, S_min=0.05, S_max=50, S_noise=1.003, cfg_scale=0
):
    
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    ref_images = ref_images.to(torch.float64) 
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        
        denoised = net(x_hat, t_hat, latents, ref_images, mask_images, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        if i < num_steps - 1:
            denoised = net(x_next, t_next, latents, ref_images, mask_images, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
    return x_next

#----------------------------------------------------------------------------

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
    
#----------------------------------------------------------------------------

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


#----------------------------------------------------------------------------
# Sample saver

def save_samples(images, batch_seeds, batch_base_name, out_dir):
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for name, image_np in zip(batch_base_name, images_np):
        image_dir = out_dir
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, name)
        if image_np.shape[2] == 1:
            Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
        else:
            Image.fromarray(image_np, 'RGB').save(image_path)

#----------------------------------------------------------------------------

@click.command()
@click.option('--indir',                     help='Input directory for only-second-stage sampler', metavar='DIR',     type=str)
@click.option('--indir_ref',                     help='Input directory for only-second-stage sampler', metavar='DIR',     type=str)
@click.option('--indir_mask',                     help='Input directory for only-second-stage sampler', metavar='DIR',     type=str)
@click.option('--outdir',                    help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                     help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                   help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',        help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',   help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--sampler_stages',            help='Which stage to conduct sampler', metavar='first|second|both',      type=click.Choice(['first', 'second', 'both']), default='both')

# first stage sampler config
@click.option('--network_first',             help='Network pickle filename', metavar='PATH|URL',                      type=str)
@click.option('--num_steps_first',           help='Number of sampling steps for first stage', metavar='INT',          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min_first',           help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max_first',           help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho_first',                 help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--cfg_scale_first',           help='Scale of classifier-free guidance', metavar='FLOAT',               type=click.FloatRange(min=0), default=1, show_default=True)
@click.option('--S_churn', 'S_churn_first',  help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min_first',      help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max_first',      help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise_first',  help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

# second stage sampler config
@click.option('--network_second',            help='Network pickle filename', metavar='PATH|URL',                      type=str)
@click.option('--num_steps_second',          help='Number of sampling steps for second stage', metavar='INT',         type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--sigma_min_second',          help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max_second',          help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho_second',                help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--cfg_scale_second',          help='Scale of classifier-free guidance', metavar='FLOAT',               type=click.FloatRange(min=0), default=1, show_default=True)

def main(outdir, subdirs, seeds, class_idx, max_batch_size, sampler_stages, 
         network_first=None, network_second=None, indir=None, indir_ref=None,
         indir_mask=None, device=torch.device('cuda'), **sampler_kwargs):
    
    dist.init()
    img_list = glob.glob(os.path.join(indir, '*'))
    num_batches = ((len(img_list) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    
    if sampler_stages in ['second', 'both']:
   
        dist.print0(f'Loading second stage network from "{network_second}"...')
        
        assert network_second.endswith('pkl') or network_second.endswith('pt'), "Unknown format of the ckpt filename"
        if network_second.endswith('.pkl'):
            with dnnlib.util.open_url(network_second, verbose=(dist.get_rank() == 0)) as f:
                net_second = pickle.load(f)['ema'].to(device)
        elif network_second.endswith('.pt'):
            data = torch.load(network_second, map_location=torch.device('cpu'))
            net_second = data['ema'].eval().to(device)
        
        second_stage_sampler_kwargs = {
            k[:-7]: v for k, v in sampler_kwargs.items() if k.endswith('_second') and v is not None
        }
    
    if sampler_stages == 'second':
        # Preload for only-second-stage sampling.
        dist.print0(f'Preloading first stage samples from "{indir}"...')
        preload_images = []
        preload_base_name = []
        preload_ref_images = []
        preload_mask_images = []
        for batch_seeds in rank_batches:
            image_paths = [os.path.join(indir, img_list[int(seed)]) for seed in batch_seeds]
            batch_base_name = [os.path.basename(image_path) for image_path in image_paths]
            image_ref_paths = [os.path.join(indir_ref, name) for name in batch_base_name]
            image_mask_paths = [os.path.join(indir_mask, name) for name in batch_base_name]
            batch_images = [np.array(Image.open(path)) for path in image_paths]
            batch_images = [image[np.newaxis, :, :] if image.ndim == 2 else image.transpose(2, 0, 1) for image in batch_images]
            batch_images = np.concatenate([image[np.newaxis, ...] for image in batch_images], axis=0)
            preload_images.append(batch_images)
            preload_base_name.append(batch_base_name)

            batch_ref_images = [np.array(Image.open(path)) for path in image_ref_paths]
            batch_ref_images = [image[np.newaxis, :, :] if image.ndim == 2 else image.transpose(2, 0, 1) for image in batch_ref_images]
            batch_ref_images = np.concatenate([image[np.newaxis, ...] for image in batch_ref_images], axis=0)
            preload_ref_images.append(batch_ref_images)

            batch_mask_images = [np.array(Image.open(path)) for path in image_mask_paths]
            batch_mask_images = [image[np.newaxis, :, :] if image.ndim == 2 else image.transpose(2, 0, 1) for image in batch_mask_images]
            batch_mask_images = np.concatenate([image[np.newaxis, ...] for image in batch_mask_images], axis=0)
            preload_mask_images.append(batch_mask_images)
            
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    dist.print0('second stage config:', second_stage_sampler_kwargs)
    
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for i, batch_seeds in tzip(range(len(rank_batches)), rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
      if not os.path.exists(os.path.join(outdir, preload_base_name[i][0])):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        
        # Pick labels.
        class_labels = None
        label_dim = net_second.label_dim
        if label_dim:
            class_labels = torch.eye(label_dim, device=device)[batch_seeds % label_dim]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        rnd = StackedRandomGenerator(device, batch_seeds)
        
        if sampler_stages in ['first', 'both']:
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, net_first.img_channels, net_first.img_resolution, net_first.img_resolution], device=device)
            images = refdiff_sampler(net_first, latents, class_labels, randn_like=rnd.randn_like, **first_stage_sampler_kwargs)
        else:
            images = torch.tensor(preload_images[i], device=device, dtype=torch.float64) / 127.5 - 1
            ref_images = torch.tensor(preload_ref_images[i], device=device, dtype=torch.float64) / 127.5 - 1
            mask_images = torch.tensor(preload_mask_images[i], device=device, dtype=torch.float64) / 255.
        
        if sampler_stages == 'first':
            # Save outputs
            save_samples(images, batch_seeds, outdir)
            continue
        else:
            # Upsample for second stage generation.
            images = F.interpolate(images, 256)
            
        if sampler_stages in ['second', 'both']:
            # Second stage generation.
            images = refdiff_sampler(net_second, images, ref_images, mask_images, class_labels, randn_like=rnd.randn_like, **second_stage_sampler_kwargs)
           
            save_samples(images, batch_seeds, preload_base_name[i], outdir)
        
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
