import os
import torch
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from utils import get_data, load_vqgan, set_precision, set_all_seeds
from args import get_args, load_args, print_args


def change_random(vec, num_changes):
    for _ in range(num_changes):
        h1, w1 = np.random.randint(vec.shape[2]), np.random.randint(vec.shape[3])
        h2, w2 = h1, w1
        while h1 == h2 and w1 == w2:
            h2, w2 = np.random.randint(vec.shape[2]), np.random.randint(vec.shape[3])
        a = vec[0, :, h1, w1].clone()
        b = vec[0, :, h2, w2].clone()
        vec[0, :, h1, w1], vec[0, :, h2, w2] = b, a
    return vec


def mirror_image(img_np):
    """Mirror image horizontally and make it square by bicubic interpolation"""
    mirrored = np.concatenate([img_np, np.flip(img_np, axis=-1)], axis=-1)
    
    # Get current dimensions
    h, w = mirrored.shape
    
    # Make it square by resizing to the larger dimension using bicubic interpolation
    size = max(h, w)
    zoom_h = size / h
    zoom_w = size / w
    square_img = zoom(mirrored, (zoom_h, zoom_w), order=3)  # order=3 is bicubic
    
    return square_img


def save_side_by_side(original, altered, diff, save_path_base):
    import matplotlib.pyplot as plt

    # Simple figure with square subplots since images are now square
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(left=0.05, right=0.88, wspace=0.1)

    titles = ["Original", "Altered", "Difference"]
    images = [original, altered, diff]
    cmaps = ['viridis', 'viridis', 'seismic']
    vmins = [0, 0, -1]
    vmaxs = [1, 1, 1]

    ims = []
    for ax, img, title, cmap, vmin, vmax in zip(axs, images, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=18)
        ax.set_aspect('equal')
        ax.axis("off")
        ims.append(im)

    # Simple colorbar positioning
    cbar = plt.colorbar(ims[2], ax=axs, shrink=0.8, aspect=20)
    cbar.set_label("Change from Original", fontsize=16, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(f"{save_path_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path_base}.tiff", format='tiff', bbox_inches='tight', 
                dpi=300, pil_kwargs={'compression': 'tiff_lzw', 'optimize': True})
    plt.close()


class LatentPixelSwap:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)
        args = load_args(args)
        self.args = args
        self.device = args.device
        self.vqgan = load_vqgan(args).eval().to(self.device)

        self.eval_dir = os.path.join("../evals", args.model_name)
        self.latent_results_dir = os.path.join(self.eval_dir, "latent_results")
        os.makedirs(self.latent_results_dir, exist_ok=True)

        self.run()

    def run(self):
        args = self.args
        (_, _, test_dataloader), _, _ = get_data(args, use_val_split=True)

        # Skip to the nth sample (0-indexed, so subtract 1 from 1-indexed argument)
        test_iter = iter(test_dataloader)
        imgs, _ = next(test_iter)
        img = imgs.to(self.device)[args.sample_idx - 1].unsqueeze(0)

        with torch.no_grad():
            latent, _, _ = self.vqgan.encode(img)
            recon_orig = self.vqgan.decode(latent).clamp(0, 1).squeeze().cpu().numpy()

            latent_mod = change_random(latent.clone(), args.num_swaps)
            recon_mod = self.vqgan.decode(latent_mod).clamp(0, 1).squeeze().cpu().numpy()

        # Mirror for visualization only
        recon_orig_vis = mirror_image(recon_orig)
        recon_mod_vis = mirror_image(recon_mod)
        diff_vis = recon_mod_vis - recon_orig_vis

        save_path = os.path.join(self.latent_results_dir, f"pixel_swap_{args.num_swaps}_sample_{args.sample_idx}")
        save_side_by_side(recon_orig_vis, recon_mod_vis, diff_vis, save_path)


if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Latent Pixel Swap Arguments")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_swaps', type=int, default=64)
    parser.add_argument('--sample_idx', type=int, default=5, help='Sample index to use (1-indexed)')
    cli_args = parser.parse_known_args()[0]
    args.num_swaps = cli_args.num_swaps
    args.sample_idx = cli_args.sample_idx

    LatentPixelSwap(args)