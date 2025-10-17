import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
from copy import deepcopy
import torch.autograd as autograd
import torch.nn.functional as F

from utils import load_vqgan
from args import load_args

"""
# Code adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
# WGAN-GP Paper: https://arxiv.org/abs/1704.00028
# We mostly follow the architecture guidelines from the WGAN-GP paper, referred to as the "CIFAR-10 ResNet architecture"
"""

def sn(layer, use_spectral):
    """Apply spectral norm if enabled."""
    return spectral_norm(layer) if use_spectral else layer


class ResBlockG(nn.Module):
    """
    Generator pre-activation ResBlock.
    Two 3x3 convs; if upsample=True, perform nearest-neighbor upsampling BEFORE the 2nd conv.
    BN only in G (per paper).
    """
    def __init__(self, in_ch: int, out_ch: int, upsample: bool = False):
        super().__init__()
        self.upsample = upsample
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.learnable_skip = (in_ch != out_ch) or upsample
        if self.learnable_skip:
            self.conv_sc = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.conv2(out)

        skip = x
        if self.upsample:
            skip = F.interpolate(skip, scale_factor=2, mode="nearest")
        if self.learnable_skip:
            skip = self.conv_sc(skip)
        return out + skip


class ResBlockD(nn.Module):
    """
    Critic pre-activation ResBlock.
    No normalization. Mean/avg pooling AFTER the 2nd conv when downsample=True.
    """
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False, use_spectral: bool = False):
        super().__init__()
        self.downsample = downsample
        self.conv1 = sn(nn.Conv2d(in_ch,  out_ch, 3, padding=1), use_spectral)
        self.conv2 = sn(nn.Conv2d(out_ch, out_ch, 3, padding=1), use_spectral)
        self.learnable_skip = (in_ch != out_ch) or downsample
        if self.learnable_skip:
            self.conv_sc = sn(nn.Conv2d(in_ch, out_ch, 1, bias=False), use_spectral)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x, inplace=True)
        out = self.conv1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)

        skip = x
        if self.downsample:
            skip = F.avg_pool2d(skip, 2)
        if self.learnable_skip:
            skip = self.conv_sc(skip)
        return out + skip


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()

        temp_args = deepcopy(args)
        temp_args.is_gan = False
        vq_args = load_args(temp_args)
        if args.gan_use_cvq:
            temp_cvq_args = deepcopy(args)
            temp_cvq_args.is_c = True
            temp_cvq_args.is_gan = False
            cvq_args = load_args(temp_cvq_args)

        self.latent_dim = vq_args.latent_dim

        input_dim = args.latent_dim + args.c_transform_dim
        self.condition_proj = nn.Linear(
            cvq_args.c_latent_dim * cvq_args.c_fmap_dim ** 2 if args.gan_use_cvq else args.c_input_dim,
            args.c_transform_dim
        )

        assert vq_args.image_size >= 32, "Image size should be at least 32x32"
        self.init_size = vq_args.image_size//32  # 8×8 -> 16×16
        high = self.latent_dim * 4
        mid = self.latent_dim * 2

        self.fc = nn.Linear(input_dim, high * self.init_size ** 2)

        # Generator ResNet stack (pre-activation, NN upsampling inside ResBlock)
        self.block_up = ResBlockG(high, mid, upsample=True)               # 8x8 -> 16x16
        self.block_refine = ResBlockG(mid, self.latent_dim, upsample=False)
        self.bn_out = nn.BatchNorm2d(self.latent_dim)
        self.conv_out = sn(nn.Conv2d(self.latent_dim, self.latent_dim, 3, stride=1, padding=1), args.use_spectral)

    def forward(self, z, c):
        c = self.condition_proj(c)
        x = self.fc(torch.cat([z, c], dim=1)).view(z.size(0), -1, self.init_size, self.init_size)
        x = self.block_up(x)
        x = self.block_refine(x)
        x = F.relu(self.bn_out(x), inplace=True)
        return self.conv_out(x)


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()

        temp_args = deepcopy(args)
        temp_args.is_gan = False
        vq_args = load_args(temp_args)
        self.latent_dim = vq_args.latent_dim
        self.gan_use_cvq = args.gan_use_cvq
        if args.gan_use_cvq:
            temp_cvq_args = deepcopy(args)
            temp_cvq_args.is_c = True
            temp_cvq_args.is_gan = False
            cvq_args = load_args(temp_cvq_args)

        in_channels = self.latent_dim + 1
        base = self.latent_dim  # Stronger discriminator base

        self.c_transform_dim = vq_args.decoder_start_resolution ** 2
        self.condition_proj = nn.Linear(
            cvq_args.c_latent_dim * cvq_args.c_fmap_dim ** 2 if args.gan_use_cvq else args.c_input_dim,
            self.c_transform_dim
        )

        # Discriminator ResNet stack (no norm): two Down blocks + two plain blocks
        self.b1 = ResBlockD(in_channels, base, downsample=True, use_spectral=args.use_spectral)   # 16x16 -> 8x8
        self.b2 = ResBlockD(base,        base, downsample=True, use_spectral=args.use_spectral)   # 8x8  -> 4x4
        self.b3 = ResBlockD(base,        base, downsample=False, use_spectral=args.use_spectral)
        self.b4 = ResBlockD(base,        base, downsample=False, use_spectral=args.use_spectral)
        self.lin = sn(nn.Linear(base, 1), args.use_spectral)

    def forward(self, z, c):
        size = int(np.sqrt(self.c_transform_dim))
        c = self.condition_proj(c).view(-1, 1, size, size)
        x = torch.cat([z, c], dim=1)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = F.relu(x, inplace=True)
        x = x.mean(dim=(2, 3))  # global mean pool
        return self.lin(x)


# VQGAN wrapper for WGAN-GP training on the latent space --> WGAN-AE and WGAN-VQ
class VQGANLatentWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        temp_args = deepcopy(args)
        temp_args.is_gan = False
        vq_args = load_args(temp_args)
        self.vq_args = vq_args
        self.vqgan = load_vqgan(vq_args).eval()
        for param in self.vqgan.parameters():
            param.requires_grad = False

        if args.gan_use_cvq:
            temp_cvq_args = deepcopy(args)
            temp_cvq_args.is_c = True
            temp_cvq_args.is_gan = False
            cvq_args = load_args(temp_cvq_args)
            self.cvqgan = load_vqgan(cvq_args).eval()
            for param in self.cvqgan.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def encode(self, x):
        return self.vqgan.encoder(x)

    @torch.no_grad()
    def decode(self, E):
        q = self.vqgan.quant_conv(E)
        z, _, _ = self.vqgan.codebook(q)
        return self.vqgan.decode(z)

    @torch.no_grad()
    def c_encode(self, c):
        return self.cvqgan.encoder(c)

    @torch.no_grad()
    def c_decode(self, E):
        q = self.cvqgan.quant_conv(E)
        c, _, _ = self.cvqgan.codebook(q)
        return self.cvqgan.decode(c)


def compute_gradient_penalty(D, real_samples, fake_samples, c, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, c)
    fake = torch.ones_like(d_interpolates, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
