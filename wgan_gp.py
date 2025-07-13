# Code adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
# Paper: https://arxiv.org/abs/1704.00028

import torch
import torch.nn as nn
import torch.autograd as autograd
from copy import deepcopy

from utils import load_vqgan
from args import load_args


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
        
        if args.gan_use_cvq:
            input_dim = args.latent_dim + cvq_args.c_latent_dim * cvq_args.c_fmap_dim ** 2
        else:
            input_dim = args.latent_dim + args.c_input_dim

        self.init_size = 2  # → 2×2 → 4×4 → 8×8 → 16×16

        high = self.latent_dim * 8
        mid = self.latent_dim * 4
        low = self.latent_dim * 2

        self.fc = nn.Linear(input_dim, high * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(high),
            nn.ConvTranspose2d(high, mid, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(mid, low, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(low, self.latent_dim, 4, stride=2, padding=1)
        )

    def forward(self, z, c):
        x = torch.cat((z, c), dim=1)
        out = self.fc(x).view(x.size(0), -1, self.init_size, self.init_size)
        return self.conv_blocks(out)


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
        base = self.latent_dim // 4
        mid = base * 2

        self.condition_proj = nn.Linear(cvq_args.c_latent_dim * cvq_args.c_fmap_dim ** 2 if args.gan_use_cvq else args.c_input_dim, 16 * 16)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=7, stride=2, padding=3),  # larger kernel
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, mid, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(mid * 4 * 4, 1)
        )

    def forward(self, z, c):
        c = self.condition_proj(c).view(z.size(0), 1, 16, 16)
        return self.conv(torch.cat([z, c], dim=1))


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


class VQGANLatentWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        temp_args = deepcopy(args)
        temp_args.is_gan = False
        self.vqgan = load_vqgan(load_args(temp_args)).eval()
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


# # # Original arguments from the WGAN-GP implementation
# parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
# parser.add_argument("--learning_rate", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
# parser.add_argument("--c_input_dim", type=int, default=3, help="dimensionality of the conditions")
# parser.add_argument("--image_size", type=int, default=256, help="size of each image dimension")
# parser.add_argument("--image_channels", type=int, default=1, help="number of image channels")
# parser.add_argument('--seed', type=int, default=1)
# parser.add_argument('--is_c', type=str2bool, default=False)
# parser.add_argument('--is_t', type=str2bool, default=False)
# parser.add_argument('--dataset_path', type=str, default='../data/gamma_4579_half.npy')
# parser.add_argument('--conditions_path', type=str, default='../data/inp_paras_4579.npy')
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
# parser.add_argument("--gan_sample_interval", type=int, default=400, help="interval betwen image samples")