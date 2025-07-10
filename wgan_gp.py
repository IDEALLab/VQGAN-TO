# Code adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
# Paper: https://arxiv.org/abs/1704.00028

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from copy import deepcopy

from utils import load_vqgan
from args import load_args


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        temp_args = deepcopy(args)
        temp_args.is_gan = False
        vq_args = load_args(temp_args)
        self.img_shape = (vq_args.latent_dim, 16, 16)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim + args.c_input_dim, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, int(np.prod(self.img_shape))),
        )

    def forward(self, z, c):
        x = torch.cat((z, c), dim=1)
        latents = self.model(x)
        return latents.view(x.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        temp_args = deepcopy(args)
        temp_args.is_gan = False
        vq_args = load_args(temp_args)
        self.img_shape = (vq_args.latent_dim, 16, 16)
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)) + args.c_input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, z, c):
        z_flat = z.reshape(z.size(0), -1)
        input_vec = torch.cat((z_flat, c), dim=1)
        validity = self.model(input_vec)
        return validity


def compute_gradient_penalty(D, real_samples, fake_samples, c, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, c)
    fake = torch.ones(real_samples.shape[0], 1, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class VQGANLatentWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        temp_args = deepcopy(args)
        temp_args.is_gan = False
        self.vqgan = load_vqgan(load_args(temp_args)).eval()
        for param in self.vqgan.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, x):
        quant_z = self.vqgan.encoder(x)
        return quant_z

    @torch.no_grad()
    def decode(self, E):
        q = self.vqgan.quant_conv(E)
        z, _, _ = self.vqgan.codebook(q)
        return self.vqgan.decode(z)


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
# parser.add_argument('--train_samples', type=int, default=999999)
# parser.add_argument('--dataset_path', type=str, default='../data/gamma_4579_half.npy')
# parser.add_argument('--conditions_path', type=str, default='../data/inp_paras_4579.npy')
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
# parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
# parser.add_argument("--gan_sample_interval", type=int, default=400, help="interval betwen image samples")