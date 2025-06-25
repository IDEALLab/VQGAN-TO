# Code adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
# Paper: https://arxiv.org/abs/1704.00028

import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from utils import get_data, set_precision, set_all_seeds, str2bool

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
parser.add_argument("--cond_dim", type=int, default=3, help="dimensionality of the conditions")
parser.add_argument("--image_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--image_channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--is_c', type=str2bool, default=False)
parser.add_argument('--is_t', type=str2bool, default=False)
parser.add_argument('--train_samples', type=int, default=999999)
parser.add_argument('--dataset_path', type=str, default='../data/gamma_4579_half.npy')
parser.add_argument('--conditions_path', type=str, default='../data/inp_paras_4579.npy')

args = parser.parse_args()
print(args)

img_shape = (args.image_channels, args.image_size, args.image_size)

cuda = True if torch.cuda.is_available() else False

set_precision()
set_all_seeds(args.seed)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim + args.cond_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, c):
        input = torch.cat((z, c), dim=1)
        img = self.model(input)
        return img.view(img.size(0), *img_shape)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + args.cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img, c):
        img_flat = img.view(img.size(0), -1)
        input_vec = torch.cat((img_flat, c), dim=1)
        validity = self.model(input_vec)
        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(args.image_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples, c):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, c)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
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


# ----------
#  Training
# ----------
(dataloader, val_dataloader, _), means, stds = get_data(args, use_val_split=True)

batches_done = 0  
for epoch in range(args.n_epochs):  
    for i, (imgs, c) in enumerate(dataloader):  

        # Configure input  
        real_imgs = Variable(imgs.type(Tensor))  
        # c = Variable(Tensor(np.random.uniform(0, 1, (imgs.shape[0], args.cond_dim))))

        # ---------------------  
        #  Train Discriminator  
        # ---------------------  

        optimizer_D.zero_grad()  

        # Sample noise as generator input  
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))  

        # Generate a batch of images  
        fake_imgs = generator(z, c)  

        # Real images  
        real_validity = discriminator(real_imgs, c)  
        # Fake images  
        fake_validity = discriminator(fake_imgs, c)  
        # Gradient penalty  
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, c)  
        # Adversarial loss  
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty  

        d_loss.backward()  
        optimizer_D.step()  

        optimizer_G.zero_grad()  

        # Train the generator every n_critic steps  
        if i % args.n_critic == 0:  

            # -----------------  
            #  Train Generator  
            # -----------------  

            # Generate a batch of images  
            fake_imgs = generator(z, c)  
            # Loss measures generator's ability to fool the discriminator  
            # Train on fake images  
            fake_validity = discriminator(fake_imgs, c)  
            g_loss = -torch.mean(fake_validity)  

            g_loss.backward()  
            optimizer_G.step()  

            print(  
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"  
                % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())  
            )  

            if batches_done % args.sample_interval == 0:  
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)  

            batches_done += args.n_critic