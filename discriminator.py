"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""

import torch.nn as nn
import torch.nn.utils as nn_utils

class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [self._conv(args.spectral_disc, args.image_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                self._conv(args.spectral_disc, num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(self._conv(args.spectral_disc, num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

        # CUSTOM ADAPTER for CVQGAN
        self.cvqgan_adapter = nn.Sequential(
            nn.Linear(args.image_channels, args.image_size),
            nn.ReLU(),
            nn.Linear(args.image_size, args.image_channels*args.image_size**2),
            nn.ReLU(),
            nn.Unflatten(1, (args.image_channels, args.image_size, args.image_size))  # shape: (B, 3, 256, 256)
        )

    def _conv(self, use_spectral_norm, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if use_spectral_norm:
            return nn.utils.parametrizations.spectral_norm(conv)
        return conv

    def forward(self, x):
        if x.ndim == 2:
            x = self.cvqgan_adapter(x) 
        return self.model(x)
