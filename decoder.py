import torch.nn as nn
import torch.nn.utils as nn_utils
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        in_channels = args.decoder_channels[0]
        resolution = args.decoder_start_resolution
        layers = [self._conv(args.spectral_norm, args.latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(args.decoder_channels)):
            out_channels = args.decoder_channels[i]
            for j in range(args.decoder_num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in args.decoder_attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))

        # EDIT FROM ORIGINAL: Required so the decoder outputs logits...converted back to image space using sigmoid later
        if not args.use_focal_loss:
            layers.append(Swish())
        
        layers.append(self._conv(args.spectral_norm, in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def _conv(self, use_spectral_norm, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if use_spectral_norm:
            return nn_utils.spectral_norm(conv)
        return conv

    def forward(self, x):
        return self.model(x)

