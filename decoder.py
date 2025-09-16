import torch.nn as nn
import torch.nn.utils as nn_utils
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish, LinearCombo


"""
Code for class Decoder adapted from https://github.com/dome272/VQGAN-pytorch/blob/main/decoder.py with augmentations for DAE option
"""
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        in_channels = args.decoder_channels[0]
        resolution = args.decoder_start_resolution
        self.dropout = nn.Dropout2d(p=args.DAE_dropout)
        layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
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
                # Apply dropout after each residual block for part 1 of DAE-VQGAN training
                if args.use_DAE: 
                    layers.append(self.dropout)
                
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Decoder for CVQGAN option
class CondDecoder(nn.Module):
    def __init__(self, args):
        super(CondDecoder, self).__init__()

        self.model = nn.Sequential(
            LinearCombo(args.c_latent_dim*args.c_fmap_dim**2, args.c_hidden_dim),
            LinearCombo(args.c_hidden_dim, args.c_hidden_dim),
            nn.Linear(args.c_hidden_dim, args.c_input_dim)
        )
    
    def forward(self, x):
        return self.model(x.contiguous().view(len(x), -1))