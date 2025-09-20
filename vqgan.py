import torch
import torch.nn as nn
from copy import deepcopy
from encoder import Encoder, CondEncoder
from decoder import Decoder, CondDecoder
from codebook import Codebook, Online_Codebook


"""
Code adapted from https://github.com/dome272/VQGAN-pytorch/blob/main/vqgan.py
With augmentations for DAE option and compatibility with CVQGAN
"""
class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        if args.is_c:
            self.encoder = CondEncoder(args).to(device=args.device)
            self.decoder = CondDecoder(args).to(device=args.device)
            self.quant_conv = nn.Conv2d(args.c_latent_dim, args.c_latent_dim, 1).to(device=args.device)
            self.post_quant_conv = nn.Conv2d(args.c_latent_dim, args.c_latent_dim, 1).to(device=args.device)
        else:
            self.encoder = Encoder(args).to(device=args.device)
            self.decoder = Decoder(args).to(device=args.device)
            self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
            self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

        if args.use_DAE:
            assert not args.is_c, "DAE is not supported for conditional VQGAN"
            assert args.DAE_dropout > 0.0, "DAE dropout must be greater than 0"
            temp_args = deepcopy(args)
            temp_args.DAE_dropout = 0.0
            self.add_module("new_decoder", Decoder(temp_args).to(device=args.device))
        else:
            assert args.DAE_dropout == 0.0, "DAE dropout must be 0 if not using DAE"
        
        self.codebook = (Online_Codebook(args) if args.use_Online else Codebook(args)).to(device=args.device)

    def switch_to_new_decoder(self):
        if hasattr(self, "new_decoder"):
            self.decoder = self.new_decoder
            delattr(self, "new_decoder")
        else:
            print("Warning: Tried to switch to new_decoder, but it does not exist.")

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))








