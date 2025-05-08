# training_transformer.py

import os
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import get_data, str2bool #, plot_images


class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()

        self.train(args)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer

    def train(self, args):
        train_dataset, test_dataset, means, stds = get_data(args)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (imgs, c) in zip(pbar, train_dataset):
                    self.optim.zero_grad()
                    imgs = imgs.to(device=args.device)
                    logits, targets = self.model(imgs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
            log, sampled_imgs = self.model.log_images(imgs[0][None])
            vutils.save_image(sampled_imgs, os.path.join("results", f"transformer_{epoch}.jpg"), nrow=4)
            # plot_images(log)
            torch.save(self.model.state_dict(), os.path.join("checkpoints", f"transformer_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image_size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num_codebook_vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image_channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset_path', type=str, default='../data/gamma_4579_half.npy', help='Path to data (default: /data)') # New dataset path
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch_size', type=int, default=16, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc_start', type=int, default=0, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc_factor', type=float, default=1., help='')
    parser.add_argument('--rec_loss_factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual_loss_factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    # New arguments
    parser.add_argument('--conditions_path', type=str, default='../data/inp_paras_4579.npy', help='Path to conditions (default: ../data/inp_paras_4579.npy)')
    parser.add_argument('--problem_id', type=str, default='mto', help='Problem ID (default: mto)')
    parser.add_argument('--algo', type=str, default='vqgan', help='Algorithm name (default: vqgan)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--track', type=str2bool, default=True, help='track or not (default: True)')
    parser.add_argument('--save_model', type=str2bool, default=True, help='Save model checkpoint (default: True)')
    parser.add_argument('--sample_interval', type=int, default=215, help='Interval for saving sample images (default: 1000)')
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("Tr-%Y-%m-%d_%H-%M-%S"), help='Run name for this training session (default: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))')

    # Decoder-specific args
    parser.add_argument('--spectral_decoder', type=str2bool, default=False, help='Apply spectral normalization to Conv layers (default: False)')
    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[512, 256, 256, 128, 128], help='List of channel sizes for Decoder (default: [512, 256, 256, 128, 128])')
    parser.add_argument('--decoder_attn_resolutions', type=int, nargs='+', default=[16], help='Resolutions for attention in Decoder (default: [16])')
    parser.add_argument('--decoder_num_res_blocks', type=int, default=3, help='Number of residual blocks per stage in Decoder (default: 3)')
    parser.add_argument('--decoder_start_resolution', type=int, default=16, help='Starting resolution in Decoder (default: 16)')

    # Encoder-specific args
    parser.add_argument('--encoder_channels', type=int, nargs='+', default=[128, 128, 128, 256, 256, 512], help='List of channel sizes for Encoder (default: [128, 128, 128, 256, 256, 512])')
    parser.add_argument('--encoder_attn_resolutions', type=int, nargs='+', default=[16], help='Resolutions for attention in Encoder (default: [16])')
    parser.add_argument('--encoder_num_res_blocks', type=int, default=2, help='Number of residual blocks per stage in Encoder (default: 2)')
    parser.add_argument('--encoder_start_resolution', type=int, default=256, help='Starting resolution in Encoder (default: 256)')

    # Training-specific args
    parser.add_argument('--use_greyscale_lpips', type=str2bool, default=True, help='Use Greyscale LPIPS for perceptual loss (default: False)')
    parser.add_argument('--spectral_disc', type=str2bool, default=False, help='Apply spectral normalization to Conv layers of discriminator (default: False)')
    parser.add_argument('--use_DAE', type=str2bool, default=False, help='Use Decoupled Autoencoder for training (default: False)') # Not implemented
    parser.add_argument('--use_Online', type=str2bool, default=False, help='Use Online Clustered Codebook (default: False)') # Not implemented

    # Transformer-specific args
    parser.add_argument('--model_name', type=str, default="baseline", help='Saved model name for VQGAN Stage 1 (default: baseline)')
    parser.add_argument('--pkeep', type=float, default=1.0, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos_token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    args.checkpoint_path = os.path.join(r"../saves", args.model_name, "checkpoints", "vqgan.pth")

    train_transformer = TrainTransformer(args)


