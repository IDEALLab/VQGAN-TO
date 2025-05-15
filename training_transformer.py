# training_transformer.py

import os
import json
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
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

        self.log_losses = {'epochs': [], 'train_loss_avg': [], 'test_loss_avg': []}
        saves_dir = os.path.join(r"../saves", args.run_name)
        self.results_dir = os.path.join(saves_dir, "results")
        self.checkpoints_dir = os.path.join(saves_dir, "checkpoints")
        self.saves_dir = saves_dir

        self.prepare_training()

        # Save the arguments for later evaluation
        self.save_args(args)

        self.train(args)

    def save_args(self, args):
        """Save the training arguments for later use in evaluation"""
        
        os.makedirs(self.saves_dir, exist_ok=True)
        args_dict = vars(args)
        
        # Convert any non-serializable objects to strings
        for key, value in args_dict.items():
            if not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                args_dict[key] = str(value)
        
        # Save as JSON
        with open(os.path.join(self.saves_dir, "training_args.json"), 'w') as f:
            json.dump(args_dict, f, indent=4)
        
        print(f"Training arguments saved to {os.path.join(self.saves_dir, 'training_args.json')}")

    def prepare_training(self):
        os.makedirs(self.saves_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

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
        dataloader, test_dataloader, means, stds = get_data(args)

        for epoch in tqdm(range(args.epochs)):
            train_losses = []
            test_losses = []
            for imgs, c in dataloader:
                self.optim.zero_grad()
                imgs = imgs.to(device=args.device)
                c = c.to(device=args.device)
                logits, targets = self.model(imgs, c)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss.backward()
                self.optim.step()
                train_losses.append(loss.item())
            
            # Evaluate for test loss
            self.model.eval()
            with torch.no_grad():
                _, sampled_imgs = self.model.log_images(imgs[0][None], c[0][None])
                for imgs, c in test_dataloader:
                    imgs = imgs.to(device=args.device)
                    c = c.to(device=args.device)
                    logits, targets = self.model(imgs, c)
                    test_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    test_losses.append(test_loss.item())
            self.model.train()

            if args.track:
                batches_done = epoch * len(dataloader)
                # Calculate average losses for this epoch
                train_loss_avg = sum(train_losses) / len(train_losses)
                test_loss_avg = sum(test_losses) / len(test_losses)
                
                # Track epoch averages
                self.log_losses['epochs'].append(epoch)
                self.log_losses['train_loss_avg'].append(np.log(train_loss_avg))
                self.log_losses['test_loss_avg'].append(np.log(test_loss_avg))
                
                # Plot and save losses
                plt.figure(figsize=(10, 5))
                plt.plot(self.log_losses['epochs'], self.log_losses['train_loss_avg'], label='Train Log-Loss')
                plt.plot(self.log_losses['epochs'], self.log_losses['test_loss_avg'], label='Test Log-Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Average Log-Loss')
                plt.title('Transformer Train/Test Loss')
                plt.legend()
                plt.grid(True)
                
                loss_fname = os.path.join(self.results_dir, "log_loss.png")
                plt.savefig(loss_fname, format="png", dpi=300, bbox_inches="tight", transparent=True)
                plt.close()
                
                # Convert dictionary to arrays for proper numpy saving
                loss_data = np.array([
                    self.log_losses['epochs'],
                    self.log_losses['train_loss_avg'],
                    self.log_losses['test_loss_avg']
                ])
                
                vutils.save_image(sampled_imgs, os.path.join(self.results_dir, f"{batches_done}.png"), nrow=4)
                # Save the latest model state
                torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, f"transformer.pt"))
                # Save the loss data with a fixed name (overwriting previous versions)
                np.save(os.path.join(self.results_dir, "log_loss.npy"), loss_data)

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
    parser.add_argument('--learning_rate', type=float, default=5e-04, help='Learning rate (default: 2.25e-05)')
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
    parser.add_argument('--model_name', type=str, default="c6", help='Saved model name for VQGAN Stage 1 (default: baseline)')
    parser.add_argument('--c_model_name', type=str, default="cvq", help='Saved model name for CVQGAN (default: cvq)')
    parser.add_argument('--pkeep', type=float, default=1.0, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos_token', type=int, default=0, help='Start of Sentence token.')
    parser.add_argument('--t_is_c', type=str2bool, default=True, help='Use CVQGAN for prepended conditions (default: True)')

    # CVQGAN-specific args
    parser.add_argument('--is_c', type=str2bool, default=False, help='Train a CVQGAN (default: False)')
    parser.add_argument('--c_input_dim', type=int, default=3, help='Input dimension for CVQGAN (default: 3)')
    parser.add_argument('--c_hidden_dim', type=int, default=256, help='Hidden dimension for CVQGAN (default: 256)')
    parser.add_argument('--c_latent_dim', type=int, default=4, help='Latent (codebook vector) dimension for CVQGAN (default: 4)')
    parser.add_argument('--c_num_codebook_vectors', type=int, default=64, help='Number of codebook vectors for CVQGAN (default: 64)')
    parser.add_argument('--c_fmap_dim', type=int, default=4, help='Feature map dimension for CVQGAN (default: 4)')


    args = parser.parse_args()
    args.checkpoint_path = os.path.join(r"../saves", args.model_name, "checkpoints", "vqgan.pth")
    args.c_checkpoint_path = os.path.join(r"../saves", args.c_model_name, "checkpoints", "vqgan.pth")

    train_transformer = TrainTransformer(args)


