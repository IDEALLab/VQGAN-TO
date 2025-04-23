import os
import sys
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from discriminator import Discriminator
from lpips import LPIPS, GreyscaleLPIPS
from vqgan import VQGAN
from utils import get_data, weights_init, plot_data, print_args, set_precision, set_all_seeds


class TrainVQGAN:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)

        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = (GreyscaleLPIPS() if args.use_greyscale_lpips else LPIPS()).eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        self.log_losses = {'epochs': [], 'd_loss_avg': [], 'g_loss_avg': []}

        saves_dir = os.path.join(r"../saves", args.run_name)
        self.results_dir = os.path.join(saves_dir, "results")
        self.checkpoints_dir = os.path.join(saves_dir, "checkpoints")
        self.saves_dir = saves_dir

        if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
            self.vqgan = torch.compile(self.vqgan)
            self.discriminator = torch.compile(self.discriminator)

        self.prepare_training()
        
        # Save the arguments for later evaluation
        self.save_args(args)

        self.train(args)

    def save_args(self, args):
        """Save the training arguments for later use in evaluation"""
        import json
        
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

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    def prepare_training(self):
        os.makedirs(self.saves_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def train(self, args):
        scaler = GradScaler()
        dataloader, test_dataloader, means, stds = get_data(args)
        steps_per_epoch = len(dataloader)
        
        for epoch in tqdm(range(args.epochs)):
            epoch_d_losses = []
            epoch_g_losses = []
            for i, (imgs, c) in enumerate(dataloader):
                imgs = imgs.to(device=args.device, non_blocking=True)
                with autocast(device_type=args.device, dtype=torch.float16):
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    scaler.scale(vq_loss).backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    scaler.scale(gan_loss).backward()

                    scaler.step(self.opt_vq)
                    scaler.step(self.opt_disc)
                    scaler.update()

                    epoch_d_losses.append(gan_loss.item())
                    epoch_g_losses.append(vq_loss.item())

                if args.track:
                    batches_done = epoch * len(dataloader) + i

                    # print(
                    #     f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] [D loss: {gan_loss.item()}] [G loss: {vq_loss.item()}]"
                    # )

                    # This saves images of real vs. generated designs every sample_interval
                    if batches_done % args.sample_interval == 0:
                        combined = np.stack([
                            decoded_images[-1].cpu().detach().numpy(), 
                            imgs[-1].cpu().detach().numpy()
                        ])
                        img_fname = os.path.join(self.results_dir, f"{batches_done}.png")
                        # img_fname = os.path.join(self.results_dir, f"{batches_done}.tiff")

                        plot_data(
                            combined, 
                            titles = ['Reconstruction', 'Real'], 
                            ranges = [[0, 1], [0, 1]], 
                            fname = img_fname,
                            cbar = False, 
                            dpi = 400, 
                            mirror_image = True, 
                            cmap = sns.color_palette("viridis", as_cmap=True), 
                            fontsize = 20
                        )

                        # --------------
                        #  Save models
                        # --------------
                        if args.save_model:
                            ckpt_gen = {
                                "epoch": epoch,
                                "batches_done": batches_done,
                                "generator": self.vqgan.state_dict(),
                                "optimizer_generator": self.opt_vq.state_dict(),
                                "loss": vq_loss.item(),
                            }
                            ckpt_disc = {
                                "epoch": epoch,
                                "batches_done": batches_done,
                                "discriminator": self.discriminator.state_dict(),
                                "optimizer_discriminator": self.opt_disc.state_dict(),
                                "loss": gan_loss.item(),
                            }

                            torch.save(ckpt_gen, os.path.join(self.checkpoints_dir, "vqgan.pth"))
                            torch.save(ckpt_disc, os.path.join(self.checkpoints_dir, "disc.pth"))
            
            if args.track:
                # Calculate average losses for this epoch
                d_loss_avg = sum(epoch_d_losses) / len(epoch_d_losses)
                g_loss_avg = sum(epoch_g_losses) / len(epoch_g_losses)
                
                # Track epoch averages
                self.log_losses['epochs'].append(epoch)
                self.log_losses['d_loss_avg'].append(np.log(d_loss_avg))
                self.log_losses['g_loss_avg'].append(np.log(g_loss_avg))
                
                # Plot and save losses
                plt.figure(figsize=(10, 5))
                plt.plot(self.log_losses['epochs'], self.log_losses['d_loss_avg'], label='D. Log-Loss')
                plt.plot(self.log_losses['epochs'], self.log_losses['g_loss_avg'], label='G. Log-Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Average Loss')
                plt.title('Training Losses Per Epoch')
                plt.legend()
                plt.grid(True)
                
                # Save the figure with a fixed name (overwriting previous versions)
                # loss_fname = os.path.join(self.results_dir, "log_loss.eps")
                # plt.savefig(loss_fname, format="eps", dpi=600, bbox_inches="tight", facecolor="white")
                loss_fname = os.path.join(self.results_dir, "log_loss.png")
                plt.savefig(loss_fname, format="png", dpi=300, bbox_inches="tight", transparent=True)
                plt.close()
                
                # Convert dictionary to arrays for proper numpy saving
                loss_data = np.array([
                    self.log_losses['epochs'],
                    self.log_losses['d_loss_avg'],
                    self.log_losses['g_loss_avg']
                ])
                
                # Save the loss data with a fixed name (overwriting previous versions)
                np.save(os.path.join(self.results_dir, "log_loss.npy"), loss_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='../data/gamma_4579_half.npy', help='Path to data (default: /data)') # New dataset path
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=16, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=0, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    # New arguments
    parser.add_argument('--conditions-path', type=str, default='../data/inp_paras_4579.npy', help='Path to conditions (default: ../data/inp_paras_4579.npy)')
    parser.add_argument('--problem-id', type=str, default='mto', help='Problem ID (default: mto)')
    parser.add_argument('--algo', type=str, default='vqgan', help='Algorithm name (default: vqgan)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--track', type=bool, default=True, help='track or not (default: True)')
    parser.add_argument('--save_model', type=bool, default=True, help='Save model checkpoint (default: True)')
    parser.add_argument('--sample_interval', type=int, default=215, help='Interval for saving sample images (default: 1000)')
    parser.add_argument('--run-name', type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help='Run name for this training session (default: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))')

    # Decoder-specific args
    parser.add_argument('--spectral_norm', type=bool, default=False, help='Apply spectral normalization to Conv layers (default: False)')
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
    parser.add_argument('--use_greyscale_lpips', type=bool, default=False, help='Use LPIPS for perceptual loss (default: False)')
    parser.add_argument('--use_DAE', type=bool, default=False, help='Use Decoupled Autoencoder for training (default: False)') # Not implemented
    parser.add_argument('--use_Online', type=bool, default=False, help='Use Online Clustered Codebook (default: False)') # Not implemented


    # TODO: Add arguments for encoder/decoder channel sizes, other options as in previous implementation

    args = parser.parse_args()
    print_args(args, "Training Arguments")
    train_vqgan = TrainVQGAN(args)



