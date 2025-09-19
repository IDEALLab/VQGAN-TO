import os
import numpy as np
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from discriminator import Discriminator, weights_init
from lpips import LPIPS, GreyscaleLPIPS, NoLPIPS
from vqgan import VQGAN
from utils import get_data, plot_data, set_precision, set_all_seeds, plot_3d_scatter_comparison, safe_compile
from args import get_args, save_args, print_args


"""
Comprehensive training and metrics calculation + saving for VQGAN models (Stage 1)
"""
class TrainVQGAN:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)

        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        if args.is_c:
            self.perceptual_loss = NoLPIPS().eval().to(device=args.device)
        else:
            self.perceptual_loss = (GreyscaleLPIPS() if args.use_greyscale_lpips else LPIPS()).eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.log_losses = {
            'epochs': [],
            'd_loss_avg': [],
            'g_loss_avg': [],
            'val_loss_avg': [],
            'rec_loss_avg': [],
            'perceptual_loss_avg': [],
            'q_loss_avg': [],
            'val_rec_loss_avg': [],
            'val_perceptual_loss_avg': [],
            'val_q_loss_avg': [],
        }

        self.log_codebook_usage = {'epochs': [], 'active_vectors': [], 'usage_percentage': [], 'entropy': []}

        saves_dir = os.path.join(r"../saves", args.run_name)
        self.results_dir = os.path.join(saves_dir, "results")
        self.checkpoints_dir = os.path.join(saves_dir, "checkpoints")
        self.saves_dir = saves_dir

        self.vqgan = safe_compile(self.vqgan)
        self.discriminator = safe_compile(self.discriminator)

        self.prepare_training()
        self.train(args)

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
        (dataloader, val_dataloader, _), means, stds = get_data(args, use_val_split=True)
        best_val_loss = float('inf')
        val_loss_avg = float('inf')
        
        for epoch in tqdm(range(args.epochs)):
            epoch_d_losses = []
            epoch_g_losses = []
            val_losses = []
            epoch_rec_losses = []
            epoch_percep_losses = []
            epoch_q_losses = []
            val_rec_losses = []
            val_percep_losses = []
            val_q_losses = []
            epoch_codebook_usage = {}  # Reset codebook usage tracking for each epoch

            # Note: freezes encoder + quant_conv if using DAE after specified epoch
            if epoch == args.DAE_switch_epoch and args.use_DAE:
                self.vqgan.switch_to_new_decoder()
                for m in [self.vqgan.encoder, self.vqgan.quant_conv]:
                    for p in m.parameters():
                        p.requires_grad = False
                
                self.opt_vq = torch.optim.Adam(
                    list(self.vqgan.decoder.parameters()) +
                    list(self.vqgan.codebook.parameters()) +
                    list(self.vqgan.post_quant_conv.parameters()),
                    lr=args.learning_rate, eps=1e-08, betas=(args.beta1, args.beta2)
                )
                tqdm.write(f"DAE Switched to full decoder and froze encoder + quant_conv at epoch {epoch}")
            
            for i, (imgs, _) in enumerate(dataloader):
                imgs = imgs.to(device=args.device, non_blocking=True)
                decoded_images, codebook_indices, q_loss = self.vqgan(imgs)

                disc_real = self.discriminator(imgs)
                disc_fake = self.discriminator(decoded_images)

                disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch, threshold=args.disc_start)

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
                vq_loss.backward(retain_graph=True)

                self.opt_disc.zero_grad()
                gan_loss.backward()

                self.opt_vq.step()
                self.opt_disc.step()

                epoch_d_losses.append(gan_loss.item())
                epoch_g_losses.append(vq_loss.item())
                epoch_rec_losses.append(rec_loss.mean().item())
                epoch_percep_losses.append(perceptual_loss.mean().item())
                epoch_q_losses.append(q_loss.item())

                # Log codebook usage statistics of every batch
                if args.track:
                    indices = codebook_indices.cpu().detach().numpy()
                    unique_indices, counts = np.unique(indices, return_counts=True)
                    for idx, count in zip(unique_indices, counts):
                        if idx in epoch_codebook_usage:
                            epoch_codebook_usage[idx] += count
                        else:
                            epoch_codebook_usage[idx] = count

            if args.vq_track_val_loss:

                self.vqgan.eval()
                self.discriminator.eval()

                with torch.no_grad():
                    for i, (imgs, _) in enumerate(val_dataloader):
                        imgs = imgs.to(device=args.device, non_blocking=True)
                        decoded_images, codebook_indices, q_loss = self.vqgan(imgs)
                        perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                        rec_loss = torch.abs(imgs - decoded_images)
                        perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                        perceptual_rec_loss = perceptual_rec_loss.mean()
                        vq_val_loss = perceptual_rec_loss + q_loss
                        val_losses.append(vq_val_loss.item())
                        val_rec_losses.append(rec_loss.mean().item())
                        val_percep_losses.append(perceptual_loss.mean().item())
                        val_q_losses.append(q_loss.item())
                
                self.vqgan.train()
                self.discriminator.train()

            if args.track:
                # Calculate and log metrics every epoch
                if args.vq_track_val_loss:
                    val_loss_avg = sum(val_losses) / len(val_losses)
                d_loss_avg = sum(epoch_d_losses) / len(epoch_d_losses)
                g_loss_avg = sum(epoch_g_losses) / len(epoch_g_losses)
                rec_loss_avg = np.mean(epoch_rec_losses)
                percep_loss_avg = np.mean(epoch_percep_losses)
                q_loss_avg = np.mean(epoch_q_losses)

                self.log_losses['rec_loss_avg'].append(np.log(rec_loss_avg + 1e-8))
                self.log_losses['perceptual_loss_avg'].append(np.log(percep_loss_avg + 1e-8))
                self.log_losses['q_loss_avg'].append(np.log(q_loss_avg + 1e-8))

                if args.vq_track_val_loss:
                    self.log_losses['val_rec_loss_avg'].append(np.log(np.mean(val_rec_losses) + 1e-8))
                    self.log_losses['val_perceptual_loss_avg'].append(np.log(np.mean(val_percep_losses) + 1e-8))
                    self.log_losses['val_q_loss_avg'].append(np.log(np.mean(val_q_losses) + 1e-8))
                else:
                    self.log_losses['val_rec_loss_avg'].append(None)
                    self.log_losses['val_perceptual_loss_avg'].append(None)
                    self.log_losses['val_q_loss_avg'].append(None)
                
                # Calculate and track codebook usage statistics for this epoch
                active_codes = len(epoch_codebook_usage)
                usage_percentage = active_codes/args.num_codebook_vectors*100

                # Compute normalized entropy from usage counts
                counts = np.array(list(epoch_codebook_usage.values()))
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-8))  # Add epsilon to avoid log(0)
                max_entropy = np.log(args.num_codebook_vectors)
                normalized_entropy = entropy / max_entropy
                
                if args.vq_track_val_loss:
                    self.log_losses['val_loss_avg'].append(np.log(val_loss_avg + 1e-8))
                else:
                    self.log_losses['val_loss_avg'].append(None)
                if epoch >= args.disc_start:
                    self.log_losses['d_loss_avg'].append(np.log(d_loss_avg + 1e-8))
                else:
                    self.log_losses['d_loss_avg'].append(None)
                self.log_losses['g_loss_avg'].append(np.log(g_loss_avg + 1e-8))
                self.log_losses['epochs'].append(epoch)
                
                self.log_codebook_usage['active_vectors'].append(active_codes)
                self.log_codebook_usage['usage_percentage'].append(usage_percentage)
                self.log_codebook_usage['entropy'].append(normalized_entropy)
                self.log_codebook_usage['epochs'].append(epoch)

                if epoch % args.sample_interval == 0:
                    # Plot and save losses + codebook usage
                    plt.figure(figsize=(10, 5))
                    
                    if epoch >= args.disc_start:
                        # Only plot d_loss_avg for epochs >= disc_start
                        valid_d_epochs = [e for e, d in zip(self.log_losses['epochs'], self.log_losses['d_loss_avg']) if d is not None]
                        valid_d_vals = [d for d in self.log_losses['d_loss_avg'] if d is not None]
                        plt.plot(valid_d_epochs, valid_d_vals, label='D. Log-Loss')
                    if args.vq_track_val_loss:
                        # Only plot val_loss_avg if early stopping is enabled
                        valid_val_epochs = [e for e, v in zip(self.log_losses['epochs'], self.log_losses['val_loss_avg']) if v is not None]
                        valid_val_vals = [v for v in self.log_losses['val_loss_avg'] if v is not None]
                        plt.plot(valid_val_epochs, valid_val_vals, label='Val. Log-Loss')

                    plt.plot(self.log_losses['epochs'], self.log_losses['g_loss_avg'], label='G. Log-Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Average Loss')
                    plt.title('Training Losses Per Epoch')
                    plt.legend()
                    plt.grid(True)
                    
                    loss_fname = os.path.join(self.results_dir, "log_loss.png")
                    plt.savefig(loss_fname, format="png", dpi=300, bbox_inches="tight", transparent=True)
                    plt.close()

                    # Plot separated generator loss components - TRAIN
                    plt.figure(figsize=(10, 6))
                    epochs = self.log_losses['epochs']
                    plt.plot(epochs, self.log_losses['rec_loss_avg'], label='rec_loss')
                    plt.plot(epochs, self.log_losses['perceptual_loss_avg'], label='perceptual_loss')
                    plt.plot(epochs, self.log_losses['q_loss_avg'], label='q_loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Log Loss')
                    plt.title('Train Generator Losses')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.results_dir, "train_g_losses.png"), format="png", dpi=300, bbox_inches="tight", transparent=True)
                    plt.close()

                    # Plot separated generator loss components - VAL
                    if args.vq_track_val_loss:
                        plt.figure(figsize=(10, 6))
                        valid_epochs = self.log_losses['epochs']
                        plt.plot(valid_epochs, [v for v in self.log_losses['val_rec_loss_avg'] if v is not None], label='rec_loss')
                        plt.plot(valid_epochs, [v for v in self.log_losses['val_perceptual_loss_avg'] if v is not None], label='perceptual_loss')
                        plt.plot(valid_epochs, [v for v in self.log_losses['val_q_loss_avg'] if v is not None], label='q_loss')
                        plt.xlabel('Epochs')
                        plt.ylabel('Log Loss')
                        plt.title('Validation Generator Losses')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.results_dir, "val_g_losses.png"), format="png", dpi=300, bbox_inches="tight", transparent=True)
                        plt.close()
                    
                    # Codebook always uses a lot of vectors first epoch, massively decreases second epoch --> Only plot after skipping the first epoch
                    if len(self.log_codebook_usage['epochs']) > 1:
                        plt.figure(figsize=(10, 5))
                        epochs_to_plot = self.log_codebook_usage['epochs'][1:]
                        vectors_to_plot = self.log_codebook_usage['active_vectors'][1:]
                        plt.plot(epochs_to_plot, vectors_to_plot, label='Active Vectors', color='green')
                        plt.xlabel('Epochs')
                        plt.ylabel('Number of Active Vectors')
                        plt.title('Codebook Usage Over Training')
                        plt.legend()
                        plt.grid(True)
                        
                        codebook_fname = os.path.join(self.results_dir, "codebook_usage.png")
                        plt.savefig(codebook_fname, format="png", dpi=300, bbox_inches="tight", transparent=True)
                        plt.close()

                        # Plot normalized entropy over training
                        plt.figure(figsize=(10, 5))
                        entropy_vals = self.log_codebook_usage['entropy'][1:]
                        epochs_to_plot = self.log_codebook_usage['epochs'][1:]
                        plt.plot(epochs_to_plot, entropy_vals, label='Normalized Entropy', color='orange')
                        plt.xlabel('Epochs')
                        plt.ylabel('Entropy (Normalized)')
                        plt.title('Codebook Entropy Over Training')
                        plt.legend()
                        plt.grid(True)

                        entropy_fname = os.path.join(self.results_dir, "codebook_entropy.png")
                        plt.savefig(entropy_fname, format="png", dpi=300, bbox_inches="tight", transparent=True)
                        plt.close()
                    
                    np.savez(os.path.join(self.results_dir, "log_loss.npz"), **self.log_losses)
                    np.save(os.path.join(self.results_dir, "codebook_usage.npy"), np.array([self.log_codebook_usage[k] for k in self.log_codebook_usage]))

                    # Save sample images from the current epoch
                    if args.is_c:
                        scatter_fname = os.path.join(self.results_dir, f"scatter_epoch_{epoch}.png")
                        plot_3d_scatter_comparison(decoded_images, imgs, scatter_fname)
                    else:
                        combined = np.stack([decoded_images[-1].cpu().detach().numpy(), imgs[-1].cpu().detach().numpy()])
                        img_fname = os.path.join(self.results_dir, f"epoch_{epoch}.png")
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

                    # Save the latest model state
                    if args.save_model and (not args.vq_min_validation or (args.vq_min_validation and val_loss_avg < best_val_loss)):
                        ckpt_gen = {
                            "epoch": epoch,
                            "generator": self.vqgan.state_dict(),
                            "optimizer_generator": self.opt_vq.state_dict(),
                            "loss": vq_loss.item(),
                        }
                        ckpt_disc = {
                            "epoch": epoch,
                            "discriminator": self.discriminator.state_dict(),
                            "optimizer_discriminator": self.opt_disc.state_dict(),
                            "loss": gan_loss.item(),
                        }

                        torch.save(ckpt_gen, os.path.join(self.checkpoints_dir, "vqgan.pth"))
                        torch.save(ckpt_disc, os.path.join(self.checkpoints_dir, "disc.pth"))

                        if val_loss_avg < best_val_loss and args.vq_min_validation:
                            best_val_loss = val_loss_avg
                            tqdm.write(f"VQGAN checkpoint saved at epoch {epoch} with validation loss {val_loss_avg:.4f}.")

if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Training Arguments")
    save_args(args)
    train_vqgan = TrainVQGAN(args)
