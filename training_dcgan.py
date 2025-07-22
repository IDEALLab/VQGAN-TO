import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from lpips import GreyscaleLPIPS

from dcgan import Generator, Discriminator, VQGANLatentWrapper
from utils import get_data, set_precision, set_all_seeds
from args import get_args, print_args, save_args


class TrainDCGAN:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)

        self.args = args
        self.device = args.device

        self.generator = Generator(args).to(self.device)
        self.discriminator = Discriminator(args).to(self.device)
        self.vq_wrapper = VQGANLatentWrapper(args).to(self.device)

        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=args.gan_g_learning_rate, betas=(args.beta1, args.beta2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=args.gan_d_learning_rate, betas=(args.beta1, args.beta2)
        )

        saves_dir = os.path.join("../saves", args.run_name)
        self.results_dir = os.path.join(saves_dir, "results")
        self.checkpoints_dir = os.path.join(saves_dir, "checkpoints")
        self.saves_dir = saves_dir

        os.makedirs(self.saves_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.best_val_l1 = float("inf")
        self.losses = {
            'epochs': [],
            'train_d_loss': [],
            'train_g_loss': [],
            'val_l1_loss': [],
        }

        self.train()

    def train(self):
        lpips = GreyscaleLPIPS().to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        (dataloader, val_dataloader, _), means, stds = get_data(self.args, use_val_split=True)

        real_label = 1.0
        fake_label = 0.0

        for epoch in tqdm(range(self.args.epochs)):
            train_d_losses, train_g_losses = [], []

            for i, (imgs, c) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                c = c.to(self.device)

                if self.args.gan_use_cvq:
                    c = self.vq_wrapper.c_encode(c)
                    c = c.view(c.shape[0], -1)

                batch_size = imgs.shape[0]
                real_latents = self.vq_wrapper.encode(imgs)
                # Get real image reconstructions for comparison
                recon_imgs = self.vq_wrapper.decode(real_latents).clamp(0, 1)

                # Create labels
                label_real = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
                label_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=self.device)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                self.optimizer_D.zero_grad()

                # Train with real
                output_real = self.discriminator(real_latents, c).view(-1)
                d_loss_real = criterion(output_real, label_real)
                d_loss_real.backward()

                # Train with fake
                z = torch.randn(batch_size, self.args.latent_dim, device=self.device)
                fake_latents = self.generator(z, c)
                output_fake = self.discriminator(fake_latents.detach(), c).view(-1)
                d_loss_fake = criterion(output_fake, label_fake)
                d_loss_fake.backward()

                # Update D
                d_loss = d_loss_real + d_loss_fake
                self.optimizer_D.step()
                train_d_losses.append(d_loss.item())

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.optimizer_G.zero_grad()

                # Since D was just updated, we need to recompute D(G(z))
                output_fake = self.discriminator(fake_latents, c).view(-1)
                g_loss = criterion(output_fake, label_real)  # Fake labels are real for generator cost

                real_mean, real_std = real_latents.mean(dim=[2, 3]), real_latents.std(dim=[2, 3])
                fake_mean, fake_std = fake_latents.mean(dim=[2, 3]), fake_latents.std(dim=[2, 3])

                # Decode generated latents to images
                gen_imgs = self.vq_wrapper.decode(fake_latents).clamp(0, 1)

                # Compute auxiliary losses (kept as comments)
                perceptual_loss = lpips(recon_imgs, gen_imgs).mean()
                recon_loss = F.l1_loss(recon_imgs, gen_imgs)
                vf_loss = torch.abs(recon_imgs.mean() - gen_imgs.mean())
                intermediate_loss = (0.25 - ((gen_imgs - 0.5) ** 2)).mean()
                dist_loss = F.l1_loss(fake_mean, real_mean) + F.l1_loss(fake_std, real_std)

                # Generator loss: adversarial + auxiliary
                print(f"g_loss: {g_loss.item()}")
                print(f"recon_loss: {recon_loss.item()}")
                print(f"perceptual_loss: {perceptual_loss.item()}")
                print(f"vf_loss: {vf_loss.item()}")
                print(f"intermediate_loss: {intermediate_loss.item()}")
                print(f"dist_loss: {dist_loss.item()}\n\n")
                g_loss += 1e0 * recon_loss
                g_loss += 1e1 * perceptual_loss
                g_loss += 1e1 * vf_loss
                g_loss += 1e0 * intermediate_loss
                g_loss += 1e0 * dist_loss

                g_loss.backward()
                self.optimizer_G.step()
                train_g_losses.append(g_loss.item())

            # Log average losses
            train_d_avg = sum(train_d_losses) / len(train_d_losses)
            train_g_avg = sum(train_g_losses) / len(train_g_losses) if train_g_losses else 0.0
            self.losses['epochs'].append(epoch)
            self.losses['train_d_loss'].append(train_d_avg)
            self.losses['train_g_loss'].append(train_g_avg)

            # Validation L1 loss
            self.generator.eval()
            self.vq_wrapper.eval()
            val_l1_losses = []

            with torch.no_grad():
                for val_imgs, val_c in val_dataloader:
                    val_imgs = val_imgs.to(self.device)
                    val_c = val_c.to(self.device)

                    if self.args.gan_use_cvq:
                        val_c = self.vq_wrapper.c_encode(val_c)
                        val_c = val_c.view(val_c.shape[0], -1)

                    z = torch.randn(val_imgs.shape[0], self.args.latent_dim, device=self.device)
                    gen_latents = self.generator(z, val_c)
                    gen_imgs = self.vq_wrapper.decode(gen_latents).clamp(0, 1)

                    recon_latents = self.vq_wrapper.encode(val_imgs)
                    recon_imgs = self.vq_wrapper.decode(recon_latents).clamp(0, 1)

                    val_l1_loss = F.l1_loss(recon_imgs, gen_imgs)
                    val_l1_losses.append(val_l1_loss.item())

            val_l1_avg = sum(val_l1_losses) / len(val_l1_losses)
            print(val_l1_avg)
            self.losses['val_l1_loss'].append(val_l1_avg)

            np.save(os.path.join(self.results_dir, "loss.npy"), np.array([self.losses[k] for k in self.losses]))

            if epoch % self.args.gan_sample_interval == 0:
                # Plot log-scaled losses (log(1 + loss))
                plt.figure(figsize=(10, 5))
                plt.plot(self.losses['epochs'], np.log1p(np.abs(self.losses['train_d_loss'])), label='Train D Loss (log |abs|)')
                plt.plot(self.losses['epochs'], np.log1p(np.abs(self.losses['train_g_loss'])), label='Train G Loss (log |abs|)')
                plt.plot(self.losses['epochs'], np.log1p(self.losses['val_l1_loss']), label='Val L1 Loss (log)')
                plt.xlabel('Epochs')
                plt.ylabel('log(1 + |Loss|)')
                plt.title('DCGAN Log-Scaled Losses')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.results_dir, "log_loss.png"), dpi=300, bbox_inches="tight", transparent=True)
                plt.close()

                self.generator.eval()
                self.vq_wrapper.eval()

                with torch.no_grad():
                    sample_imgs, sample_c = next(iter(dataloader))
                    sample_imgs = sample_imgs.to(self.device)
                    sample_c = sample_c.to(self.device)
                    if self.args.gan_use_cvq:
                        sample_c = self.vq_wrapper.c_encode(sample_c)
                        sample_c = sample_c.view(sample_c.shape[0], -1)

                    real_latents = self.vq_wrapper.encode(sample_imgs)
                    recon_imgs = self.vq_wrapper.decode(real_latents).clamp(0, 1)

                    z = torch.randn(sample_imgs.shape[0], self.args.latent_dim, device=self.device)
                    gen_latents = self.generator(z, sample_c)
                    gen_imgs = self.vq_wrapper.decode(gen_latents).clamp(0, 1)

                    triplets = torch.cat([sample_imgs, recon_imgs, gen_imgs], dim=0)
                    grid = make_grid(triplets, nrow=sample_imgs.size(0), normalize=True, scale_each=True)
                    save_image(grid, os.path.join(self.results_dir, f"epoch_{epoch}_triplet.png"))

                # Conditional checkpointing + early stopping
                if self.args.save_model:
                    if not self.args.gan_min_validation or val_l1_avg < self.best_val_l1:
                        self.best_val_l1 = min(self.best_val_l1, val_l1_avg)
                        tqdm.write(f"DCGAN checkpoint saved at epoch {epoch}.")
                        torch.save(self.generator.state_dict(), os.path.join(self.checkpoints_dir, "generator.pt"))
                        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoints_dir, "discriminator.pt"))
                    elif self.args.early_stop:
                        print(f"Early stopping at epoch {epoch} due to no val loss improvement...")
                        break

            self.generator.train()
            self.vq_wrapper.train()

        torch.save(self.generator.state_dict(), os.path.join(self.checkpoints_dir, "generator_final.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoints_dir, "discriminator_final.pt"))


if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Training Arguments")
    save_args(args)
    train_dcgan = TrainDCGAN(args)