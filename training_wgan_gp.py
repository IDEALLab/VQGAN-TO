import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from wgan_gp import Generator, Discriminator, compute_gradient_penalty, VQGANLatentWrapper
from utils import get_data, set_precision, set_all_seeds
from args import get_args, print_args, save_args


class TrainWGAN_GP:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)

        self.args = args
        self.device = args.device

        self.generator = Generator(args).to(self.device)
        self.discriminator = Discriminator(args).to(self.device)
        self.vq_wrapper = VQGANLatentWrapper(args).to(self.device)

        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2)
        )

        saves_dir = os.path.join("../saves", args.run_name)
        self.results_dir = os.path.join(saves_dir, "results")
        self.checkpoints_dir = os.path.join(saves_dir, "checkpoints")
        self.saves_dir = saves_dir

        os.makedirs(self.saves_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.losses = {
            'epochs': [],
            'train_d_loss': [],
            'train_g_loss': [],
        }

        self.train()

    def train(self):
        (dataloader, _, _), _, _ = get_data(self.args, use_val_split=False)

        for epoch in tqdm(range(self.args.epochs)):
            train_d_losses, train_g_losses = [], []

            for i, (imgs, c) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                c = c.to(self.device)

                if args.gan_use_cvq:
                    c = self.vq_wrapper.c_encode(c)
                    c = c.view(c.shape[0], -1)

                real_latents = self.vq_wrapper.encode(imgs)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                z = torch.randn(imgs.shape[0], self.args.latent_dim, device=self.device)
                fake_imgs = self.generator(z, c)

                real_validity = self.discriminator(real_latents, c)
                fake_validity = self.discriminator(fake_imgs, c)
                gp = compute_gradient_penalty(self.discriminator, real_latents, fake_imgs, c, self.device)

                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.args.lambda_gp * gp
                d_loss.backward()
                self.optimizer_D.step()
                train_d_losses.append(d_loss.item())

                # Train Generator every n_critic steps
                if i % self.args.n_critic == 0:
                    self.optimizer_G.zero_grad()
                    z = torch.randn(imgs.shape[0], self.args.latent_dim, device=self.device)
                    fake_imgs = self.generator(z, c)
                    fake_validity = self.discriminator(fake_imgs, c)
                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward()
                    self.optimizer_G.step()
                    train_g_losses.append(g_loss.item())

            # Log average losses
            train_d_avg = sum(train_d_losses) / len(train_d_losses)
            train_g_avg = sum(train_g_losses) / len(train_g_losses) if train_g_losses else 0.0

            self.losses['epochs'].append(epoch)
            self.losses['train_d_loss'].append(train_d_avg)
            self.losses['train_g_loss'].append(train_g_avg)

            np.save(os.path.join(self.results_dir, "loss.npy"), np.array([self.losses[k] for k in self.losses]))

            if epoch % self.args.gan_sample_interval == 0:
                # Plot losses
                plt.figure(figsize=(10, 5))
                plt.plot(self.losses['epochs'], self.losses['train_d_loss'], label='Train D Loss')
                plt.plot(self.losses['epochs'], self.losses['train_g_loss'], label='Train G Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('WGAN-GP Training Loss')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.results_dir, "loss.png"), dpi=300, bbox_inches="tight", transparent=True)
                plt.close()

                # Save triplet grid: original / reconstructed / generated
                self.generator.eval()
                self.vq_wrapper.eval()

                with torch.no_grad():
                    sample_imgs, sample_c = next(iter(dataloader))
                    sample_imgs = sample_imgs.to(self.device)
                    sample_c = sample_c.to(self.device)
                    if args.gan_use_cvq:
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

                self.generator.train()
                self.vq_wrapper.train()

                if self.args.save_model:
                    torch.save(self.generator.state_dict(), os.path.join(self.checkpoints_dir, "generator.pt"))
                    torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoints_dir, "discriminator.pt"))

        # Final checkpoint
        torch.save(self.generator.state_dict(), os.path.join(self.checkpoints_dir, "generator_final.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoints_dir, "discriminator_final.pt"))


if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Training Arguments")
    save_args(args)
    train_wgan_gp = TrainWGAN_GP(args)
