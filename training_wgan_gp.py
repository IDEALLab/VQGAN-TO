import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

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

        # Setup directories using gan_name
        saves_dir = os.path.join("../saves", args.gan_name)
        self.results_dir = os.path.join(saves_dir, "results")
        self.checkpoints_dir = os.path.join(saves_dir, "checkpoints")
        self.saves_dir = saves_dir

        os.makedirs(self.saves_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.log_losses = {'epochs': [], 'train_loss_avg': [], 'val_loss_avg': []}
        self.train()

    def train(self):
        (dataloader, val_dataloader, _), _, _ = get_data(self.args, use_val_split=True)
        best_val_loss = float('inf')
        batches_done = 0

        for epoch in tqdm(range(self.args.epochs)):
            train_d_losses, train_g_losses = [], []
            val_d_losses, val_g_losses = [], []

            for i, (imgs, c) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                c = c.to(self.device)

                real_latents = self.vq_wrapper.encode(imgs)

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

                if i % self.args.n_critic == 0:
                    self.optimizer_G.zero_grad()
                    z = torch.randn(imgs.shape[0], self.args.latent_dim, device=self.device)
                    fake_imgs = self.generator(z, c)
                    fake_validity = self.discriminator(fake_imgs, c)
                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward()
                    self.optimizer_G.step()
                    train_g_losses.append(g_loss.item())

                    if batches_done % self.args.gan_sample_interval == 0:
                        save_image(fake_imgs.data[:25], os.path.join(self.results_dir, f"epoch_{epoch}_samples.png"), nrow=5, normalize=True)

                    batches_done += self.args.n_critic

            self.generator.eval()
            self.discriminator.eval()
            with torch.no_grad():
                for val_imgs, val_c in val_dataloader:
                    val_imgs = val_imgs.to(self.device)
                    val_c = val_c.to(self.device)
                    val_latents = self.vq_wrapper.encode(val_imgs)
                    z = torch.randn(val_imgs.shape[0], self.args.latent_dim, device=self.device)
                    fake_val_imgs = self.generator(z, val_c)

                    real_val = self.discriminator(val_latents, val_c)
                    fake_val = self.discriminator(fake_val_imgs, val_c)
                    val_gp = compute_gradient_penalty(self.discriminator, val_latents, fake_val_imgs, val_c, self.device)

                    val_d_loss = -torch.mean(real_val) + torch.mean(fake_val) + self.args.lambda_gp * val_gp
                    val_g_loss = -torch.mean(self.discriminator(self.generator(z, val_c), val_c))

                    val_d_losses.append(val_d_loss.item())
                    val_g_losses.append(val_g_loss.item())

            self.generator.train()
            self.discriminator.train()

            if self.args.track:
                train_loss_avg = sum(train_d_losses) / len(train_d_losses)
                val_loss_avg = sum(val_d_losses) / len(val_d_losses)

                self.log_losses['epochs'].append(epoch)
                self.log_losses['train_loss_avg'].append(np.log(train_loss_avg + 1e-8))
                self.log_losses['val_loss_avg'].append(np.log(val_loss_avg + 1e-8))

                np.save(os.path.join(self.results_dir, "log_loss.npy"), np.array([self.log_losses[k] for k in self.log_losses]))

                if epoch % self.args.gan_sample_interval == 0:
                    plt.figure(figsize=(10, 5))
                    plt.plot(self.log_losses['epochs'], self.log_losses['train_loss_avg'], label='Train Log-Loss')
                    plt.plot(self.log_losses['epochs'], self.log_losses['val_loss_avg'], label='Val Log-Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Average Log-Loss')
                    plt.title('WGAN-GP Train/Val Loss')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(self.results_dir, "log_loss.png"), dpi=300, bbox_inches="tight", transparent=True)
                    plt.close()

                    if self.args.save_model:
                        torch.save(self.generator.state_dict(), os.path.join(self.checkpoints_dir, "generator.pt"))
                        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoints_dir, "discriminator.pt"))

                        if val_loss_avg < best_val_loss and self.args.gan_min_validation:
                            best_val_loss = val_loss_avg
                            tqdm.write(f"WGAN-GP checkpoint saved at epoch {epoch} with validation loss {val_loss_avg:.4f}.")

        torch.save(self.generator.state_dict(), os.path.join(self.checkpoints_dir, "generator_final.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoints_dir, "discriminator_final.pt"))


if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Training Arguments")
    save_args(args)
    train_wgan_gp = TrainWGAN_GP(args)
