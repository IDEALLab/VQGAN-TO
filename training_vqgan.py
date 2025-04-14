import os
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import get_data, weights_init, plot_data
import wandb

results_dir = r"../results"
checkpoints_dir = r"../checkpoints"
wandb_dir = r"../wandb"

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

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

    @staticmethod
    def prepare_training():
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

    def train(self, args):
        # Logging
        run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
        if args.track:
            wandb.init(settings=wandb.Settings(mode=args.wandb_online), project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True, name=run_name, dir=wandb_dir)

        dataloader, test_dataloader, means, stds = get_data(args)
        steps_per_epoch = len(dataloader)
        for epoch in tqdm(range(args.epochs)):
            with tqdm(range(steps_per_epoch)) as pbar:
                for i, (imgs, c) in enumerate(dataloader):
                    imgs = imgs.to(device=args.device)
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
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if args.track:
                        batches_done = epoch * len(dataloader) + i

                        wandb.log(
                            {
                                "d_loss": gan_loss.item(),
                                "g_loss": vq_loss.item(),
                                "epoch": epoch,
                                "batch": batches_done,
                            }
                        )
                        print(
                            f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] [D loss: {gan_loss.item()}] [G loss: {vq_loss.item()}]"
                        )

                        pbar.set_postfix(
                            VQ_Loss = np.round(vq_loss.cpu().detach().numpy().item(), 5),
                            GAN_Loss = np.round(gan_loss.cpu().detach().numpy().item(), 5)
                        )

                        # This saves a grid image of 25 generated designs every sample_interval
                        if batches_done % args.sample_interval == 0:
                            combined = np.stack([
                                decoded_images[-1].cpu().detach().numpy(), 
                                imgs[-1].cpu().detach().numpy()
                            ])
                            img_fname = os.path.join(results_dir, f"{batches_done}.png")

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
                            
                            wandb.log({"designs": wandb.Image(img_fname)})

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

                                torch.save(ckpt_gen, os.path.join(checkpoints_dir, "vqgan.pth"))
                                torch.save(ckpt_disc, os.path.join(checkpoints_dir, "disc.pth"))
                                artifact_gen = wandb.Artifact(f"{args.algo}_generator", type="model")
                                artifact_gen.add_file(os.path.join(checkpoints_dir, "vqgan.pth"))
                                artifact_disc = wandb.Artifact(f"{args.algo}_discriminator", type="model")
                                artifact_disc.add_file(os.path.join(checkpoints_dir, "disc.pth"))

                                wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
                                wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

        wandb.finish()

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
    parser.add_argument('--track', type=bool, default=True, help='track with wandb or not (default: True)')
    parser.add_argument('--wandb-online', type=str, default="offline", help='WandB online mode (default: online)')
    parser.add_argument('--wandb-project', type=str, default='vqgan', help='WandB project name (default: vqgan)')
    parser.add_argument('--wandb-entity', type=str, default=None, help='WandB entity name (default: None)')
    parser.add_argument('--save_model', type=bool, default=False, help='Save model checkpoint (default: True)')
    parser.add_argument('--sample_interval', type=int, default=3440, help='Interval for saving sample images (default: 1000)')

    # TODO: Add arguments for encoder/decoder channel sizes, other options as in previous implementation

    args = parser.parse_args()
    train_vqgan = TrainVQGAN(args)



