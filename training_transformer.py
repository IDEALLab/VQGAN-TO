import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import get_data, set_precision, set_all_seeds
from args import get_args, save_args, print_args


class TrainTransformer:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)
    
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()

        self.log_losses = {'epochs': [], 'train_loss_avg': [], 'test_loss_avg': []}
        saves_dir = os.path.join(r"../saves", args.run_name)
        self.results_dir = os.path.join(saves_dir, "results")
        self.checkpoints_dir = os.path.join(saves_dir, "checkpoints")
        self.saves_dir = saves_dir

        self.prepare_training()
        self.train(args)

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
    args = get_args()
    print_args(args, title="Training Arguments")
    save_args(args)
    train_transformer = TrainTransformer(args)
