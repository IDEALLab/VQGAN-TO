import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils

from transformer import VQGANTransformer
from utils import get_data, set_precision, set_all_seeds, safe_compile
from args import get_args, save_args, print_args


"""
Comprehensive training and metrics calculation + saving for Transformer models (Stage 2)
"""
class TrainTransformer:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)
    
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers(args)

        self.log_losses = {'epochs': [], 'train_loss_avg': [], 'val_loss_avg': []}
        saves_dir = os.path.join(r"../saves", args.run_name)
        self.results_dir = os.path.join(saves_dir, "results")
        self.checkpoints_dir = os.path.join(saves_dir, "checkpoints")
        self.saves_dir = saves_dir

        self.model = safe_compile(self.model)

        self.prepare_training()
        self.train(args)

    def prepare_training(self):
        os.makedirs(self.saves_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def configure_optimizers(self, args):
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

        optimizer = torch.optim.AdamW(optim_groups, lr=args.t_learning_rate, betas=(0.9, 0.95))
        return optimizer

    def train(self, args):
        (dataloader, val_dataloader, _), means, stds = get_data(args, use_val_split=True)
        best_val_loss = float('inf')

        for epoch in tqdm(range(args.epochs)):
            train_losses = []
            val_losses = []
            for imgs, c in dataloader:
                self.optim.zero_grad()
                imgs = imgs.to(device=args.device, non_blocking=True)
                c = c.to(device=args.device, non_blocking=True)
                logits, targets = self.model(imgs, c, pkeep=(args.pkeep if epoch >= args.pkeep_delay else 1.0))
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss.backward()
                self.optim.step()
                train_losses.append(loss.item())
            
            self.model.eval()

            # Validation loss and per-token accuracy
            total_token_correct = torch.zeros(256, device=args.device)
            total_token_count = torch.zeros(256, device=args.device)

            with torch.no_grad():
                for imgs, c in val_dataloader:
                    imgs = imgs.to(device=args.device, non_blocking=True)
                    c = c.to(device=args.device, non_blocking=True)
                    logits, targets = self.model(imgs, c)
                    val_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    val_losses.append(val_loss.item())

                    preds = torch.argmax(logits, dim=-1)  # [B, 256]
                    correct = (preds == targets).float()  # [B, 256]
                    total_token_correct += correct.sum(dim=0)
                    total_token_count += torch.tensor([imgs.shape[0]], device=args.device)

            if args.track:
                # Calculate average losses for this epoch
                train_loss_avg = sum(train_losses) / len(train_losses)
                val_loss_avg = sum(val_losses) / len(val_losses)
                
                # Track epoch averages
                self.log_losses['epochs'].append(epoch)
                self.log_losses['train_loss_avg'].append(np.log(train_loss_avg + 1e-8))
                self.log_losses['val_loss_avg'].append(np.log(val_loss_avg + 1e-8))
                
                # Save the loss data with a fixed name (overwriting previous versions)
                np.save(os.path.join(self.results_dir, "log_loss.npy"), np.array([self.log_losses[k] for k in self.log_losses]))

                if epoch % args.t_sample_interval == 0:
                    # Average accuracy per token position across val set
                    mean_token_accuracy = (total_token_correct / total_token_count.clamp(min=1)).view(16, 16).cpu().numpy()
                    avg_accuracy = mean_token_accuracy.mean()

                    plt.figure(figsize=(4, 4))
                    im = plt.imshow(mean_token_accuracy, cmap="viridis", vmin=0, vmax=1)
                    plt.axis("off")
                    plt.title(f"Mean Token Accuracy (Epoch {epoch}) — Avg: {avg_accuracy:.3f}", fontsize=12)
                    plt.colorbar(im, fraction=0.046, pad=0.04)
                    plt.savefig(os.path.join(self.results_dir, f"accuracy_{epoch}.png"), format="png", dpi=300, bbox_inches="tight", transparent=True)
                    plt.close()

                    # Plot and save losses
                    plt.figure(figsize=(10, 5))
                    plt.plot(self.log_losses['epochs'], self.log_losses['train_loss_avg'], label='Train Log-Loss')
                    plt.plot(self.log_losses['epochs'], self.log_losses['val_loss_avg'], label='Val Log-Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Average Log-Loss')
                    plt.title('Transformer Train/Val Loss')
                    plt.legend()
                    plt.grid(True)
                    
                    loss_fname = os.path.join(self.results_dir, "log_loss.png")
                    plt.savefig(loss_fname, format="png", dpi=300, bbox_inches="tight", transparent=True)
                    plt.close()
                    
                    sample_imgs, sample_cond = next(iter(val_dataloader))
                    sampled_imgs = self.model.log_images(sample_imgs[0][None].to(args.device), sample_cond[0][None].to(args.device), top_k=None, greedy=False)[1]
                    vutils.save_image(sampled_imgs, os.path.join(self.results_dir, f"epoch_{epoch}.png"), nrow=4)
                    # Save model if validation loss improved from the last interval
                    if val_loss_avg < best_val_loss or not args.T_min_validation:
                        best_val_loss = min(best_val_loss, val_loss_avg)
                        tqdm.write(f"Transformer checkpoint saved at epoch {epoch}.")
                        torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, "transformer.pt"))
                    elif args.early_stop:
                        print(f"Early stopping at epoch {epoch} due to no val loss improvement...")
                        break

            self.model.train()
        
        torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, f"transformer_final.pt"))


if __name__ == '__main__':
    args = get_args()
    args.is_t = True
    args.t_name = args.run_name
    print_args(args, title="Training Arguments")
    save_args(args)
    train_transformer = TrainTransformer(args)
