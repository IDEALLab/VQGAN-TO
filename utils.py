import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Compute BCE loss per element
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        
        # Compute pt
        pt = torch.exp(-bce_loss)
        
        # Apply focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    
# Now loads in the full dataset with conditions
def get_data(args):
    x = torch.from_numpy(np.load(args.dataset_path).astype(np.float32)).reshape(-1, args.image_channels, args.image_size, args.image_size)
    c_orig = torch.from_numpy(np.load(args.conditions_path).astype(np.float32))
    c, means, stds = normalize(c_orig)

    generator = torch.Generator().manual_seed(0)
    train_data, test_data = random_split(TensorDataset(x, c), [0.75, 0.25], generator=generator)
    dataloader, test_dataloader = load_data(args, train_data, test_data, generator)
    
    return dataloader, test_dataloader, means, stds


# Now adds in test dataset
def load_data(args, train_data, test_data, g):
    dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=g)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, generator=g)
    return dataloader, test_dataloader


# Did not exist in original code (not conditional)
def normalize(data):
    means = torch.mean(data, dim=0)
    stds = torch.std(data, dim=0)
    return (data-means)/stds, list(means.numpy()), list(stds.numpy())


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def mirror(data, dim=-1, reshape=None, difference=False):
    while len(data.shape) < 4:
        data = data.unsqueeze(0)
    new = torch.cat((data, torch.flip(data, (dim,))), dim)
    if reshape is not None:
        if difference:
            new = torch.clamp(F.interpolate(new, reshape, mode='bicubic'), -1, 1)
        else:
            new = torch.clamp(F.interpolate(new, reshape, mode='bicubic'), 0, 1)
    return new


def plot_data(data, titles, ranges, fname=None, dpi=100, mirror_image=False, cmap=None, cbar=True, fontsize=20):
    L = len(titles)
    fig, axs = plt.subplots(1, L, figsize=(int(5*L), 4))
    [ax.axes.xaxis.set_visible(False) for ax in axs]
    [ax.axes.yaxis.set_visible(False) for ax in axs]
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams["figure.dpi"] = dpi

    for idx, (figure, title, current_range) in enumerate(zip(data, titles, ranges)):
        if mirror_image:
            figure = np.array(mirror(torch.tensor(figure), reshape=(400, 400), difference=(title=="Difference")))[0]
        if title == "Difference":
            sns.heatmap(ax=axs[idx], data=figure[0], cbar=cbar, vmin=current_range[0], vmax=current_range[1], center=0, cmap="RdBu_r")
        else:
            sns.heatmap(ax=axs[idx], data=figure[0], cbar=cbar, vmin=current_range[0], vmax=current_range[1], cmap=cmap)
        axs[idx].set_title(title, fontsize=fontsize)
    if fname is None:
        plt.show()
    else:
        if fname.endswith(".eps"):
            # For vector graphics — high DPI, white background
            plt.savefig(fname, format="eps", dpi=600, bbox_inches="tight", facecolor="white")
        elif fname.endswith(".tiff") or fname.endswith(".tif"):
            # For raster images — no transparency, white background
            plt.savefig(fname, format="tiff", dpi=300, bbox_inches="tight", facecolor="white")
        else:
            # For formats like PNG or PDF — allow transparency
            plt.savefig(fname, format="png", dpi=300, bbox_inches="tight", transparent=True)

        fig.clear()
        plt.close(fig)
        del(fig)
        del(axs)

def print_args(args, title="Current Arguments"):
    """Print all arguments in a formatted way"""
    print(f"\n{'-'*20} {title} {'-'*20}")
    args_dict = vars(args)
    for k, v in sorted(args_dict.items()):
        print(f"{k}: {v}")
    print(f"{'-'*50}\n")
