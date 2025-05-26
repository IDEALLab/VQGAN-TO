import os
import numpy as np
import random
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from copy import deepcopy
from vqgan import VQGAN
        

def set_precision():
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        if major >= 8:
            torch.set_float32_matmul_precision('high')
    

def set_all_seeds(seed):
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set environment variables (helps with some PyTorch operations)
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")


def get_data(args, use_val_split=False):
    # Load and normalize conditions
    c_orig = torch.from_numpy(np.load(args.conditions_path).astype(np.float32))
    c, means, stds = normalize(c_orig)

    # Load main dataset or use conditions as input
    if args.is_c:
        x = deepcopy(c)
    else:
        x = torch.from_numpy(np.load(args.dataset_path).astype(np.float32)).reshape(
            -1, args.image_channels, args.image_size, args.image_size
        )

    dataset = TensorDataset(x, c)
    generator = torch.Generator().manual_seed(args.seed)

    if use_val_split:
        total = len(dataset)
        train_len = int(0.75 * total)
        val_len = int(0.15 * total)
        test_len = total - train_len - val_len
        train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len], generator=generator)
        return load_data(args, train_data, val_data, test_data, generator), means, stds
    else:
        train_data, test_data = random_split(dataset, [int(0.75 * len(dataset)), len(dataset) - int(0.75 * len(dataset))], generator=generator)
        return load_data(args, train_data, None, test_data, generator), means, stds


def get_num_workers():
    # Try SLURM setting first, then fallback to system cores
    return int(os.environ.get("SLURM_CPUS_PER_TASK", max(2, os.cpu_count() // 2)))


def load_data(args, train_data, val_data, test_data, g):
    common_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': get_num_workers(),
        'pin_memory': True,
        'persistent_workers': True,
        'worker_init_fn': seed_worker
    }

    train_loader = DataLoader(train_data, shuffle=True, generator=g, **common_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, generator=g, **common_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, generator=g, **common_kwargs) if val_data is not None else None

    return train_loader, val_loader, test_loader


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


def plot_3d_scatter_comparison(decoded_images, real_images, fname):
    """
    Saves a 3D scatter plot comparing decoded and real image embeddings.
    Expects shape (B, 3) for both tensors.
    """
    decoded = decoded_images.detach().cpu().numpy()
    real = real_images.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(real[:, 0], real[:, 1], real[:, 2], c='blue', label='Real', alpha=0.6)
    ax.scatter(decoded[:, 0], decoded[:, 1], decoded[:, 2], c='red', label='Decoded', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper right')
    ax.set_title('3D Comparison of Real vs Decoded')

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def load_vqgan(args):
    """
    Loads a VQGAN or CVQGAN model checkpoint based on provided args.
    
    args: object with attributes like run_name/model_name, device, and is_c
    """
    model_path = os.path.join("../saves", args.run_name, "checkpoints", "vqgan.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=args.device, weights_only=True)
    state_dict = checkpoint["generator"]

    if all(k.startswith("_orig_mod.") for k in list(state_dict.keys())[:5]):
        print("Detected _orig_mod. prefix, removing it from keys...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = VQGAN(args).to(args.device)
    model.load_state_dict(state_dict, strict=False)

    if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        model = torch.compile(model)

    return model
