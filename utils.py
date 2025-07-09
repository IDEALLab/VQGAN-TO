import os
import json
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
from scipy import ndimage
from sklearn.metrics import pairwise_distances
# from torch_topological.nn import SummaryStatisticLoss, CubicalComplex
from paretoset import paretoset


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

        if args.is_t and args.train_samples < train_len:
            train_data = torch.utils.data.Subset(train_data, list(range(train_len - args.train_samples, train_len)))

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


def mirror(data, dim=-1, reshape=None, difference=False, mode='bicubic'):
    while len(data.shape) < 4:
        data = data.unsqueeze(0)
    new = torch.cat((data, torch.flip(data, (dim,))), dim)
    if reshape is not None:
        if difference:
            new = torch.clamp(F.interpolate(new, reshape, mode=mode), -1, 1)
        else:
            new = F.interpolate(new, reshape, mode=mode)
    return new


def plot_data(data, titles, ranges, fname=None, dpi=100, mirror_image=False, cmap=None, cbar=True, fontsize=20, reshape_size=(400, 400), mode="bicubic"):
    L = len(titles)
    fig, axs = plt.subplots(1, L, figsize=(int(5*L), 4))
    [ax.axes.xaxis.set_visible(False) for ax in axs]
    [ax.axes.yaxis.set_visible(False) for ax in axs]
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams["figure.dpi"] = dpi

    for idx, (figure, title, current_range) in enumerate(zip(data, titles, ranges)):
        if mirror_image:
            figure = np.array(mirror(torch.tensor(figure), reshape=reshape_size, difference=(title=="Difference"), mode=mode))[0]
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

def process_state_dict(state_dict):
    if any("_orig_mod." in k for k in list(state_dict.keys())[:5]):
        print("Detected '_orig_mod.' in keys, removing all occurrences from keys...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict

def load_vqgan(args):
    """
    Loads a VQGAN or CVQGAN model checkpoint based on provided args.
    
    args: object with attributes like run_name/model_name, device, and is_c
    """
    model_path = os.path.join("../saves", args.run_name, "checkpoints", "vqgan.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=args.device, weights_only=True)
    state_dict = process_state_dict(checkpoint["generator"])

    model = VQGAN(args).to(args.device)
    model.load_state_dict(state_dict, strict=False)

    if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        model = torch.compile(model)

    return model

###########
# METRICS #
###########

def generate_pareto(data, sense=["max", "min"]):
    return paretoset(data, sense=sense)

def scale_pareto(data, pG, pB, sense=["max", "min"]):
    for i in range(data.shape[1]):
        multi = (1 if sense[i] == "min" else -1)
        data[:, i] = (data[:, i] - pG[i])/(multi*(pB[i] - pG[i])) - 0.5*(multi - 1)
    return data

def HD(pareto, sense=["max", "min"]):
    if (not "min" in sense) or (not "max" in sense):
        p1 = [0,1]
        p2 = [1,0]
    else:
        p1 = [0,0]
        p2 = [1,1]
    sorted = pareto.sort(0).values
    sorted = torch.vstack((torch.tensor([p1]), sorted))
    sorted = torch.vstack((sorted, torch.tensor([p2])))
    area = 0
    for _, item in enumerate(zip(sorted[0:-1], sorted[1:])):
        if sense[1] == "max":
            area += (item[1][0]-item[0][0])*item[1][1]
        else:
            area += (item[1][0]-item[0][0])*(1-item[1][1])
    return 1 - area

def Dominated_Area(sp):
    sp = sp[np.argsort(sp[:,0])]
    sp = torch.cat((sp, torch.tensor([[1, sp[-1][1]]])))

    area = 0
    for _, (p0, p1) in enumerate(zip(sp[0:-1], sp[1:])):
        area += (1-p0[1])*(p1[0]-p0[0]) + 0.5*(p1[0]-p0[0])*(p0[1]-p1[1])
    return area

def KPS(pareto):
    return torch.tensor([float(torch.max(p)-torch.min(p)) for p in pareto.T])

# MMD
# Code adapted from our CEBGAN repository at https://github.com/IDEALLab/CEBGAN_JMD_2021/blob/main/CEBGAN/src/utils/metrics.py 
def gaussian_kernel(X, Y, sigma=2.0):
    beta = 1. / (2. * sigma**2)
    dist = pairwise_distances(X, Y)
    s = beta * dist.flatten()
    return np.exp(-s)

def MMD(X_gen, X_test):
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
      
    mmd = np.mean(gaussian_kernel(X_gen, X_gen)) - \
            2 * np.mean(gaussian_kernel(X_gen, X_test)) + \
            np.mean(gaussian_kernel(X_test, X_test))
            
    return np.sqrt(mmd)

# Topological distance: absolute value of difference between Betti numbers for summary statistic loss
def topo_distance(X, Y=None, preprocess=True, normalize=True, padding=True, reduction='sum', rounding_bias=0, imageops=True, return_pair=False):
    c = None # CubicalComplex()
    # topo_loss = SummaryStatisticLoss()

    if not torch.is_tensor(X): X = torch.tensor(X)
    if not Y is None and not torch.is_tensor(Y): Y = torch.tensor(Y)
    if len(X.shape) == 2:
        X = X.unsqueeze(0)
    if len(X.shape) == 4:
        X = X.squeeze(1)
    if not Y is None:
        if len(Y.shape) == 2:
            Y = Y.unsqueeze(0)
        if len(Y.shape) == 4:
            Y = Y.squeeze(1)
        assert len(X) == len(Y)

    if preprocess:
        X = torch.round(torch.clamp(X, 0, 1) + rounding_bias)
        if not Y is None: 
            Y = torch.round(torch.clamp(Y, 0, 1) + rounding_bias)

    losses = torch.zeros(len(X), 1+int(return_pair))

    if not Y is None:
        for idx, (x, y) in enumerate(zip(X, Y)):

            if imageops:
                x = ndimage.binary_opening(x, structure=np.ones((3,3))).astype(np.int32)
                y = ndimage.binary_opening(y, structure=np.ones((3,3))).astype(np.int32)
                x = ndimage.binary_closing(x, structure=np.ones((3,3)), border_value=1-int(padding)).astype(np.int32)
                y = ndimage.binary_closing(y, structure=np.ones((3,3)), border_value=1-int(padding)).astype(np.int32)
                x = torch.tensor(x)
                y = torch.tensor(y)

            cx = c(x)
            cy = c(y)

            if normalize:
                # losses[idx] = topo_loss(cx, cy)/topo_loss(cy)
                losses[idx] = np.abs(len(cx[0].pairing) - len(cy[0].pairing))/len(cy[0].pairing)
            elif return_pair:
                losses[idx, 0] = len(cx[0].pairing)
                losses[idx, 1] = len(cy[0].pairing)
            else:
                # losses[idx] = topo_loss(cx, cy)
                losses[idx] = np.abs(len(cx[0].pairing) - len(cy[0].pairing))
    
    else:
        for idx, x in enumerate(X):
            if imageops:
                x = ndimage.binary_opening(x, structure=np.ones((3,3))).astype(np.int32)
                x = ndimage.binary_closing(x, structure=np.ones((3,3)), border_value=1-int(padding)).astype(np.int32)
                x = torch.tensor(x)

            cx = c(x)
            losses[idx] = len(cx[0].pairing)

    if reduction == 'mean':
        return torch.mean(losses)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        return losses

# R-Div
# Code adapted from our CEBGAN repository at https://github.com/IDEALLab/CEBGAN_JMD_2021/blob/main/CEBGAN/src/utils/metrics.py 
def variance(X):
    cov = np.cov(X.T)
    var = np.trace(cov)/cov.shape[0]
#    var = np.mean(np.var(X, axis=0))
#    var = np.linalg.det(cov)
#    var = var**(1./cov.shape[0])
    return var

def rdiv(X_train, X_gen):
    ''' Relative div '''
    X_train = np.squeeze(X_train)
#    train_div = np.sum(np.var(X_train, axis=0))
#    gen_div = np.sum(np.var(X_gen, axis=0))
    X_train = X_train.reshape((X_train.shape[0], -1))
    train_div = variance(X_train)
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    gen_div = variance(X_gen)
#    n = 100
#    gen_div = train_div = 0
#    for i in range(n):
#        a, b = np.random.choice(X_gen.shape[0], 2, replace=False)
#        gen_div += np.linalg.norm(X_gen[a] - X_gen[b])
#        c, d = np.random.choice(X_train.shape[0], 2, replace=False)
#        train_div += np.linalg.norm(X_train[c] - X_train[d])
    rdiv = gen_div/train_div
    return rdiv

# Volume Fraction Loss. N specifies how many equal-sized quadrants to split the data into (default 1 i.e. no splitting)
def vf_loss(input, target, N=1, d=1):
    loss = []
    input = np.squeeze(input)
    target = np.squeeze(target)

    for _, (i, t) in enumerate(zip(input, target)):
        i_split = np.array([np.vsplit(a, N) for a in np.hsplit(i, N)])
        t_split = np.array([np.vsplit(a, N) for a in np.hsplit(t, N)])
        i_split = i_split.reshape(-1, i_split.shape[-2], i_split.shape[-1])
        t_split = t_split.reshape(-1, t_split.shape[-2], t_split.shape[-1])
        temp_loss = []

        for _, (x, y) in enumerate(zip(i_split, t_split)):
            sx = np.sum(x)/np.product(x.shape)
            sy = np.sum(y)/np.product(y.shape)
            temp_loss.append(np.abs(sx - sy)**d)

        loss.append(np.mean(temp_loss))

    return np.mean(loss)