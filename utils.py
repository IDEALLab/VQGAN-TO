import os
import json
import numpy as np
import random
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor

import matplotlib.pyplot as plt
from copy import deepcopy
from vqgan import VQGAN
from scipy import ndimage
from sklearn.metrics import pairwise_distances
from torch_topological.nn import CubicalComplex #, SummaryStatisticLoss
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

########################
### WARM START UTILS ###
########################

def gamma_to_tensor(gamma):
    gamma_field1 = np.reshape(gamma[0:64000], (400, 160))
    gamma_field2 = np.reshape(gamma[64000:80000], (400, 40))
    gamma_field = np.concatenate((gamma_field1, gamma_field2), axis=1)
    gamma_field_full = np.flipud(np.concatenate((gamma_field, np.flip(gamma_field, 1)), axis=1))
    return gamma_field_full

def tensor_to_gamma(tensor):
    gamma_field_full = np.flipud(tensor)
    gamma_field = np.split(gamma_field_full, 2, axis=1)[0]
    gamma_field1, gamma_field2 = np.split(gamma_field, [160], axis=1)

    gamma_field1 = gamma_field1.flatten()
    gamma_field2 = gamma_field2.flatten()
    return np.concatenate([gamma_field1, gamma_field2])

def read_gamma(path):
    de1 = '86400\n(\n'
    de2 = '\n)\n'
    with open(path, 'r') as f:
        file_content = f.read()

    # Normalize line endings to Unix-style
    file_content = file_content.replace('\r\n', '\n').replace('\r', '\n')

    if de1 not in file_content or de2 not in file_content:
        raise ValueError(f"Invalid gamma format or corrupted file at: {path}")

    head, body = file_content.split(de1)
    body, tail = body.split(de2)
    return head, body, tail
    
def gamma_to_npy(path):
    _, field, _ = read_gamma(path)
    gamma = np.asarray(field.split('\n'), dtype=float)
    tensor = gamma_to_tensor(gamma)
    return tensor

def npy_to_gamma(tensor, path, name='gamma', template='../data/gamma_template'): # tensor shape (h, w) = (400, 400)
    de1 = '86400\n(\n'
    de2 = '\n)\n'
    if template is None:
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'templates'
        )
        template = os.path.join(template_path, 'gamma_template')
    head, field, tail = read_gamma(template) # location    "1922";
    head = head.replace('location    "200";', 'location    "0";')
    gamma = np.asarray(field.split('\n'), dtype=float)
    gamma[:80000] = tensor_to_gamma(tensor)

    os.makedirs(path, exist_ok=True)
    np.savetxt(
        os.path.join(path, name), 
        gamma, '%.2e', 
        header=''.join([head, de1[:-1]]), 
        footer=''.join([de2[1:], tail]),
        comments=''
        )

########################
### WARM START UTILS ###
########################


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
        val_len = int(args.val_fraction * total)
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

def get_data_split_indices(args, use_val_split=False):
    """
    Return train/val/test indices corresponding to the original dataset,
    matching the behavior of torch.utils.data.random_split with a fixed seed.
    """
    # Load dataset size
    if args.is_c:
        c = np.load(args.conditions_path).astype(np.float32)
        dataset_length = len(c)
    else:
        x = np.load(args.dataset_path).astype(np.float32)
        dataset_length = len(x)

    generator = torch.Generator().manual_seed(args.seed)

    # Generate the shuffled indices
    shuffled_indices = torch.randperm(dataset_length, generator=generator).tolist()

    if use_val_split:
        train_len = int(0.75 * dataset_length)
        val_len = int(args.val_fraction * dataset_length)
        test_len = dataset_length - train_len - val_len

        train_indices = shuffled_indices[:train_len]
        val_indices = shuffled_indices[train_len:train_len + val_len]
        test_indices = shuffled_indices[train_len + val_len:]

        if args.is_t and args.train_samples < train_len:
            train_indices = train_indices[-args.train_samples:]

        return train_indices, val_indices, test_indices
    else:
        train_len = int(0.75 * dataset_length)
        test_len = dataset_length - train_len

        train_indices = shuffled_indices[:train_len]
        test_indices = shuffled_indices[train_len:]

        return train_indices, None, test_indices


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

def generate_pareto(data, sense=["min", "min"]):
    return paretoset(data, sense=sense)

def scale_pareto(data, pG, pB, sense=["min", "min"]):
    for i in range(data.shape[1]):
        multi = (1 if sense[i] == "min" else -1)
        data[:, i] = (data[:, i] - pG[i])/(multi*(pB[i] - pG[i])) - 0.5*(multi - 1)
    return data

def HD(pareto, sense=["min", "min"]):
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
def gaussian_kernel(X, Y, sigma):
    beta = 1. / (2. * sigma**2)
    dists = pairwise_distances(X, Y)  # shape (n_x, n_y)
    return np.exp(-beta * dists**2)

def median_heuristic_sigma(X, Y):
    Z = np.concatenate([X, Y], axis=0)
    dists = pairwise_distances(Z, Z)
    median = np.median(dists)
    return median / np.sqrt(2)

def MMD(X_gen, X_test):
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    
    sigma = median_heuristic_sigma(X_test, X_test) # Always the same for the sake of comparison
    print("For MMD, using sigma:", sigma)

    mmd = np.mean(gaussian_kernel(X_gen, X_gen, sigma)) - \
          2 * np.mean(gaussian_kernel(X_gen, X_test, sigma)) + \
          np.mean(gaussian_kernel(X_test, X_test, sigma))
          
    return np.sqrt(mmd)

# ---------------------------
# Inception feature extractor
# ---------------------------
@torch.no_grad()
def compute_inception_features(
    images_np: np.ndarray,           # shape (N, 1 or 3, H, W), values in [0,1]
    device: str = None,
    batch_size: int = 64,
) -> np.ndarray:                     # returns (N, 2048)
    """
    Extract Inception-V3 avgpool (2048-D) features for KID.
    - Accepts grayscale by repeating to 3 channels.
    - Resizes to 299x299 and applies ImageNet normalization.
    """
    assert images_np.ndim == 4, "Expected images_np of shape (N,C,H,W) in [0,1]"
    N, C, H, W = images_np.shape

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.from_numpy(images_np).float().to(device)  # [0,1]
    if C == 1:
        x = x.repeat(1, 3, 1, 1)

    # Load weights & get normalization robustly across torchvision versions
    try:
        weights = Inception_V3_Weights.IMAGENET1K_V1
    except Exception:
        weights = Inception_V3_Weights.DEFAULT

    # Mean and STD of ImageNet for normalization
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std  = [0.229, 0.224, 0.225]

    try:
        from torchvision.transforms import Normalize
        tfm = weights.transforms()
        if hasattr(tfm, "transforms"):
            for t in tfm.transforms:
                if isinstance(t, Normalize):
                    normalize_mean = list(t.mean)
                    normalize_std  = list(t.std)
                    break
        elif hasattr(tfm, "normalize"):
            normalize_mean = list(tfm.normalize.mean)
            normalize_std  = list(tfm.normalize.std)
    except Exception:
        pass

    mean = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(normalize_std,  device=device).view(1, 3, 1, 1)

    model = inception_v3(weights=weights, aux_logits=True).to(device).eval()
    extractor = create_feature_extractor(model, return_nodes={"avgpool": "feat"})

    feats = []
    for i in range(0, N, batch_size):
        xb = x[i:i+batch_size]
        xb = F.interpolate(xb, size=(299, 299), mode="bilinear", align_corners=False)
        xb = (xb - mean) / std
        out = extractor(xb)["feat"]              # (B, 2048, 1, 1)
        feats.append(out.flatten(1).cpu().numpy())

    return np.concatenate(feats, axis=0)         # (N, 2048)

# ---------------------------
# KID core (unbiased MMD^2 with polynomial kernel)
# ---------------------------
def _poly_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3, gamma: float | None = None, coef0: float = 1.0):
    """
    Polynomial kernel (used by KID): k(x,y) = (gamma * x^T y + coef0)^degree
    Default gamma = 1/d, where d is the feature dimension.
    """
    d = X.shape[1]
    if gamma is None:
        gamma = 1.0 / d
    return (gamma * (X @ Y.T) + coef0) ** degree

def _mmd2_unbiased_from_kernels(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    """
    Unbiased U-statistic estimate of MMD^2 given kernel matrices.
    Diagonals are excluded in Kxx and Kyy.
    """
    # Work on copies so we can zero diagonals safely if caller reuses matrices.
    Kxx = Kxx.copy()
    Kyy = Kyy.copy()
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    n = Kxx.shape[0]
    m = Kyy.shape[0]
    mmd2 = (Kxx.sum() / (n * (n - 1))
          + Kyy.sum() / (m * (m - 1))
          - 2.0 * Kxy.mean())
    return float(max(mmd2, 0.0))  # numerical safety

def KID_from_features(
    F_gen: np.ndarray,
    F_real: np.ndarray,
    n_subsets: int = 100,
    subset_size: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Compute KID as the mean ± std of unbiased MMD^2 across random subsets.
    Returns (kid_mean, kid_std).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = min(len(F_gen), subset_size)
    m = min(len(F_real), subset_size)
    vals = []
    for _ in range(n_subsets):
        idx_g = rng.choice(len(F_gen), n, replace=False)
        idx_r = rng.choice(len(F_real), m, replace=False)
        X = F_gen[idx_g]
        Y = F_real[idx_r]
        Kxx = _poly_kernel(X, X)  # degree=3, gamma=1/d, coef0=1
        Kyy = _poly_kernel(Y, Y)
        Kxy = _poly_kernel(X, Y)
        vals.append(_mmd2_unbiased_from_kernels(Kxx, Kyy, Kxy))
    vals = np.asarray(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1))

# ---------------------------
# Convenience wrapper: images -> features -> KID
# ---------------------------
def KID(
    X_gen: np.ndarray,   # (N,C,H,W) in [0,1]
    X_real: np.ndarray,  # (M,C,H,W) in [0,1]
    device: str | None = None,
    batch_size: int = 64,
    n_subsets: int = 100,
    subset_size: int = 1000,
    return_std: bool = True,
):
    """
    Computes KID (mean over subsets). Set return_std=True to also get std.
    """
    F_gen  = compute_inception_features(X_gen,  device=device, batch_size=batch_size)
    F_real = compute_inception_features(X_real, device=device, batch_size=batch_size)
    kid_mean, kid_std = KID_from_features(F_gen, F_real, n_subsets=n_subsets, subset_size=subset_size)
    return (kid_mean, kid_std) if return_std else kid_mean


# Topological distance: absolute value of difference between Betti numbers for summary statistic loss
def topo_distance(X, Y=None, preprocess=True, normalize=True, padding=True, reduction='sum', rounding_bias=0, imageops=True, return_pair=False):
    c = CubicalComplex()
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
        threshold = 0.5 - rounding_bias
        X = (X > threshold).float()
        if Y is not None:
            Y = (Y > threshold).float()

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
def rdiv(X_train, X_gen):
    """Full pairwise distances without sampling"""
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    
    # Compute all pairwise distances
    train_dists = pairwise_distances(X_train, X_train)
    gen_dists = pairwise_distances(X_gen, X_gen)
    
    # Get upper triangular part (exclude diagonal and duplicates)
    train_div = np.mean(train_dists[np.triu_indices_from(train_dists, k=1)])
    gen_div = np.mean(gen_dists[np.triu_indices_from(gen_dists, k=1)])
    
    return gen_div / train_div

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