import os
import numpy as np
import random
import seaborn as sns
import platform
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from huggingface_hub import hf_hub_download

import matplotlib.pyplot as plt
from copy import deepcopy
from vqgan import VQGAN
from scipy import ndimage
from sklearn.metrics import pairwise_distances
from torch_topological.nn import CubicalComplex


"""
Various new utility functions for data loading, preprocessing, metrics, and reproducibility
"""


#############################
### REPRODUCIBILITY UTILS ###
#############################
def set_precision():
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        if major >= 8:
            torch.set_float32_matmul_precision('high')
    

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def safe_compile(model):
    """
    Try torch.compile if supported, otherwise return model unchanged.
    """
    if hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
        if platform.system() != "Windows": # Not supported on Windows
            try:
                return torch.compile(model)
            except Exception as e:
                print(f"[Warning] torch.compile failed, falling back to eager: {e}")
                return model
    return model


########################
### WARM-START UTILS ###
########################
# These utils are specific to the MTO dataset which uses a custom gamma file format
# As such they are hard-coded for 400x400 images with the specific MTO problem layout


def gamma_to_tensor(gamma):
    """
    Backend: converts extracted gamma array to a properly ordered one representing a 400x400 image
    """
    gamma_field1 = np.reshape(gamma[0:64000], (400, 160))
    gamma_field2 = np.reshape(gamma[64000:80000], (400, 40))
    gamma_field = np.concatenate((gamma_field1, gamma_field2), axis=1)
    gamma_field_full = np.flipud(np.concatenate((gamma_field, np.flip(gamma_field, 1)), axis=1))
    return gamma_field_full


def tensor_to_gamma(tensor):
    """
    Backend: Reorders a 400x400 image array to gamma array format
    """
    gamma_field_full = np.flipud(tensor)
    gamma_field = np.split(gamma_field_full, 2, axis=1)[0]
    gamma_field1, gamma_field2 = np.split(gamma_field, [160], axis=1)

    gamma_field1 = gamma_field1.flatten()
    gamma_field2 = gamma_field2.flatten()
    return np.concatenate([gamma_field1, gamma_field2])


def read_gamma(path):
    """
    Reads a gamma file and extracts the relevant field as a string
    """
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
    """
    Converts gamma file to numpy array, returning the final 400x400 image
    """
    _, field, _ = read_gamma(path)
    gamma = np.asarray(field.split('\n'), dtype=float)
    tensor = gamma_to_tensor(gamma)
    return tensor


def npy_to_gamma(tensor, path, name='gamma', template='./gamma_template'): # tensor shape (h, w) = (400, 400)
    """
    Converts numpy array to gamma file, saving to specified path. Hard-coded for 400x400 MTO images.
    """
    de1 = '86400\n(\n'
    de2 = '\n)\n'
    head, field, tail = read_gamma(template)
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


############################
### DATA RETRIEVAL UTILS ###
############################
def get_data(args, use_val_split=False):
    # If loading from Hugging Face Hub, download first
    if getattr(args, "load_from_hf", False):
        # args.conditions_path and args.dataset_path should be filenames in the repo
        repo_id = getattr(args, "repo_id", "IDEALLab/MTO-2D")
        cond_file = hf_hub_download(
            repo_id=repo_id,
            filename=getattr(args, "hf_conditions_path", "inp_paras_5666.npy"),
            repo_type="dataset"
        )
        data_file = hf_hub_download(
            repo_id=repo_id,
            filename=getattr(args, "hf_dataset_path", "gamma_5666_half.npy"),
            repo_type="dataset"
        )
    else:
        cond_file = args.conditions_path
        data_file = args.dataset_path

    # Load and normalize conditions
    c_orig = torch.from_numpy(np.load(cond_file).astype(np.float32))
    c, means, stds = normalize(c_orig)

    # Load main dataset or use conditions as input
    if args.is_c:
        x = deepcopy(c)
    else:
        x = torch.from_numpy(np.load(data_file).astype(np.float32))
        L = len(x)
        S = int(np.sqrt(np.prod(x.shape)/(L*args.image_channels)))
        x = x.reshape(L, args.image_channels, S, S)

    if args.data_fraction < 1.0:
        total_samples = len(x)
        selected_samples = int(total_samples * args.data_fraction)
        print(f"Using a fraction of the dataset: {selected_samples}/{total_samples} samples.")
        x = x[:selected_samples]
        c = c[:selected_samples]

    if not args.is_c:
        if not (x.shape[-1] == args.image_size and x.shape[-2] == args.image_size):
            print("Warning: Image size mismatch compared to input argument; resizing automatically.")
            x = F.interpolate(x, size=(args.image_size, args.image_size), mode='bicubic')
        assert len(x) == len(c), "Data and conditions length mismatch, please check dataset_path and conditions_path"

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


def load_data(args, train_data, val_data, test_data, g):
    data_fraction = getattr(args, "data_fraction", 1.0)
    common_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': 1,
        'pin_memory': True if data_fraction == 1.0 else False,
        'persistent_workers': True if data_fraction == 1.0 else False,
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
    Mirrors get_data()'s source selection (HF vs local) and data_fraction logic.
    """
    # Resolve data sources (HF or local)
    if getattr(args, "load_from_hf", False):
        repo_id = getattr(args, "repo_id", "IDEALLab/MTO-2D")
        cond_file = hf_hub_download(
            repo_id=repo_id,
            filename=getattr(args, "hf_conditions_path", "inp_paras_5666.npy"),
            repo_type="dataset"
        )
        data_file = hf_hub_download(
            repo_id=repo_id,
            filename=getattr(args, "hf_dataset_path", "gamma_5666_half.npy"),
            repo_type="dataset"
        )
    else:
        cond_file = args.conditions_path
        data_file = args.dataset_path

    # Determine dataset length
    if args.is_c:
        c = np.load(cond_file).astype(np.float32)
        dataset_length = len(c)
    else:
        x = np.load(data_file).astype(np.float32)
        dataset_length = len(x)

    # Apply data_fraction truncation like in get_data
    data_fraction = getattr(args, "data_fraction", 1.0)
    if data_fraction < 1.0:
        selected_samples = int(dataset_length * data_fraction)
        dataset_length = selected_samples

    generator = torch.Generator().manual_seed(args.seed)

    # Generate the shuffled indices
    shuffled_indices = torch.randperm(dataset_length, generator=generator).tolist()

    if use_val_split:
        train_len = int(0.75 * dataset_length)
        val_len = int(getattr(args, "val_fraction", 0.05) * dataset_length)
        train_indices = shuffled_indices[:train_len]
        val_indices = shuffled_indices[train_len:train_len + val_len]
        test_indices = shuffled_indices[train_len + val_len:]

        if args.is_t and args.train_samples < train_len:
            train_indices = train_indices[-args.train_samples:]

        return train_indices, val_indices, test_indices
    else:
        train_len = int(0.75 * dataset_length)
        train_indices = shuffled_indices[:train_len]
        test_indices = shuffled_indices[train_len:]

        return train_indices, None, test_indices


def normalize(data):
    """
    Normalize the conditions for zero mean and unit variance
    """
    means = torch.mean(data, dim=0)
    stds = torch.std(data, dim=0)
    return (data-means)/stds, list(means.numpy()), list(stds.numpy())


######################
### PLOTTING UTILS ###
######################
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
    plt.rcParams["savefig.dpi"] = dpi
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


###########################
### MODEL LOADING UTILS ###
###########################
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
    model = safe_compile(model)

    return model


###############
### METRICS ###
###############


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
    """
    MMD
    Code adapted from our CEBGAN repository at https://github.com/IDEALLab/CEBGAN_JMD_2021/blob/main/CEBGAN/src/utils/metrics.py 
    """
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    
    sigma = median_heuristic_sigma(X_test, X_test) # Always the same for the sake of comparison
    print("For MMD, using sigma:", sigma)

    mmd = np.mean(gaussian_kernel(X_gen, X_gen, sigma)) - \
          2 * np.mean(gaussian_kernel(X_gen, X_test, sigma)) + \
          np.mean(gaussian_kernel(X_test, X_test, sigma))
          
    return np.sqrt(mmd)


def topo_distance(X, Y=None, preprocess=True, normalize=True, padding=True, reduction='sum', rounding_bias=0, imageops=True, return_pair=False):
    """
    Differentiable persistence diagrams for structured data, such as images. 
    https://pytorch-topological.readthedocs.io/en/latest/nn.html#torch_topological.nn.CubicalComplex
    """
    c = CubicalComplex()

    if not torch.is_tensor(X):
        X = torch.tensor(X)
    if Y is not None and not torch.is_tensor(Y):
        Y = torch.tensor(Y)
    if len(X.shape) == 2:
        X = X.unsqueeze(0)
    if len(X.shape) == 4:
        X = X.squeeze(1)
    if Y is not None:
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

    if Y is not None:
        for idx, (x, y) in enumerate(zip(X, Y)):
            # Apply morphological operations to clean up noisy designs if imageops is True
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
                losses[idx] = np.abs(len(cx[0].pairing) - len(cy[0].pairing))/len(cy[0].pairing)
            elif return_pair:
                losses[idx, 0] = len(cx[0].pairing)
                losses[idx, 1] = len(cy[0].pairing)
            else:
                losses[idx] = np.abs(len(cx[0].pairing) - len(cy[0].pairing))
    
    else:
        for idx, x in enumerate(X):
            # Apply morphological operations to clean up noisy designs if imageops is True
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


def rdiv(X_train, X_gen):
    """
    R-Div: full pairwise distances without sampling
    Code adapted from our CEBGAN repository at https://github.com/IDEALLab/CEBGAN_JMD_2021/blob/main/CEBGAN/src/utils/metrics.py 
    """
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    
    # Compute all pairwise distances
    train_dists = pairwise_distances(X_train, X_train)
    gen_dists = pairwise_distances(X_gen, X_gen)
    
    # Get upper triangular part (exclude diagonal and duplicates)
    train_div = np.mean(train_dists[np.triu_indices_from(train_dists, k=1)])
    gen_div = np.mean(gen_dists[np.triu_indices_from(gen_dists, k=1)])
    
    return gen_div / train_div


def vf_loss(input, target, N=1, d=1):
    """
    Volume Fraction Loss. N specifies how many equal-sized quadrants to split the data into (default 1 i.e. no splitting)
    """
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