import os
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Subset, DataLoader

from utils import get_data, load_vqgan, set_precision, set_all_seeds, topo_distance
from args import get_args, load_args, print_args


def intermediate_value_loss(data, b1=0.2, b2=0.8):
    data = np.ceil(np.clip((data - b1), 0, 1) * np.clip((b2 - data), 0, 1))
    return data.mean()


def change_specific(vec, p1, p2):
    a = vec[0, :, p1[0], p1[1]].clone()
    b = vec[0, :, p2[0], p2[1]].clone()
    vec[0, :, p2[0], p2[1]] = a
    vec[0, :, p1[0], p1[1]] = b
    return vec


def change_random(vec, num_changes):
    for _ in range(num_changes):
        h1, w1 = np.random.randint(vec.shape[2]), np.random.randint(vec.shape[3])
        h2, w2 = h1, w1
        while h1 == h2 and w1 == w2:
            h2, w2 = np.random.randint(vec.shape[2]), np.random.randint(vec.shape[3])
        vec = change_specific(vec, [h1, w1], [h2, w2])
    return vec


def flip_pixel(vec, h, w):
    vec[:, :, h, w] *= -1
    return vec


class LatentAnalysis:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)

        self.eval_dir = os.path.join("../evals", args.model_name)
        self.latent_results_dir = os.path.join(self.eval_dir, "latent_results")
        os.makedirs(self.latent_results_dir, exist_ok=True)

        args = load_args(args)
        self.args = args
        self.args.batch_size = 2  # IMPORTANT: Set batch size to 2 for latent analysis (hard-coded)
        self.device = args.device
        self.vqgan = load_vqgan(args).eval().to(self.device)
        self.run()

    def run(self):
        args = self.args
        # Load all test data
        (_, _, test_dataloader_full), _, _ = get_data(args, use_val_split=True)
        test_dataset_full = test_dataloader_full.dataset

        single_fluid_path = "../data/single_fluid.npy"

        if os.path.exists(single_fluid_path):
            print(f"Loading cached single-fluid indices from {single_fluid_path}")
            valid_indices = np.load(single_fluid_path)
        else:
            print("Computing single-fluid indices...")
            valid_indices = []
            for i in tqdm(range(len(test_dataset_full)), desc="Checking fluid segments (reconstructed)"):
                x, _ = test_dataset_full[i]
                x = x.unsqueeze(0).to(args.device)
                with torch.no_grad():
                    x_recon, _, _ = self.vqgan(x)
                    x_recon = x_recon.clamp(0, 1).detach().cpu().numpy()[0]
                num_segments = topo_distance(1 - x_recon, padding=False, normalize=False, imageops=False, rounding_bias=0.3)
                if num_segments == 1:
                    valid_indices.append(i)
            valid_indices = np.array(valid_indices)
            os.makedirs("../data", exist_ok=True)
            np.save(single_fluid_path, valid_indices)
            print(f"Saved {len(valid_indices)} single-fluid samples.")

        # Create subset and DataLoader
        subset = Subset(test_dataset_full, valid_indices.tolist())
        test_dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

        interp_topos, interp_ivs, interp_dvs = [], [], []
        topo_accum = []
        qs_dim = args.latent_dim
        num_alts = 64
        alts = torch.randint(high=qs_dim, size=(num_alts, 2, 2))
        num_steps = 100

        with torch.no_grad():
            for batch_idx, (imgs, _) in enumerate(tqdm(test_dataloader, desc="Batches")):
                imgs = imgs.to(self.device)
                decoded_images, _, _ = self.vqgan(imgs)
                decoded_images = decoded_images.clamp(0, 1)
                encoded, _, _ = self.vqgan.encode(imgs)
                zs = encoded.detach().cpu().numpy()
                outputs = decoded_images.cpu().numpy()

                if len(zs) == 2:
                    interp_topos.append([])
                    s1, s2 = zs[0], zs[1]
                    diff = s2 - s1

                    iv_list, dv_list = [], []
                    for step in range(num_steps + 1):
                        alpha = step / num_steps
                        new = torch.tensor(s1 + alpha * diff).unsqueeze(0).to(self.device)

                        if step in [0, num_steps]:
                            new_decode = self.vqgan.decode(new).detach().cpu().numpy()[0]  # Bypass quantization at endpoints
                        else:
                            new_q, _, _ = self.vqgan.codebook(new)
                            new_decode = self.vqgan.decode(new_q).detach().cpu().numpy()[0]

                        new_decode = np.clip(new_decode, 0, 1)
                        dist = topo_distance(1 - new_decode, padding=False, normalize=False, imageops=False, rounding_bias=0.3)
                        dist = dist.item()

                        iv_list.append(intermediate_value_loss(new_decode))
                        if step > 0:
                            dv_list.append(np.abs(new_decode - prev_decode).mean())
                        prev_decode = deepcopy(new_decode)
                        interp_topos[-1].append(dist)

                    interp_ivs.append(iv_list)
                    interp_dvs.append(dv_list)

                # Topological perturbation on a single random latent sample
                q = torch.tensor(zs[0]).to(self.device).unsqueeze(0)
                recon = self.vqgan.decode(q).detach().cpu().numpy()[0]
                q_alt = torch.clone(q).to(self.device)

                for alt in alts:
                    q_alt = change_specific(q_alt, alt[0], alt[1])
                
                altered = self.vqgan.decode(q_alt).detach().cpu().numpy()[0]

                stats = [
                    topo_distance(recon, normalize=False), # Num solid segments in original
                    topo_distance(altered, normalize=False), # Num solid segments in altered
                    topo_distance(1 - recon, padding=False, normalize=False), # Num fluid segments in original
                    topo_distance(1 - altered, padding=False, normalize=False) # Num fluid segments in altered
                ]

                topo_accum.append(stats)

        np.save(os.path.join(self.latent_results_dir, 'interp_topos.npy'), np.array(interp_topos))
        np.save(os.path.join(self.latent_results_dir, 'interp_ivs.npy'), np.array(interp_ivs))
        np.save(os.path.join(self.latent_results_dir, 'interp_dvs.npy'), np.array(interp_dvs))
        np.save(os.path.join(self.latent_results_dir, 'topo_info_all.npy'), np.array(topo_accum))
        np.save(os.path.join(self.latent_results_dir, 'topo_alts.npy'), alts.cpu().numpy())


if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Latent Analysis Arguments")
    LatentAnalysis(args)
