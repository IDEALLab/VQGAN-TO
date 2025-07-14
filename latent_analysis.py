import os
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

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
        self.device = args.device
        self.vqgan = load_vqgan(args).eval().to(self.device)
        self.run()

    def run(self):
        args = self.args
        (_, _, test_dataloader), _, _ = get_data(args)

        interp_mses, interp_ivs, interp_dvs = [], [], []
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

                # Pick two random indices for interpolation
                idx1, idx2 = np.random.choice(len(zs), size=2, replace=False)
                o1, o2 = outputs[idx1], outputs[idx2]
                s1, s2 = zs[idx1], zs[idx2]
                diff = s2 - s1

                mse_list, iv_list, dv_list = [], [], []
                for step in range(num_steps + 1):
                    alpha = step / num_steps
                    new = torch.tensor(s1 + alpha * diff).unsqueeze(0).to(self.device)
                    new_q, _, _ = self.vqgan.codebook(new)
                    new_decode = self.vqgan.decode(new_q).detach().cpu().numpy()[0]
                    new_decode = np.clip(new_decode, 0, 1)

                    if step == 0:
                        mse0 = np.sum((o1 - new_decode) ** 2) + np.sum((o2 - new_decode) ** 2)
                    mse = (np.sum((o1 - new_decode) ** 2) + np.sum((o2 - new_decode) ** 2)) / mse0
                    mse_list.append(mse)
                    iv_list.append(intermediate_value_loss(new_decode))
                    if step > 0:
                        dv_list.append(np.abs(new_decode - prev_decode).mean())
                    prev_decode = deepcopy(new_decode)

                interp_mses.append(mse_list)
                interp_ivs.append(iv_list)
                interp_dvs.append(dv_list)

                # Topological perturbation on a single random latent sample
                idx = np.random.choice(len(zs))
                q = zs[idx]
                q_tensor = torch.tensor(q).unsqueeze(0).to(self.device)
                recon = self.vqgan.decode(q_tensor).detach().cpu().numpy()[0]

                altered_accum = []
                for alt in alts:
                    q_alt = torch.tensor(q).unsqueeze(0).to(self.device)
                    q_alt = change_specific(q_alt, alt[0], alt[1])
                    q_alt, _, _ = self.vqgan.codebook(q_alt)
                    altered = self.vqgan.decode(q_alt).detach().cpu().numpy()[0]
                    altered_accum.append(altered)

                altered_mean = np.mean(np.stack(altered_accum), axis=0)

                stats = [
                    topo_distance(recon, normalize=False),
                    topo_distance(altered_mean, normalize=False),
                    topo_distance(1 - recon, padding=False, normalize=False),
                    topo_distance(1 - altered_mean, padding=False, normalize=False)
                ]

                print(f"Batch {batch_idx}: Solid Δ = {stats[1] - stats[0]:.3f}, Fluid Δ = {stats[3] - stats[2]:.3f}")
                topo_accum.append(stats)

        np.save(os.path.join(self.latent_results_dir, 'interp_mses.npy'), np.array(interp_mses))
        np.save(os.path.join(self.latent_results_dir, 'interp_ivs.npy'), np.array(interp_ivs))
        np.save(os.path.join(self.latent_results_dir, 'interp_dvs.npy'), np.array(interp_dvs))
        np.save(os.path.join(self.latent_results_dir, 'topo_info_all.npy'), np.array(topo_accum))
        np.save(os.path.join(self.latent_results_dir, 'topo_info_mean.npy'), np.mean(np.array(topo_accum), axis=0))
        np.save(os.path.join(self.latent_results_dir, 'topo_alts.npy'), alts.cpu().numpy())


if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Latent Analysis Arguments")
    LatentAnalysis(args)
