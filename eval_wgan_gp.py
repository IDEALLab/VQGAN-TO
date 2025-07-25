import os
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
from scipy.ndimage import label

from wgan_gp import Generator, VQGANLatentWrapper
from utils import get_data, set_precision, set_all_seeds, plot_data, process_state_dict, MMD, rdiv, get_data_split_indices, npy_to_gamma, mirror
from args import get_args, load_args, print_args


class EvalWGAN_GP:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)

        self.eval_dir = os.path.join("../evals", args.gan_name)
        self.results_dir = os.path.join(self.eval_dir, "results")
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        args = load_args(args)
        self.args = args

        self.model = Generator(args).to(device=args.device)
        self.vq_wrapper = VQGANLatentWrapper(args).to(device=args.device)

        ckpt_path = os.path.join("../saves", args.run_name, "checkpoints", "generator.pt")
        checkpoint = process_state_dict(torch.load(ckpt_path, map_location=args.device, weights_only=True))
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        self.vq_wrapper.eval()

        self.evaluate()

    def evaluate(self):
        (dataloader, _, test_dataloader), means, stds = get_data(self.args, use_val_split=True)
        _, _, test_indices = get_data_split_indices(self.args, use_val_split=True)
        orig_indices = np.load("../data/new/nonv/index_5666.npy")

        all_generated = []
        all_real_eval = []
        all_real_train = []

        all_gen_vfs = []
        all_real_vfs = []
        all_volume_mae = []

        component_counts = []
        solid_counts = []
        solid_counts_real = []

        vf_mean = means[2]
        vf_std = stds[2]

        print("Loading training data for evaluation...")
        with torch.no_grad():
            for imgs, cond in dataloader:
                imgs = imgs.to(self.args.device, non_blocking=True)
                imgs = mirror(imgs, reshape=(400, 400)).clamp(0, 1).cpu().numpy()
                all_real_train.append(imgs)
        print("Completed loading training data for evaluation.")

        with torch.no_grad():
            for i, (imgs, cond) in enumerate(tqdm(test_dataloader, desc="Evaluating WGAN-GP")):
                sample_imgs = imgs.to(self.args.device)
                sample_c = cond.to(self.args.device)
                if self.args.gan_use_cvq:
                    sample_c = self.vq_wrapper.c_encode(sample_c)
                    sample_c = sample_c.view(sample_c.shape[0], -1)

                # Get real latents and reconstructions
                real_latents = self.vq_wrapper.encode(sample_imgs)
                recon_imgs = self.vq_wrapper.decode(real_latents).clamp(0, 1)

                # Generate fake samples
                z = torch.randn(sample_imgs.shape[0], self.args.latent_dim, device=self.args.device)
                gen_latents = self.model(z, sample_c)
                fake_imgs = self.vq_wrapper.decode(gen_latents).clamp(0, 1)
                
                # Apply mirroring to match transformer data handling
                full_sample = mirror(fake_imgs, reshape=(400, 400)).clamp(0, 1).cpu().numpy()
                recon = mirror(recon_imgs, reshape=(400, 400)).clamp(0, 1).cpu().numpy()
                original = mirror(sample_imgs, reshape=(400, 400)).clamp(0, 1).cpu().numpy()

                # Save individual gamma files with original indexing (like transformer)
                for j in range(full_sample.shape[0]):
                    test_index = i * self.args.batch_size + j
                    original_index = orig_indices[test_indices[test_index]]
                    gamma_tensor = full_sample[j, 0]
                    npy_to_gamma(gamma_tensor, path=self.results_dir, name=f"gamma_{original_index}")

                all_generated.append(full_sample)
                all_real_eval.append(original)

                # Compute VFs and MAE (using mirrored data)
                gen_vfs = full_sample.reshape(full_sample.shape[0], -1).mean(axis=1)
                ref_vfs = original.reshape(original.shape[0], -1).mean(axis=1)

                mae = np.abs(gen_vfs - ref_vfs)
                all_volume_mae.extend(mae)
                all_gen_vfs.extend(gen_vfs)
                all_real_vfs.extend(ref_vfs)

                # Count disconnected fluid segments (using mirrored data)
                binary_samples = (full_sample > 0.5).astype(np.uint8)
                binary_samples_real = (original > 0.5).astype(np.uint8)
                for b in range(binary_samples.shape[0]):
                    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
                    _, num_fluid = label(binary_samples[b, 0], structure=structure)
                    _, num_solid = label(1 - binary_samples[b, 0], structure=structure)
                    component_counts.append(num_fluid)
                    solid_counts.append(num_solid)
                for b in range(binary_samples_real.shape[0]):
                    _, num_solid_real = label(1 - binary_samples_real[b, 0], structure=structure)
                    solid_counts_real.append(num_solid_real)

                # Append combined images for later saving (like transformer)
                combined = np.stack([original, recon, full_sample], axis=1)

                if i == 0:
                    all_samples = combined
                else:
                    all_samples = np.concatenate([all_samples, combined], axis=0)

        # Save full generated samples (matching transformer structure)
        all_generated = np.concatenate(all_generated, axis=0)
        all_real_eval = np.concatenate(all_real_eval, axis=0)
        all_real_train = np.concatenate(all_real_train, axis=0)
        solid_counts = np.array(solid_counts)
        solid_counts_real = np.array(solid_counts_real)

        # Save data matching transformer format
        # np.save(os.path.join(self.eval_dir, "generated.npy"), all_generated)
        np.save(os.path.join(self.eval_dir, "vfs_gen"), np.array(all_gen_vfs))
        np.save(os.path.join(self.eval_dir, "vfs_real.npy"), np.array(all_real_vfs))
        np.save(os.path.join(self.eval_dir, "vfs_mae.npy"), np.array(all_volume_mae))

        # Summary metrics
        print("Calculating MMD")
        mmd = MMD(all_generated, all_real_eval)
        print("Calculating R-Div")
        r_div = rdiv(all_real_train, all_generated)
        print("Calculating remainder of metrics.")
        vf_mae = np.mean(all_volume_mae)
        avg_disconnected = np.mean(component_counts) - 1
        sse = np.mean(np.abs(solid_counts - solid_counts_real) / solid_counts_real)

        print(f"\nWGAN-GP Evaluation:")
        print(f"  Volume Fraction MAE:     {vf_mae:.6f}")
        print(f"  Avg # Disconnected Fluid Segments: {avg_disconnected:.6f}")
        print(f"  MMD:                     {mmd:.6f}")
        print(f"  R-Div:                   {r_div:.6f}")
        print(f"  SSE:                     {sse:.6f}")

        metrics = {
            "volume_fraction_mae": vf_mae,
            "avg_disconnected_fluid_segments": avg_disconnected,
            "mmd": mmd,
            "r_div": r_div,
            "sse": sse
        }
        
        # Save all data matching transformer format (except indices)
        np.save(os.path.join(self.eval_dir, "test_indices.npy"), np.array(test_indices))
        np.save(os.path.join(self.eval_dir, "samples.npy"), all_samples)
        np.save(os.path.join(self.eval_dir, "metrics.npy"), metrics)


if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Initial Arguments")
    eval_wgan_gp = EvalWGAN_GP(args)