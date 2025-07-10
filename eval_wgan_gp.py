import os
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
from scipy.ndimage import label

from wgan_gp import Generator, VQGANLatentWrapper
from utils import get_data, set_precision, set_all_seeds, plot_data, process_state_dict
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
        checkpoint = process_state_dict(torch.load(ckpt_path, map_location=args.device), weights_only=True)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

        self.evaluate()

    def evaluate(self):
        (_, _, test_dataloader), means, stds = get_data(self.args, use_val_split=True)
        all_generated = []
        all_gen_vfs = []
        all_real_vfs = []
        all_volume_mae = []
        component_counts = []

        vf_mean = means[2]
        vf_std = stds[2]

        with torch.no_grad():
            for i, (imgs, cond) in enumerate(tqdm(test_dataloader, desc="Evaluating WGAN-GP")):
                cond = cond.to(self.args.device, non_blocking=True)
                imgs = imgs.to(self.args.device, non_blocking=True)

                z = torch.randn(cond.size(0), self.args.latent_dim, device=self.args.device)
                fake_latents = self.model(z, cond).clamp(0, 1)
                fake_imgs = self.vq_wrapper.decode(fake_latents).clamp(0, 1)
                all_generated.append(fake_imgs.cpu().numpy())

                gen_vfs = fake_imgs.view(fake_imgs.size(0), -1).mean(dim=1).cpu().numpy()
                ref_vfs_normalized = cond[:, 2].cpu().numpy()
                ref_vfs = ref_vfs_normalized * vf_std + vf_mean

                mae = np.abs(gen_vfs - ref_vfs)
                all_volume_mae.extend(mae)
                all_gen_vfs.extend(gen_vfs)
                all_real_vfs.extend(ref_vfs)

                binary_samples = (fake_imgs.cpu().numpy() > 0.5).astype(np.uint8)
                for b in range(binary_samples.shape[0]):
                    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
                    labeled, num_components = label(binary_samples[b, 0], structure=structure)
                    component_counts.append(num_components)

                for j in range(len(fake_imgs)):
                    combined = np.stack([
                        imgs[j].cpu().numpy(),
                        fake_imgs[j].cpu().numpy()
                    ])
                    fname = os.path.join(self.results_dir, f"sample_{i*self.args.batch_size + j}.png")
                    plot_data(
                        combined,
                        titles=["Real", "Generated"],
                        ranges=[[0, 1], [0, 1]],
                        fname=fname,
                        cbar=False,
                        dpi=400,
                        mirror_image=True,
                        cmap=sns.color_palette("viridis", as_cmap=True),
                        fontsize=20
                    )

        all_generated = np.concatenate(all_generated, axis=0)
        np.save(os.path.join(self.eval_dir, "generated.npy"), all_generated)
        np.save(os.path.join(self.eval_dir, "vfs_gen"), np.array(all_gen_vfs))
        np.save(os.path.join(self.eval_dir, "vfs_real.npy"), np.array(all_real_vfs))
        np.save(os.path.join(self.eval_dir, "vfs_mae.npy"), np.array(all_volume_mae))

        vf_mae = np.mean(all_volume_mae)
        avg_components = np.mean(component_counts)

        print(f"\nWGAN-GP Evaluation:")
        print(f"  Volume Fraction MAE: {vf_mae:.6f}")
        print(f"  Avg # Disconnected Fluid Segments: {avg_components:.2f}")

        metrics = {
            "vf_mae": vf_mae,
            "avg_disconnected_fluid_segments": avg_components
        }
        np.save(os.path.join(self.eval_dir, "metrics.npy"), metrics)


if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Initial Arguments")
    eval_wgan_gp = EvalWGAN_GP(args)
