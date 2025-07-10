import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
from scipy.ndimage import label

from transformer import VQGANTransformer
from utils import get_data, set_precision, set_all_seeds, plot_data, process_state_dict, MMD, rdiv
from args import get_args, load_args, print_args


# TODO: MMD, R-Div, SSE
class EvalTransformer:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)

        self.eval_dir = os.path.join("../evals", args.t_name)
        self.results_dir = os.path.join(self.eval_dir, "results")
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        args = load_args(args)

        self.model = VQGANTransformer(args).to(device=args.device)
        ckpt_path = os.path.join("../saves", args.run_name, "checkpoints", "transformer.pt")
        checkpoint = process_state_dict(torch.load(ckpt_path, map_location=args.device, weights_only=True))
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

        self.evaluate(args)

    def evaluate(self, args):
        (dataloader, _, test_dataloader), means, stds = get_data(args, use_val_split=True)
        
        # from torch.utils.data import Subset, DataLoader
        # dataloader = DataLoader(
        #     Subset(dataloader.dataset, range(16)),
        #     batch_size=dataloader.batch_size,
        #     shuffle=False,
        #     num_workers=dataloader.num_workers,
        #     pin_memory=getattr(dataloader, "pin_memory", False)
        # )

        # test_dataloader = DataLoader(
        #     Subset(test_dataloader.dataset, range(16)),
        #     batch_size=test_dataloader.batch_size,
        #     shuffle=False,
        #     num_workers=test_dataloader.num_workers,
        #     pin_memory=getattr(test_dataloader, "pin_memory", False)
        # )

        all_losses = []
        all_volume_mae = []
        all_gen_vfs = []
        all_real_vfs = []

        solid_counts = []
        fluid_counts = []
        solid_counts_real = []
        
        vf_mean = means[2]
        vf_std = stds[2]

        all_generated = []
        all_real_eval = []
        all_real_train = []

        print("Loading training data for evaluation...")
        with torch.no_grad():
            for i, (imgs, cond) in enumerate(dataloader):
                imgs = imgs.to(args.device, non_blocking=True).cpu().numpy()
                all_real_train.append(imgs)
        print("Completed loading training data for evaluation.")

        with torch.no_grad():
            for i, (imgs, cond) in enumerate(tqdm(test_dataloader, desc="Evaluating Transformer")):
                imgs = imgs.to(args.device, non_blocking=True)
                cond = cond.to(args.device, non_blocking=True)

                # Cross-entropy loss
                logits, targets = self.model(imgs, cond)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                all_losses.append(loss.item())

                # Generate full samples
                logs, _ = self.model.log_images(imgs, cond, top_k=None, greedy=False)
                full_sample = logs["full_sample"].clamp(0, 1).cpu().numpy()
                recon = logs["rec"].clamp(0, 1).cpu().numpy()
                original = imgs.cpu().numpy()

                all_generated.append(full_sample)
                all_real_eval.append(original)

                # Compute VFs and MAE
                gen_vfs = full_sample.reshape(full_sample.shape[0], -1).mean(axis=1)
                ref_vfs_normalized = cond[:, 2].cpu().numpy()
                ref_vfs = ref_vfs_normalized * vf_std + vf_mean

                mae = np.abs(gen_vfs - ref_vfs)
                all_volume_mae.extend(mae)
                all_gen_vfs.extend(gen_vfs)
                all_real_vfs.extend(ref_vfs)

                # Count disconnected fluid segments
                binary_samples = (full_sample > 0.5).astype(np.uint8)
                binary_samples_real = (original > 0.5).astype(np.uint8)
                for b in range(binary_samples.shape[0]):
                    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
                    _, num_fluid = label(binary_samples[b, 0], structure=structure)
                    _, num_solid = label(1 - binary_samples[b, 0], structure=structure)
                    fluid_counts.append(num_fluid)
                    solid_counts.append(num_solid)
                for b in range(binary_samples_real.shape[0]):
                    _, num_solid_real = label(1 - binary_samples_real[b, 0], structure=structure)
                    solid_counts_real.append(num_solid_real)

                # Get quantized and predicted indices
                _, indices = self.model.encode_to_z(imgs)
                if self.model.t_is_c:
                    _, sos_tokens = self.model.encode_to_z(cond, is_c=True)
                else:
                    sos_tokens = torch.ones(imgs.shape[0], 1) * self.model.sos_token
                    sos_tokens = sos_tokens.long().to(imgs.device)

                start = indices[:, :0]
                gen_indices = self.model.sample(start, sos_tokens, steps=indices.shape[1], top_k=None, greedy=False)

                indices = indices.cpu().numpy().reshape(-1,1,16,16)
                gen_indices = gen_indices.cpu().numpy().reshape(-1,1,16,16)

                for j in range(len(imgs)):
                    combined = np.stack([original[j], recon[j], full_sample[j]])
                    fname = os.path.join(self.results_dir, f"sample_{i*args.batch_size + j}.png")
                    plot_data(
                        combined,
                        titles=["Real", "Reconstruction", "Generated"],
                        ranges=[[0, 1]] * 3,
                        fname=fname,
                        cbar=False,
                        dpi=400,
                        mirror_image=True,
                        cmap=sns.color_palette("viridis", as_cmap=True), 
                        fontsize=20
                    )

                    idx_combined = np.stack([indices[j].astype(np.float32), gen_indices[j].astype(np.float32)])
                    idx_fname = os.path.join(self.results_dir, f"indices_{i*args.batch_size + j}.png")
                    plot_data(
                        idx_combined,
                        titles=["VQGAN Indices", "Predicted Indices"],
                        ranges=[[np.min(indices), np.max(indices)]] * 2,
                        fname=idx_fname,
                        cbar=False,
                        dpi=400,
                        mirror_image=True,
                        cmap=sns.color_palette("viridis", as_cmap=True),
                        fontsize=20,
                        reshape_size=(64, 32),
                        mode="nearest"
                    )

        # Save full generated samples
        all_generated = np.concatenate(all_generated, axis=0)
        all_real_eval = np.concatenate(all_real_eval, axis=0)
        all_real_train = np.concatenate(all_real_train, axis=0)
        solid_counts = np.array(solid_counts)
        solid_counts_real = np.array(solid_counts_real)

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
        log_avg_loss = np.log(np.mean(all_losses) + 1e-8)
        vf_mae = np.mean(all_volume_mae)
        avg_disconnected = np.mean(fluid_counts) - 1
        sse = np.mean(np.abs(solid_counts - solid_counts_real) / solid_counts_real)

        print(f"\nTransformer Evaluation:")
        print(f"  Log of Average CE Loss: {log_avg_loss:.6f}")
        print(f"  Volume Fraction MAE:     {vf_mae:.6f}")
        print(f"  Avg # Disconnected Fluid Segments: {avg_disconnected:.6f}")
        print(f"  MMD:                     {mmd:.6f}")
        print(f"  R-Div:                   {r_div:.6f}")
        print(f"  SSE:                     {sse:.6f}")

        metrics = {
            "log_avg_loss": log_avg_loss,
            "volume_fraction_mae": vf_mae,
            "avg_disconnected_fluid_segments": avg_disconnected,
            "mmd": mmd,
            "r_div": r_div,
            "sse": sse
        }
        np.save(os.path.join(self.eval_dir, "metrics.npy"), metrics)


if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Initial Arguments")
    eval_transformer = EvalTransformer(args)