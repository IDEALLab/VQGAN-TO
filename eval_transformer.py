import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import label

from transformer import VQGANTransformer
from utils import get_data, set_precision, set_all_seeds, process_state_dict, MMD, rdiv, get_data_split_indices, npy_to_gamma, mirror
from args import get_args, load_args, print_args


"""
Comprehensive evaluation and metrics calculation + saving for Transformer models (Stage 2)
"""
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
        (dataloader, _, test_dataloader), _, _ = get_data(args, use_val_split=True)
        _, _, test_indices = get_data_split_indices(args, use_val_split=True)
        orig_indices = np.load("../data/new/nonv/index_5666.npy")
        L_size = args.decoder_start_resolution
        
        # Check if test_gammas directory exists and is nonempty
        test_gammas_dir = "../data/test_gammas/"
        if not os.path.exists(test_gammas_dir) or len(os.listdir(test_gammas_dir)) == 0:
            os.makedirs(test_gammas_dir, exist_ok=True)
            print("Creating test_gammas directory and converting test set...")
            self._create_test_gammas(test_dataloader, test_indices, orig_indices, test_gammas_dir, args)
            print("Completed creating test_gammas.")

        # Check if test_gammas_rounded directory exists and is nonempty
        test_gammas_rounded_dir = "../data/test_gammas_rounded/"
        if not os.path.exists(test_gammas_rounded_dir) or len(os.listdir(test_gammas_rounded_dir)) == 0:
            os.makedirs(test_gammas_rounded_dir, exist_ok=True)
            print("Creating test_gammas_rounded directory and converting thresholded test set...")
            self._create_test_gammas(test_dataloader, test_indices, orig_indices, test_gammas_rounded_dir, args, round_output=True)
            print("Completed creating test_gammas_rounded.")

        all_losses = []
        all_volume_mae = []
        all_gen_vfs = []
        all_real_vfs = []

        solid_counts = []
        fluid_counts = []
        solid_counts_real = []

        all_generated = []
        all_real_eval = []
        all_real_train = []

        print("Loading training data for evaluation...")
        with torch.no_grad():
            for i, (imgs, cond) in enumerate(dataloader):
                imgs = imgs.to(args.device, non_blocking=True)
                imgs = mirror(imgs, reshape=(400, 400)).clamp(0, 1).cpu().numpy()
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
                full_sample = mirror(logs["full_sample"], reshape=(400, 400)).clamp(0, 1).cpu().numpy()
                recon = mirror(logs["rec"], reshape=(400, 400)).clamp(0, 1).cpu().numpy()
                original = mirror(imgs, reshape=(400, 400)).clamp(0, 1).cpu().numpy()
                for j in range(full_sample.shape[0]):
                    test_index = i * args.batch_size + j
                    original_index = orig_indices[test_indices[test_index]]
                    gamma_tensor = full_sample[j, 0]
                    npy_to_gamma(gamma_tensor, path=self.results_dir, name=f"gamma_{original_index}")

                all_generated.append(full_sample)
                all_real_eval.append(original)

                # Compute VFs and MAE
                gen_vfs = full_sample.reshape(full_sample.shape[0], -1).mean(axis=1)
                ref_vfs = original.reshape(full_sample.shape[0], -1).mean(axis=1)

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

                indices = indices.cpu().numpy().reshape(-1,1,L_size,L_size)
                gen_indices = gen_indices.cpu().numpy().reshape(-1,1,L_size,L_size)

                # Append combined images and indices for later saving
                combined = np.stack([original, recon, full_sample], axis=1)
                idx_combined = np.stack([indices, gen_indices], axis=1)

                if i == 0:
                    all_samples = combined
                    all_indices = idx_combined
                else:
                    all_samples = np.concatenate([all_samples, combined], axis=0)
                    all_indices = np.concatenate([all_indices, idx_combined], axis=0)

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

        print("\nTransformer Evaluation:")
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
        
        np.save(os.path.join(self.eval_dir, "test_indices.npy"), np.array(test_indices))
        np.save(os.path.join(self.eval_dir, "samples.npy"), all_samples)
        np.save(os.path.join(self.eval_dir, "indices.npy"), all_indices)
        np.save(os.path.join(self.eval_dir, "metrics.npy"), metrics)

    def _create_test_gammas(self, test_dataloader, test_indices, orig_indices, test_gammas_dir, args, round_output=False):
        """Create gamma files for the test set"""
        with torch.no_grad():
            for i, (imgs, cond) in enumerate(tqdm(test_dataloader, desc="Converting test set to gammas")):
                imgs = imgs.to(args.device, non_blocking=True)
                original = mirror(imgs, reshape=(400, 400)).clamp(0, 1).cpu().numpy()
                if round_output:
                    original = (original > 0.5).astype(np.uint8)
                
                for j in range(original.shape[0]):
                    test_index = i * args.batch_size + j
                    original_index = orig_indices[test_indices[test_index]]
                    gamma_tensor = original[j, 0]
                    npy_to_gamma(gamma_tensor, path=test_gammas_dir, name=f"gamma_{original_index}")


if __name__ == '__main__':
    args = get_args()
    args.is_t = True
    print_args(args, title="Initial Arguments")
    eval_transformer = EvalTransformer(args)
