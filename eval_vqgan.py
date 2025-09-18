import os
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt

from lpips import LPIPS, GreyscaleLPIPS
from utils import get_data, plot_data, load_vqgan, set_precision, set_all_seeds
from args import get_args, load_args, print_args
from sklearn.metrics.pairwise import cosine_similarity


"""
Comprehensive evaluation and metrics calculation + saving for VQGAN models (Stage 1)
"""
class EvalVQGAN:
    def __init__(self, args):
        set_precision()
        set_all_seeds(args.seed)
    
        self.eval_dir = os.path.join(r"../evals", args.model_name)
        self.results_dir = os.path.join(self.eval_dir, "results")
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load saved training arguments and update current args
        args = load_args(args)
        
        # Now initialize VQGAN with potentially updated args
        self.vqgan = load_vqgan(args).eval()
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.grey_perceptual_loss = GreyscaleLPIPS().eval().to(device=args.device)
        
        # Perform evaluation
        self.evaluate(args)

    def evaluate(self, args):
        (_, _, test_dataloader), means, stds = get_data(args, use_val_split=True)
        
        # Metrics to track
        metrics = {
            'mse': [],
            'mae': [],
            'grey_lpips': [],
            # 'lpips': [],
            'codebook_usage': {}
        }
        
        # Sample images for visualization
        sample_images = []
        sample_reconstructions = []
        
        with torch.no_grad():
            for i, (imgs, c) in enumerate(tqdm(test_dataloader, desc="Evaluating")):
                imgs = imgs.to(device=args.device, non_blocking=True)
                
                # This is just to check the shape of the encoded images in the evaluation
                if i == 0:
                    encoded, _, _ = self.vqgan.encode(imgs)
                    print("Encoded shape:", encoded.shape)

                # The VQGAN forward method returns: decoded_images, codebook_indices, q_loss
                decoded_images, codebook_indices, _ = self.vqgan(imgs)
                decoded_images = decoded_images.clamp(0, 1)
                if i == 0:
                    print("Indices shape:", codebook_indices.shape)
                
                # Calculate metrics
                mse = torch.mean((imgs - decoded_images) ** 2).item()
                mae = torch.mean(torch.abs(imgs - decoded_images)).item()
                # lpips_value = self.perceptual_loss(imgs, decoded_images).mean().item()
                grey_lpips_value = self.grey_perceptual_loss(imgs, decoded_images).mean().item()
                
                metrics['mse'].append(mse)
                metrics['mae'].append(mae)
                # metrics['lpips'].append(lpips_value)
                metrics['grey_lpips'].append(grey_lpips_value)
                
                # Track codebook usage
                # codebook_indices is already the integer indices of used vectors
                indices = codebook_indices.cpu().numpy()
                unique_indices, counts = np.unique(indices, return_counts=True)
                
                for idx, count in zip(unique_indices, counts):
                    if idx in metrics['codebook_usage']:
                        metrics['codebook_usage'][idx] += count
                    else:
                        metrics['codebook_usage'][idx] = count
                
                # Save sample images (first batch only)
                if i == 0:
                    sample_images = imgs.cpu().detach().numpy()
                    sample_reconstructions = decoded_images.cpu().detach().numpy()
        
        # Calculate and print average metrics
        avg_mse = np.mean(metrics['mse'])
        avg_mae = np.mean(metrics['mae'])
        # avg_lpips = np.mean(metrics['lpips'])
        avg_grey_lpips = np.mean(metrics['grey_lpips'])
        
        print(f"Evaluation Results:")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average MAE: {avg_mae:.6f}")
        # print(f"Average LPIPS: {avg_lpips:.6f}")
        print(f"Average Greyscale LPIPS: {avg_grey_lpips:.6f}")
        
        # Calculate codebook usage statistics
        total_usage = sum(metrics['codebook_usage'].values())
        codebook_usage_pct = {k: (v / total_usage) * 100 for k, v in metrics['codebook_usage'].items()}
        active_codes = len(metrics['codebook_usage'])
        print(f"Codebook usage: {active_codes}/{args.num_codebook_vectors} vectors used ({active_codes/args.num_codebook_vectors*100:.2f}%)")

        # Compute normalized codebook entropy
        counts = np.array(list(metrics['codebook_usage'].values()))
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(args.num_codebook_vectors)
        normalized_entropy = entropy / max_entropy
        print(f"Normalized Codebook Entropy: {normalized_entropy:.4f}")

        # Cosine similarity among used codebook vectors
        used_indices = list(metrics['codebook_usage'].keys())

        # Pull weights once, on CPU, detached
        W = self.vqgan.codebook.embedding.weight.detach().cpu()  # (K, D)

        # Optional: restrict to used codes only
        if len(used_indices) >= 2:
            W = W[used_indices]

        E_np = W.numpy()
        if E_np.shape[0] >= 2:
            S = cosine_similarity(E_np)  # sklearn normalizes rows internally
            offdiag = S[~np.eye(S.shape[0], dtype=bool)]
            print(f"Mean cosine similarity (used codes): {offdiag.mean():.4f}")
            print(f"Max  cosine similarity (used codes): {offdiag.max():.4f}")

            plt.figure(figsize=(12, 6))
            plt.hist(
                offdiag,
                bins=100,
                range=(-1, 1),
                weights=np.ones_like(offdiag) * 100.0 / offdiag.size,
                alpha=0.5,
                color="blue",
            )
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Percentage (%)")
            plt.xlim(-1, 1)
            plt.ylim(0, 3)
            plt.savefig(
                os.path.join(self.results_dir, "cosine_similarity_used.png"),
                format="png",
                dpi=300,
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()
        else:
            print("Not enough codes to compute pairwise cosine similarity.")
        
        # Save metrics to file
        metrics_summary = {
            'avg_mse': avg_mse,
            'avg_mae': avg_mae,
            # 'avg_lpips': avg_lpips,
            'avg_grey_lpips': avg_grey_lpips,
            'active_codes': active_codes,
            'total_codes': args.num_codebook_vectors,
            'usage_percentage': active_codes/args.num_codebook_vectors*100,
            'normalized_entropy': normalized_entropy,
            'cosine_sims': offdiag,
            'codebook_usage': metrics['codebook_usage']
        }
        np.save(os.path.join(self.eval_dir, "metrics.npy"), metrics_summary)
        
        # Plot codebook usage distribution
        plt.figure(figsize=(12, 6))
        sorted_usage = sorted(codebook_usage_pct.items(), key=lambda x: x[1], reverse=True)
        indices_sorted = [i for i, _ in sorted_usage]
        values_sorted = [v for _, v in sorted_usage]

        plt.bar(range(len(values_sorted)), values_sorted)
        plt.xlabel('Sorted Codebook Vector Index')
        plt.ylabel('Usage Percentage (%)')
        plt.title('Codebook Utilization (Sorted)')
        plt.savefig(os.path.join(self.results_dir, "codebook_usage.png"), format="png", dpi=300, bbox_inches="tight", transparent=True)
        plt.close()
        
        # Visualize sample reconstructions
        num_samples = min(5, len(sample_images))
        for i in range(num_samples):
            combined = np.stack([
                sample_reconstructions[i], 
                sample_images[i]
            ])
            img_fname = os.path.join(self.results_dir, f"sample_{i}.png")
            
            plot_data(
                combined, 
                titles=['Reconstruction', 'Original'], 
                ranges=[[0, 1], [0, 1]], 
                fname=img_fname,
                cbar=False, 
                dpi=400, 
                mirror_image=True, 
                cmap=sns.color_palette("viridis", as_cmap=True), 
                fontsize=20
            )
        
        # If multiple samples, create a grid of all samples
        if num_samples > 1:
            fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
            for i in range(num_samples):
                axes[0, i].imshow(sample_images[i].transpose(1, 2, 0), cmap='viridis')
                axes[0, i].set_title('Original' if i == 0 else '')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(sample_reconstructions[i].transpose(1, 2, 0), cmap='viridis', vmin=0, vmax=1)
                axes[1, i].set_title('Reconstruction' if i == 0 else '')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "all_samples.png"), format="png", dpi=300, bbox_inches="tight", transparent=True)
            plt.close()

if __name__ == '__main__':
    args = get_args()
    print_args(args, title="Initial Arguments")
    eval_vqgan = EvalVQGAN(args)