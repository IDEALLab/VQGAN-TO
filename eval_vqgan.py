import os
import sys
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from lpips import LPIPS
from vqgan import VQGAN
from utils import get_data, plot_data, print_args, FocalWithLogitsLoss

class EvalVQGAN:
    def __init__(self, args):
        # Create evaluation directories first, before loading model
        self.eval_dir = os.path.join(r"../evals", args.model_name)
        self.results_dir = os.path.join(self.eval_dir, "results")
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load saved training arguments and update current args
        self.load_training_args(args)
        
        # Now initialize VQGAN with potentially updated args
        self.vqgan = VQGAN(args).to(device=args.device)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        
        # Load the trained model
        self.load_model(args)
        
        # Perform evaluation
        self.evaluate(args)

    def load_training_args(self, args):
        """Load the training arguments and update the evaluation args"""
        import json
        
        training_args_path = os.path.join(r"../saves", args.model_name, "training_args.json")
        if os.path.exists(training_args_path):
            print(f"Loading training arguments from {training_args_path}")
            
            try:
                with open(training_args_path, 'r') as f:
                    training_args_dict = json.load(f)
                
                # Update only model architecture related arguments, keep evaluation specific args
                preserve_keys = ['device', 'batch_size', 'model_name', 'test_split']
                current_args_dict = vars(args)
                preserved_values = {k: current_args_dict[k] for k in preserve_keys if k in current_args_dict}
                
                # Update args with training values
                for k, v in training_args_dict.items():
                    if k not in preserve_keys and hasattr(args, k):
                        # Convert to the right type based on current args
                        arg_type = type(getattr(args, k)) if hasattr(args, k) else type(v)
                        try:
                            if arg_type == bool and isinstance(v, str):
                                # Handle boolean values that were stored as strings
                                setattr(args, k, v.lower() == 'true')
                            else:
                                setattr(args, k, arg_type(v))
                        except (ValueError, TypeError):
                            # If conversion fails, use the value as is
                            setattr(args, k, v)
                
                # Restore preserved values
                for k, v in preserved_values.items():
                    setattr(args, k, v)
                    
                print_args(args, "Updated Evaluation Arguments")
            except Exception as e:
                print(f"Error loading training arguments: {e}")
                print("Using provided evaluation arguments instead.")
        else:
            print(f"Warning: Training arguments not found at {training_args_path}. Using provided evaluation arguments.")
    
    def load_model(self, args):
        model_path = os.path.join(r"../saves", args.model_name, "checkpoints", "vqgan.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=args.device)
        
        # Debug: Print checkpoint keys and model state_dict keys
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        state_dict = checkpoint["generator"]
        if all(k.startswith("_orig_mod.") for k in list(state_dict.keys())[:5]):
            print("Detected _orig_mod. prefix, removing it from keys...")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace("_orig_mod.", "")
                new_state_dict[new_k] = v
            state_dict = new_state_dict
        
        self.vqgan.load_state_dict(state_dict, strict=False)
        self.vqgan.eval()  # Set model to evaluation mode
        
        if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
            self.vqgan = torch.compile(self.vqgan)
    

    def evaluate(self, args):
        _, test_dataloader, means, stds = get_data(args)
        
        # Metrics to track
        metrics = {
            'mse': [],
            'mae': [],
            'focal_loss': [],
            'lpips': [],
            'codebook_usage': {}
        }
        
        # Sample images for visualization
        sample_images = []
        sample_reconstructions = []
        
        with torch.no_grad():
            for i, (imgs, c) in enumerate(tqdm(test_dataloader, desc="Evaluating")):
                imgs = imgs.to(device=args.device, non_blocking=True)
                
                with autocast(device_type=args.device, dtype=torch.float16):
                    # The VQGAN forward method returns: decoded_images, codebook_indices, q_loss
                    decoded_images, codebook_indices, _ = self.vqgan(imgs)
                    
                    # Calculate metrics
                    mse = torch.mean((imgs - decoded_images) ** 2).item()
                    mae = torch.mean(torch.abs(imgs - decoded_images)).item()
                    focal_loss = FocalWithLogitsLoss()(decoded_images, imgs).item()
                    lpips_value = self.perceptual_loss(imgs, decoded_images).mean().item()
                    
                    metrics['mse'].append(mse)
                    metrics['mae'].append(mae)
                    metrics['focal_loss'].append(focal_loss)
                    metrics['lpips'].append(lpips_value)
                    
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
        avg_focal_loss = np.mean(metrics['focal_loss'])
        avg_lpips = np.mean(metrics['lpips'])
        
        print(f"Evaluation Results:")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average MAE: {avg_mae:.6f}")
        print(f"Average Focal Loss: {avg_focal_loss:.6f}")
        print(f"Average LPIPS: {avg_lpips:.6f}")
        
        # Calculate codebook usage statistics
        total_usage = sum(metrics['codebook_usage'].values())
        codebook_usage_pct = {k: (v / total_usage) * 100 for k, v in metrics['codebook_usage'].items()}
        active_codes = len(metrics['codebook_usage'])
        print(f"Codebook usage: {active_codes}/{args.num_codebook_vectors} vectors used ({active_codes/args.num_codebook_vectors*100:.2f}%)")
        
        # Save metrics to file
        metrics_summary = {
            'avg_mse': avg_mse,
            'avg_mae': avg_mae,
            'avg_focal_loss': avg_focal_loss,
            'avg_lpips': avg_lpips,
            'active_codes': active_codes,
            'total_codes': args.num_codebook_vectors,
            'usage_percentage': active_codes/args.num_codebook_vectors*100
        }
        np.save(os.path.join(self.eval_dir, "metrics.npy"), metrics_summary)
        
        # Plot codebook usage distribution
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(codebook_usage_pct)), list(codebook_usage_pct.values()))
        plt.xlabel('Codebook Vector Index')
        plt.ylabel('Usage Percentage (%)')
        plt.title('Codebook Utilization')
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
                
                axes[1, i].imshow(sample_reconstructions[i].transpose(1, 2, 0), cmap='viridis')
                axes[1, i].set_title('Reconstruction' if i == 0 else '')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "all_samples.png"), format="png", dpi=300, bbox_inches="tight", transparent=True)
            plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN Evaluation")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 1024)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 1)')
    parser.add_argument('--dataset-path', type=str, default='../data/gamma_4579_half.npy', help='Path to data')
    parser.add_argument('--conditions-path', type=str, default='../data/inp_paras_4579.npy', help='Path to conditions')
    parser.add_argument('--device', type=str, default="cuda", help='Which device to use for evaluation')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for evaluation (default: 32)')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the trained model directory')
    parser.add_argument('--problem-id', type=str, default='mto', help='Problem ID (default: mto)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--test-split', type=float, default=0.1, help='Fraction of data to use for testing (default: 0.1)')

    # Decoder-specific args
    parser.add_argument('--spectral_norm', type=bool, default=False, help='Apply spectral normalization to Conv layers (default: False)')
    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[512, 256, 256, 128, 128], help='List of channel sizes for Decoder (default: [512, 256, 256, 128, 128])')
    parser.add_argument('--decoder_attn_resolutions', type=int, nargs='+', default=[16], help='Resolutions for attention in Decoder (default: [16])')
    parser.add_argument('--decoder_num_res_blocks', type=int, default=3, help='Number of residual blocks per stage in Decoder (default: 3)')
    parser.add_argument('--decoder_start_resolution', type=int, default=16, help='Starting resolution in Decoder (default: 16)')

    # Encoder-specific args
    parser.add_argument('--encoder_channels', type=int, nargs='+', default=[128, 128, 128, 256, 256, 512], help='List of channel sizes for Encoder (default: [128, 128, 128, 256, 256, 512])')
    parser.add_argument('--encoder_attn_resolutions', type=int, nargs='+', default=[16], help='Resolutions for attention in Encoder (default: [16])')
    parser.add_argument('--encoder_num_res_blocks', type=int, default=2, help='Number of residual blocks per stage in Encoder (default: 2)')
    parser.add_argument('--encoder_start_resolution', type=int, default=256, help='Starting resolution in Encoder (default: 256)')

    # Evaluation-specific args
    parser.add_argument('--use_focal_loss', type=bool, default=False, help='Use Focal Loss for training (default: False)')
    parser.add_argument('--use_DAE', type=bool, default=False, help='Use Decoupled Autoencoder for training (default: False)') # Not implemented
    parser.add_argument('--use_Online', type=bool, default=False, help='Use Online Clustered Codebook (default: False)') # Not implemented
    
    args = parser.parse_args()
    print_args(args, "Initial Evaluation Arguments")
    eval_vqgan = EvalVQGAN(args)