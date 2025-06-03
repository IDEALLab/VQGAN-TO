import os
import argparse
import json
from datetime import datetime
from copy import deepcopy
from utils import str2bool


def get_args():
    parser = argparse.ArgumentParser(description="VQGAN Training Args")

    # Metadata
    parser.add_argument('--dataset_path', type=str, default='../data/gamma_4579_half.npy')
    parser.add_argument('--conditions_path', type=str, default='../data/inp_paras_4579.npy')
    parser.add_argument('--dropP_path', type=str, default='../data/dropP_4579.npy')
    parser.add_argument('--meanT_path', type=str, default='../data/meanT_4579.npy')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--algo', type=str, default='vqgan')
    parser.add_argument('--problem_id', type=str, default='mto')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--track', type=str2bool, default=True)
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("Tr-%Y-%m-%d_%H-%M-%S"))

    # VQGAN Stage 1 (Autoencoder): Codebook & Training
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_codebook_vectors', type=int, default=1024)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--image_channels', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=2.25e-05)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--disc_start', type=int, default=0)
    parser.add_argument('--disc_factor', type=float, default=1.0)
    parser.add_argument('--rec_loss_factor', type=float, default=1.0)
    parser.add_argument('--perceptual_loss_factor', type=float, default=1.0)

    parser.add_argument('--use_greyscale_lpips', type=str2bool, default=True)
    parser.add_argument('--spectral_disc', type=str2bool, default=False)
    parser.add_argument('--use_DAE', type=str2bool, default=False)
    parser.add_argument('--use_Online', type=str2bool, default=False)

    # VQGAN Stage 1 (Autoencoder): Encoder/Decoder
    parser.add_argument('--encoder_channels', type=int, nargs='+', default=[128, 128, 128, 256, 256, 512])
    parser.add_argument('--encoder_attn_resolutions', type=int, nargs='+', default=[16])
    parser.add_argument('--encoder_num_res_blocks', type=int, default=2)
    parser.add_argument('--encoder_start_resolution', type=int, default=256)

    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[512, 256, 256, 128, 128])
    parser.add_argument('--decoder_attn_resolutions', type=int, nargs='+', default=[16])
    parser.add_argument('--decoder_num_res_blocks', type=int, default=3)
    parser.add_argument('--decoder_start_resolution', type=int, default=16)

    # # VQGAN Stage 2 (Transformer)
    parser.add_argument('--is_t', type=str2bool, default=False)
    parser.add_argument('--t_learning_rate', type=float, default=4.5e-06)
    parser.add_argument('--t_name', type=str, default="Tr_baseline")
    parser.add_argument('--model_name', type=str, default="baseline")
    parser.add_argument('--c_model_name', type=str, default="cvq")
    parser.add_argument('--pkeep', type=float, default=1.0)
    parser.add_argument('--sos_token', type=int, default=0)
    parser.add_argument('--t_is_c', type=str2bool, default=True)
    parser.add_argument('--n_layer', type=int, default=12)          # 12
    parser.add_argument('--n_head', type=int, default=12)           # 12
    parser.add_argument('--n_embd', type=int, default=768)          # 768
    parser.add_argument('--dropout', type=float, default=0.0)       # 0.3
    parser.add_argument('--bias', type=str2bool, default=True)      # True

    # 'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    # 'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    # 'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    # 'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params

    # Conditional VQGAN (CVQGAN)
    parser.add_argument('--is_c', type=str2bool, default=False)
    parser.add_argument('--c_input_dim', type=int, default=3)
    parser.add_argument('--c_hidden_dim', type=int, default=256)
    parser.add_argument('--c_latent_dim', type=int, default=4)
    parser.add_argument('--c_num_codebook_vectors', type=int, default=64)
    parser.add_argument('--c_fmap_dim', type=int, default=4)

    args = parser.parse_args()

    # Add derived paths for Stage 2 (Transformer)
    args.checkpoint_path = os.path.join("../saves", args.model_name, "checkpoints", "vqgan.pth")
    args.c_checkpoint_path = os.path.join("../saves", args.c_model_name, "checkpoints", "vqgan.pth")

    return args


def save_args(args):
    args_dict = vars(args)
    save_dir = os.path.join("../saves", args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Convert non-serializables
    for k, v in args_dict.items():
        if not isinstance(v, (str, int, float, bool, list, dict, tuple, type(None))):
            args_dict[k] = str(v)

    with open(os.path.join(save_dir, "training_args.json"), 'w') as f:
        json.dump(args_dict, f, indent=4)

    print(f"Saved training args to {os.path.join(save_dir, 'training_args.json')}")


def load_args(args):
    """Load the training arguments and update the evaluation args, returning the updated args."""
    args = deepcopy(args)

    training_args_path = os.path.join(
        "../saves",
        args.t_name if args.is_t else (args.c_model_name if args.is_c else args.model_name),
        "training_args.json"
    )

    if os.path.exists(training_args_path):
        print(f"Loading training arguments from {training_args_path}")

        try:
            with open(training_args_path, 'r') as f:
                training_args_dict = json.load(f)

            preserve_keys = ['device', 'batch_size']
            if not args.is_t:
                preserve_keys.append('model_name')
            current_args_dict = vars(args)
            preserved_values = {k: current_args_dict[k] for k in preserve_keys if k in current_args_dict}

            for k, v in training_args_dict.items():
                if k not in preserve_keys:
                    try:
                        setattr(args, k, v)
                    except Exception:
                        pass

            for k, v in preserved_values.items():
                setattr(args, k, v)

            print("Evaluation arguments updated from saved training configuration.")
            print_args(args, "Updated Evaluation Arguments")

        except Exception as e:
            print(f"Error loading training arguments: {e}")
            print("Using provided evaluation arguments instead.")
    else:
        print(f"Warning: Training arguments not found at {training_args_path}. Using provided evaluation arguments.")

    return args


def print_args(args, title="Current Arguments"):
    print(f"\n{'-'*20} {title} {'-'*20}")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print(f"{'-'*50}\n")


# saves/2025-04-22_13-32-04: Complete baseline. Note: using batch_size 16.
# saves/2025-04-23_12-10-46: Baseline with Greyscale LPIPS
# saves/2025-04-24_06-36-52: Baseline with Greyscale LPIPS, codebook vectors reduced from 1024 to 32 ONLY
    # Satisfactory
# saves/2025-04-23_13-04-49: Baseline with Greyscale LPIPS, codebook vectors reduced from 1024 to 32, spectral norm enabled for decoder, disc start at 3*215 [NOW OBSOLETE]
    # Leads to some generator degradation
# saves/2025-04-23_20-27-33: Baseline with Greyscale LPIPS, codebook vectors reduced from 1024 to 32, spectral norm enabled for decoder, NO discriminator [NOW OBSOLETE]
    # Among the best so far but deviates from "GAN" in VQGAN
# saves/2025-04-24_14-49-04: Baseline with Greyscale LPIPS, latent dim reduced from 256 to 16, 512-width layers reduced to 256, attn dims changed from [16] to []
# saves/2025-04-24_17-46-54: Baseline with Greyscale LPIPS, codebook vectors reduced from 1024 to 64, latent dim reduced from 256 to 16, 512-width layers reduced to 256, spectral norm enabled, NO discriminator, learning rate increased from 2.25e-5 to 2e-4 [NOW OBSOLETE]
# saves/2025-04-24_19-28-29: Same as above but with 32 codebook vectors [NOW OBSOLETE]

# saves/2025-04-28_11-30-31: Same as above but with 16 codebook vectors [NOW OBSOLETE]
# saves/2025-04-28_11-31-01: Same as above but with batch size doubled to 32, sample interval 215 --> 108 [NOW OBSOLETE]

# saves/2025-04-29_18-42-49: Same as saves/2025-04-24_19-28-29 but with attn resolutions [16] --> [16, 32, 64] [NOW OBSOLETE]
    # Little to no improvement at a significant training time cost

# saves/2025-04-30_10-54-36: Same as saves/2025-04-24_19-28-29 but with experimental least volume loss (factor 1e-1) and codebook vectors raised back to 1024 [NOW OBSOLETE]
    # Requires too much effort to balance loss with other losses and spectral norm, abandoned

# Next idea: No discriminator, specnorm for decoder NO (implementation too difficult and non-standard), hybrid size hidden layers, 8 latent dim, 64/32 codebook vectors, learning rate TBD
    # Thus try 8 combinations with learning rate 2e-4/2.25e-5, latent dim 8/4, codebook vectors 64/32