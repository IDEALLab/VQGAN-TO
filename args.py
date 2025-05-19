import os
import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from utils import str2bool
from copy import deepcopy


@dataclass
class VQGANArguments:
    # General
    latent_dim: int = 256
    image_size: int = 256
    num_codebook_vectors: int = 1024
    beta: float = 0.25
    image_channels: int = 1
    dataset_path: str = '../data/gamma_4579_half.npy'
    device: str = 'cuda'
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.9
    disc_start: int = 0
    disc_factor: float = 1.0
    rec_loss_factor: float = 1.0
    perceptual_loss_factor: float = 1.0

    # Extra metadata
    conditions_path: str = '../data/inp_paras_4579.npy'
    problem_id: str = 'mto'
    algo: str = 'vqgan'
    seed: int = 1
    track: bool = True
    save_model: bool = True
    sample_interval: int = 215
    run_name: str = field(default_factory=lambda: datetime.now().strftime("Tr-%Y-%m-%d_%H-%M-%S"))

    # Decoder-specific
    decoder_channels: list = field(default_factory=lambda: [512, 256, 256, 128, 128])
    decoder_attn_resolutions: list = field(default_factory=lambda: [16])
    decoder_num_res_blocks: int = 3
    decoder_start_resolution: int = 16

    # Encoder-specific
    encoder_channels: list = field(default_factory=lambda: [128, 128, 128, 256, 256, 512])
    encoder_attn_resolutions: list = field(default_factory=lambda: [16])
    encoder_num_res_blocks: int = 2
    encoder_start_resolution: int = 256

    # Training options
    use_greyscale_lpips: bool = True
    spectral_disc: bool = False
    use_DAE: bool = False
    use_Online: bool = False

    # Transformer
    model_name: str = "baseline"
    c_model_name: str = "cvq"
    pkeep: float = 1.0
    sos_token: int = 0
    t_is_c: bool = True
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.3  # Dropout parameter (default in nanoGPT)
    bias: bool = True     # Bias parameter (default in nanoGPT)
    # POTENTIALLY ADD IF NEEDED LATER --> is_t: bool = False

    # CVQGAN
    is_c: bool = False
    c_input_dim: int = 3
    c_hidden_dim: int = 256
    c_latent_dim: int = 4
    c_num_codebook_vectors: int = 64
    c_fmap_dim: int = 4

    # Automatically set after parsing
    checkpoint_path: str = field(init=False)
    c_checkpoint_path: str = field(init=False)

    def __post_init__(self):
        self.checkpoint_path = os.path.join("../saves", self.model_name, "checkpoints", "vqgan.pth")
        self.c_checkpoint_path = os.path.join("../saves", self.c_model_name, "checkpoints", "vqgan.pth")


def get_args():
    parser = argparse.ArgumentParser(description="VQGAN Training Args")

    for field_name, field_def in VQGANArguments.__dataclass_fields__.items():
        if field_def.init:
            arg_type = field_def.type
            default = field_def.default if field_def.default != field(default_factory=lambda: None) else None
            kwargs = {'type': arg_type, 'default': default, 'help': f'{field_name} (default: {default})'}

            if arg_type == bool:
                kwargs['type'] = str2bool
            elif arg_type == list:
                kwargs['type'] = int
                kwargs['nargs'] = '+'
            elif isinstance(default, list):
                kwargs['type'] = type(default[0]) if default else str
                kwargs['nargs'] = '+'

            parser.add_argument(f'--{field_name}', **{k: v for k, v in kwargs.items() if v is not None})

    args = parser.parse_args()
    return VQGANArguments(**vars(args))


def save_args(args):
    """Save the training arguments for later use in evaluation"""
    saves_dir = os.path.join("../saves", args.c_model_name if args.is_c else args.model_name)
    os.makedirs(saves_dir, exist_ok=True)

    args_dict = vars(args)

    # Convert any non-serializable objects to strings
    for key, value in args_dict.items():
        if not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
            args_dict[key] = str(value)

    path = os.path.join(saves_dir, "training_args.json")
    with open(path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print(f"Training arguments saved to {path}")


def load_args(args):
    """Load the training arguments and update the evaluation args, returning the updated args."""
    args = deepcopy(args)

    training_args_path = os.path.join(
        "../saves",
        args.c_model_name if args.is_c else args.model_name,
        "training_args.json"
    )

    if os.path.exists(training_args_path):
        print(f"Loading training arguments from {training_args_path}")

        try:
            with open(training_args_path, 'r') as f:
                training_args_dict = json.load(f)

            preserve_keys = ['device', 'batch_size', 'model_name', 'test_split']
            current_args_dict = vars(args)
            preserved_values = {k: current_args_dict[k] for k in preserve_keys if k in current_args_dict}

            for k, v in training_args_dict.items():
                if k not in preserve_keys and hasattr(args, k):
                    arg_type = type(getattr(args, k)) if hasattr(args, k) else type(v)
                    try:
                        if arg_type == bool and isinstance(v, str):
                            setattr(args, k, v.lower() == 'true')
                        else:
                            setattr(args, k, arg_type(v))
                    except (ValueError, TypeError):
                        setattr(args, k, v)

            for k, v in preserved_values.items():
                setattr(args, k, v)

            print("Evaluation arguments updated from saved training configuration.")

        except Exception as e:
            print(f"Error loading training arguments: {e}")
            print("Using provided evaluation arguments instead.")
    else:
        print(f"Warning: Training arguments not found at {training_args_path}. Using provided evaluation arguments.")

    return args

def print_args(args, title="Current Arguments"):
    """Print all arguments in a formatted way"""
    print(f"\n{'-'*20} {title} {'-'*20}")
    args_dict = vars(args)
    for k, v in sorted(args_dict.items()):
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