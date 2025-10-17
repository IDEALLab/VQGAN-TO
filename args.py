import os
import argparse
import json
from datetime import datetime
from copy import deepcopy


"""
Argument parsing and saving/loading functions for VQGAN, CVQGAN, Transformer, and WGAN-GP training and evaluation
"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")

    
def get_args():
    parser = argparse.ArgumentParser(description="VQGAN Training Args")

    # Metadata
    parser.add_argument('--load_from_hf', type=str2bool, default=False, help='Whether to load the data from Hugging Face. Use in conjunction with --dataset_path and --conditions_path.')
    parser.add_argument('--repo_id', type=str, default='IDEALLab/MTO-2D', help='Hugging Face repo ID to load the dataset and conditions from. Used when --load_from_hf is set to True.')
    parser.add_argument('--dataset_path', type=str, default='../data/new/nonv/gamma_5666_half.npy', help='Local path to the dataset file (numpy format).')
    parser.add_argument('--index_path', type=str, default='../data/new/nonv/index_5666.npy', help='Local path to the index file (numpy format).')
    parser.add_argument('--conditions_path', type=str, default='../data/new/nonv/inp_paras_5666.npy', help='Local path to the conditions file (numpy format).')
    parser.add_argument('--hf_dataset_path', type=str, default='gamma_5666_half.npy', help='Hugging Face path to the dataset file (numpy format). Used when --load_from_hf is set to True.')
    parser.add_argument('--hf_conditions_path', type=str, default='inp_paras_5666.npy', help='Hugging Face path to the conditions file (numpy format). Used when --load_from_hf is set to True.')
    parser.add_argument('--hf_index_path', type=str, default='index_5666.npy', help='Hugging Face path to the index file (numpy format). Used when --load_from_hf is set to True.')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='DEBUG: Fraction of the dataset to use for training')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training and evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--track', type=str2bool, default=True, help='Whether to track the training statistics')
    parser.add_argument('--save_model', type=str2bool, default=True, help='Whether to save the trained model')
    parser.add_argument('--sample_interval', type=int, default=1, help='Interval (in epochs) to save checkpoints during training')
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("Tr-%Y-%m-%d_%H-%M-%S"), help='Name of the training run for saving models and logs')
    parser.add_argument('--val_fraction', type=float, default=0.05, help='Fraction of the dataset to use for validation')


    # VQGAN Stage 1 (Autoencoder): Codebook & Training
    parser.add_argument('--latent_dim', type=int, default=256, help='Individual code dimension')
    parser.add_argument('--image_size', type=int, default=256, help='Input image size for the VQGAN')
    parser.add_argument('--num_codebook_vectors', type=int, default=1024, help='Number of codebook vectors')
    parser.add_argument('--beta', type=float, default=0.25, help='Beta hyperparameter for the codebook commitment loss')
    parser.add_argument('--image_channels', type=int, default=1, help='Number of image input channels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2.25e-05, help='Learning rate for the Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparameter for the Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='Beta2 hyperparameter for the Adam optimizer')
    parser.add_argument('--disc_start', type=int, default=0, help='Epoch to start discriminator training')
    parser.add_argument('--disc_factor', type=float, default=1.0, help='Weighting factor for the adversarial loss from the discriminator')
    parser.add_argument('--rec_loss_factor', type=float, default=1.0, help='Weighting factor for the reconstruction loss')
    parser.add_argument('--perceptual_loss_factor', type=float, default=1.0, help='Weighting factor for the perceptual loss')
    parser.add_argument('--use_greyscale_lpips', type=str2bool, default=True, help='Whether to use greyscale LPIPS loss (set to True for binary or greyscale image data)')
    parser.add_argument('--spectral_disc', type=str2bool, default=False, help='Whether to use spectral normalization in the discriminator')
    parser.add_argument('--vq_min_validation', type=str2bool, default=False, help='Whether to save the best model based on minimum validation loss')
    parser.add_argument('--vq_track_val_loss', type=str2bool, default=True, help='Whether to track validation loss during training')
    parser.add_argument('--use_Online', type=str2bool, default=False, help='Whether to use Online codebook')
    parser.add_argument('--use_DAE', type=str2bool, default=False, help='Whether to use Decoupled Autoencoder (DAE)')
    parser.add_argument('--DAE_dropout', type=float, default=0.0, help='Decoder dropout rate for DAE during part 1')
    parser.add_argument('--DAE_switch_epoch', type=int, default=50, help='Epoch to switch to part 2 of DAE training')


    # Continuous AE
    parser.add_argument('--no_vq', type=str2bool, default=False, help='Whether to use a continuous autoencoder (no VQ)')


    # VQGAN Stage 1 (Autoencoder): Encoder/Decoder
    parser.add_argument('--encoder_channels', type=int, nargs='+', default=[128, 128, 128, 256, 256, 512], help='List of channel sizes for each encoder layer')
    parser.add_argument('--encoder_attn_resolutions', type=int, nargs='+', default=[16], help='List of resolutions at which to apply attention in the encoder')
    parser.add_argument('--encoder_num_res_blocks', type=int, default=2, help='Number of residual blocks per encoder layer')
    parser.add_argument('--encoder_start_resolution', type=int, default=256, help='Starting resolution for the encoder')
    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[512, 256, 256, 128, 128], help='List of channel sizes for each decoder layer')
    parser.add_argument('--decoder_attn_resolutions', type=int, nargs='+', default=[16], help='List of resolutions at which to apply attention in the decoder')
    parser.add_argument('--decoder_num_res_blocks', type=int, default=3, help='Number of residual blocks per decoder layer')
    parser.add_argument('--decoder_start_resolution', type=int, default=16, help='Starting (latent) resolution for the decoder')


    # # VQGAN Stage 2 (Transformer)
    '''
    GPT2 CONFIGURATIONS: WE USE THE SMALLEST GPT2
    gpt2:         n_layer=12, n_head=12, n_embd=768  --> 124M params
    gpt2-medium:  n_layer=24, n_head=16, n_embd=1024 --> 350M params
    gpt2-large:   n_layer=36, n_head=20, n_embd=1280 --> 774M params
    gpt2-xl:      n_layer=48, n_head=25, n_embd=1600 --> 1558M params
    '''
    parser.add_argument('--is_t', type=str2bool, default=False, help='Specify if the model is a transformer')
    parser.add_argument('--early_stop', type=str2bool, default=True, help='Whether to use early stopping based on validation loss')
    parser.add_argument('--t_learning_rate', type=float, default=0.0006, help='Learning rate for the transformer Adam optimizer')
    parser.add_argument('--t_sample_interval', type=int, default=5, help='Interval (in epochs) to save checkpoints during Transformer training')
    parser.add_argument('--t_name', type=str, default="Tr_baseline", help='Name of the transformer model for saving and loading')
    parser.add_argument('--model_name', type=str, default="baseline", help='Name of the VQGAN model for saving and loading')
    parser.add_argument('--c_model_name', type=str, default="cvq", help='Name of the CVQGAN model for saving and loading')
    parser.add_argument('--pkeep', type=float, default=1.0, help='Probability of keeping an input token during training (for data corruption)')
    parser.add_argument('--pkeep_delay', type=int, default=10, help='Number of epochs to wait before applying pkeep corruption')
    parser.add_argument('--sos_token', type=int, default=0, help='Start-of-sequence token with no conditioning')
    parser.add_argument('--t_is_c', type=str2bool, default=True, help='Whether the transformer is conditional (uses CVQGAN codes as conditioning)')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of transformer layers')          
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')           
    parser.add_argument('--n_embd', type=int, default=768, help='Transformer embedding dimension')          
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in the transformer')       
    parser.add_argument('--bias', type=str2bool, default=True, help='Whether to use bias in the transformer layers')
    parser.add_argument('--T_min_validation', type=str2bool, default=True, help='Whether to save the best transformer model based on minimum validation loss')
    parser.add_argument('--train_samples', type=int, default=1e12, help='Number of training samples to use (for debugging -- set to a large number to use all samples)')


    # Conditional VQGAN (CVQGAN)
    parser.add_argument('--is_c', type=str2bool, default=False, help='Specify if the model is a conditional VQGAN (CVQGAN)')
    parser.add_argument('--c_input_dim', type=int, default=3, help='Number of input conditions (assuming each condition is a scalar)')
    parser.add_argument('--c_hidden_dim', type=int, default=256, help='Hidden dimension for the condition encoder MLP')
    parser.add_argument('--c_latent_dim', type=int, default=4, help='Individual code dimension for CVQGAN')
    parser.add_argument('--c_num_codebook_vectors', type=int, default=64, help='Number of codebook vectors for CVQGAN')
    parser.add_argument('--c_fmap_dim', type=int, default=4, help='Feature map dimension for CVQGAN output')


    # WGAN-GP
    parser.add_argument('--is_gan', type=str2bool, default=False, help='Specify if the model is a GAN (WGAN-GP)')
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of CPU threads to use during batch generation")
    parser.add_argument("--n_critic", type=int, default=5, help="Number of training steps for discriminator per iter")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient penalty lambda hyperparameter")
    parser.add_argument('--gan_min_validation', type=str2bool, default=False, help='Whether to save the best GAN model based on minimum validation loss')
    parser.add_argument("--gan_sample_interval", type=int, default=10, help="Interval between image samples")
    parser.add_argument('--gan_name', type=str, default="dcgan_baseline", help='Name of the GAN model for saving and loading')
    parser.add_argument('--gan_use_cvq', type=str2bool, default=True, help='Whether to use CVQGAN for GAN conditioning')
    parser.add_argument('--gan_g_learning_rate', type=float, default=0.0002, help='Learning rate for the GAN generator Adam optimizer')
    parser.add_argument('--gan_d_learning_rate', type=float, default=0.0002, help='Learning rate for the GAN discriminator Adam optimizer')
    parser.add_argument('--c_transform_dim', type=int, default=256, help='Dimension to transform condition encoding to before concatenation with input')
    parser.add_argument("--use_spectral", type=str2bool, default=False, help="Whether to use spectral normalization in the GAN conv layers")


    args, _ = parser.parse_known_args()

    # Adjust defaults if CVQGAN and relevant arguments not specified
    if args.is_c:
        if args.image_channels == parser.get_default("image_channels"):
            args.image_channels = 3 # CVQGAN assumes 3-channel conditions input
        if args.learning_rate == parser.get_default("learning_rate"):
            args.learning_rate = 0.0002 # CVQGAN typically uses a higher learning rate
        if args.disc_start == parser.get_default("disc_start"):
            args.disc_start = 1e12 # CVQGAN does not use a discriminator
        if args.epochs == parser.get_default("epochs"):
            args.epochs = 1000 # CVQGAN trains for more epochs since it is a very small model and we can afford this
        if args.sample_interval == parser.get_default("sample_interval"):
            args.sample_interval = 100 # CVQGAN does not need frequent sampling

    # Add derived paths for Stage 2 (Transformer)
    args.checkpoint_path = os.path.join("../saves", args.model_name, "checkpoints", "vqgan.pth")
    args.c_checkpoint_path = os.path.join("../saves", args.c_model_name, "checkpoints", "vqgan.pth")
    args.decoder_start_resolution = args.image_size // (2 ** (len(args.encoder_channels) - 2))


    return args


def save_args(args):
    args_dict = vars(args)
    save_dir = os.path.join("../saves", args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Convert objects to strings
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
        args.gan_name if args.is_gan else (args.t_name if args.is_t else (args.c_model_name if args.is_c else args.model_name)),
        "training_args.json"
    )

    if os.path.exists(training_args_path):
        print(f"Loading training arguments from {training_args_path}")

        try:
            with open(training_args_path, 'r') as f:
                training_args_dict = json.load(f)

            preserve_keys = ['device']
            if not args.is_t and not args.is_gan:
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
