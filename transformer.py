import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanogpt import GPT, GPTConfig
from vqgan import VQGAN
from copy import deepcopy
from utils import print_args


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token

        vq_args = self.load_training_args(args)
        self.vqgan = VQGAN(vq_args).to(device=vq_args.device)
        self.load_vqgan(vq_args)

        if args.t_is_c:
            # TODO: create a new copy of args with args.is_c = True to pass to self.cvqgan
            temp_cvq_args = deepcopy(args)
            temp_cvq_args.is_c = True
            cvq_args = self.load_training_args(temp_cvq_args)
            self.cvqgan = VQGAN(cvq_args).to(device=cvq_args.device)
            self.load_vqgan(cvq_args)

        # Create config object for NanoGPT
        transformer_config = GPTConfig(
            vocab_size=vq_args.num_codebook_vectors,
            block_size=1024,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.3,  # Add dropout parameter (default in nanoGPT)
            bias=True     # Add bias parameter (default in nanoGPT)
        )
        self.transformer = GPT(transformer_config)

        self.t_is_c = args.t_is_c
        self.pkeep = args.pkeep

    def load_training_args(self, args):
        """Load the training arguments and update the evaluation args, returning the updated args."""
        args = deepcopy(args)  # Prevent upstream mutation

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
                
                print_args(args, "Updated Evaluation Arguments")

            except Exception as e:
                print(f"Error loading training arguments: {e}")
                print("Using provided evaluation arguments instead.")
        else:
            print(f"Warning: Training arguments not found at {training_args_path}. Using provided evaluation arguments.")

        return args

    def load_vqgan(self, args):
        model_path = os.path.join(r"../saves", args.run_name, "checkpoints", "vqgan.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=args.device, weights_only=True)
        
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

        model = self.cvqgan if args.is_c else self.vqgan
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
            model = torch.compile(model)

        setattr(self, 'cvqgan' if args.is_c else 'vqgan', model)
    

    @torch.no_grad()
    def encode_to_z(self, x):
        # Conditional case
        if x.ndim == 2:
            quant_z, indices, _ = self.cvqgan.encode(x)
        else:
            quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, -1)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    def forward(self, x, c):
        _, indices = self.encode_to_z(x)

        if self.t_is_c:
            _, sos_tokens = self.encode_to_z(c)
        else:
            sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
            sos_tokens = sos_tokens.long().to("cuda")

        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        # NanoGPT forward doesn't use embeddings parameter, but takes targets
        # We're ignoring the loss returned by NanoGPT
        logits, _ = self.transformer(new_indices[:, :-1], None)
        logits = logits[:, sos_tokens.shape[1]-1:]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        
        # Option 1: Use NanoGPT's built-in generate method
        # return self.transformer.generate(x, steps, temperature, top_k)[:, c.shape[1]:]
        
        # Option 2: Keep the original sampling logic for compatibility
        for k in range(steps):
            logits, _ = self.transformer(x, None)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # Determine the actual vocabulary size for this batch
                # Count non-negative infinity values in the logits
                n_tokens = torch.sum(torch.isfinite(logits), dim=-1).min().item()
                
                # Use the minimum of top_k and the actual number of tokens
                effective_top_k = min(top_k, n_tokens)
                
                # Apply top_k with the effective value
                if effective_top_k > 0:  # Ensure we have at least one token to sample
                    logits = self.top_k_logits(logits, effective_top_k)
                else:
                    # Fallback if all logits are -inf (shouldn't happen, but just in case)
                    print("Warning: No finite logits found for sampling")
                    # Make all logits equal (uniform distribution)
                    logits = torch.zeros_like(logits)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x, c):
        log = dict()

        _, indices = self.encode_to_z(x)
        if self.t_is_c:
            _, sos_tokens = self.encode_to_z(c)
        else:
            sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
            sos_tokens = sos_tokens.long().to("cuda")

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec, half_sample, full_sample))