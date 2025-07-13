import torch
import torch.nn as nn
import torch.nn.functional as F
from nanogpt import GPT, GPTConfig
from copy import deepcopy
from utils import load_vqgan
from args import load_args, print_args


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token

        temp_vq_args = deepcopy(args)
        temp_vq_args.is_t = False
        vq_args = load_args(temp_vq_args)
        self.vqgan = load_vqgan(vq_args).eval()
        for param in self.vqgan.parameters():
            param.requires_grad = False

        if args.t_is_c:
            temp_cvq_args = deepcopy(args)
            temp_cvq_args.is_c = True
            temp_cvq_args.is_t = False
            cvq_args = load_args(temp_cvq_args)
            self.cvqgan = load_vqgan(cvq_args).eval()
            for param in self.cvqgan.parameters():
                param.requires_grad = False

        # block_size is automatically set to the combined sequence length of the VQGAN and CVQGAN
        block_size = (vq_args.image_size // (2 ** (len(vq_args.decoder_channels) - 1))) ** 2
        if args.t_is_c:
            block_size += cvq_args.c_fmap_dim ** 2

        # Create config object for NanoGPT
        transformer_config = GPTConfig(
            vocab_size=vq_args.num_codebook_vectors,
            block_size=block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,    # Add dropout parameter (default in nanoGPT)
            bias=args.bias           # Add bias parameter (default in nanoGPT)
        )
        self.transformer = GPT(transformer_config)

        self.t_is_c = args.t_is_c

    @torch.no_grad()
    def encode_to_z(self, x, is_c=False):
        # Conditional case
        if is_c:
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

    def forward(self, x, c, pkeep=1.0):
        _, indices = self.encode_to_z(x)

        if self.t_is_c:
            _, sos_tokens = self.encode_to_z(c, is_c=True)
        else:
            sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
            sos_tokens = sos_tokens.long().to(x.device)

        if pkeep < 1.0:
            mask = torch.bernoulli(pkeep * torch.ones(indices.shape, device=indices.device))
            mask = mask.round().to(dtype=torch.int64)
            random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
            new_indices = mask * indices + (1 - mask) * random_indices
        else:
            new_indices = indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        # NanoGPT forward doesn't use embeddings parameter, but takes targets
        # We're ignoring the loss returned by NanoGPT
        logits, _ = self.transformer(new_indices[:, :-1], None)
        logits = logits[:, -indices.shape[1]:]  # Always predict the last 256 tokens

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100, greedy=False):
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

            if greedy:
                ix = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def log_images(self, x, c, top_k=100, greedy=False):
        log = dict()

        _, indices = self.encode_to_z(x)
        if self.t_is_c:
            _, sos_tokens = self.encode_to_z(c, is_c=True)
        else:
            sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
            sos_tokens = sos_tokens.long().to(x.device)

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1], top_k=top_k, greedy=greedy)
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1], top_k=top_k, greedy=greedy)
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec, half_sample, full_sample))
    