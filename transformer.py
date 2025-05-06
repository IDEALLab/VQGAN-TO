import torch
import torch.nn as nn
import torch.nn.functional as F
from nanogpt import GPT, GPTConfig
from vqgan import VQGAN


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token

        self.vqgan = VQGAN(args).to(device=args.device)
        self.load_vqgan(args)

        # Create config object for NanoGPT
        transformer_config = GPTConfig(
            vocab_size=args.num_codebook_vectors,
            block_size=1024,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.3,  # Add dropout parameter (default in nanoGPT)
            bias=True     # Add bias parameter (default in nanoGPT)
        )
        self.transformer = GPT(transformer_config)

        self.pkeep = args.pkeep


    def load_vqgan(self, args):
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
        
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
    

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    def forward(self, x):
        _, indices = self.encode_to_z(x)

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
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        _, indices = self.encode_to_z(x)
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