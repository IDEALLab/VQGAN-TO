import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.c_num_codebook_vectors if args.is_c else args.num_codebook_vectors
        self.latent_dim = args.c_latent_dim if args.is_c else args.latent_dim
        self.beta = args.beta
        self.no_vq = getattr(args, "no_vq", False)

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        if args.codebook_mod_init:
            half = self.num_codebook_vectors // 2
            low = torch.empty(half, self.latent_dim).uniform_(0.0, 0.2)
            high = torch.empty(self.num_codebook_vectors - half, self.latent_dim).uniform_(0.8, 1.0)
            self.embedding.weight.data = torch.cat([low, high], dim=0)
        else:
            self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)
    def forward(self, z):
        if self.no_vq:
            # Skip quantization, return z directly
            z_q = z

            # Compute indices for logging only
            with torch.no_grad():
                z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_dim)
                d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(z_flattened, self.embedding.weight.t())
                min_encoding_indices = torch.argmin(d, dim=1)

            # Return dummy loss for logging and compatibility
            loss = torch.tensor(0.0, device=z.device)
        else:
            z = z.permute(0, 2, 3, 1).contiguous()
            z_flattened = z.view(-1, self.latent_dim)

            d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - \
                2*(torch.matmul(z_flattened, self.embedding.weight.t()))

            min_encoding_indices = torch.argmin(d, dim=1)
            z_q = self.embedding(min_encoding_indices).view(z.shape)

            loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

            z_q = z + (z_q - z).detach()

            z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss