import os
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from collections import namedtuple
import requests
from tqdm import tqdm


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def get_ckpt_path(name, root):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path):
        print(f"Downloading {name} model from {URL_MAP[name]} to {path}")
        download(URL_MAP[name], path)
    return path


class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.vgg = VGG16()
        self.lins = nn.ModuleList([
            NetLinLayer(self.channels[0]),
            NetLinLayer(self.channels[1]),
            NetLinLayer(self.channels[2]),
            NetLinLayer(self.channels[3]),
            NetLinLayer(self.channels[4])
        ])

        self.load_from_pretrained()

        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "vgg_lpips")
        state_dict = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True)
        self.load_state_dict(state_dict, strict=False)

    def forward(self, real_x, fake_x):
        features_real = self.vgg(self.scaling_layer(real_x))
        features_fake = self.vgg(self.scaling_layer(fake_x))
        diffs = {}

        for i in range(len(self.channels)):
            diffs[i] = (norm_tensor(features_real[i]) - norm_tensor(features_fake[i])) ** 2

        return sum([spatial_average(self.lins[i].model(diffs[i])) for i in range(len(self.channels))])


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        )


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        slices = [vgg_pretrained_features[i] for i in range(30)]
        self.slice1 = nn.Sequential(*slices[0:4])
        self.slice2 = nn.Sequential(*slices[4:9])
        self.slice3 = nn.Sequential(*slices[9:16])
        self.slice4 = nn.Sequential(*slices[16:23])
        self.slice5 = nn.Sequential(*slices[23:30])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        vgg_outputs = namedtuple("VGGOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        return vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


def norm_tensor(x):
    """
    Normalize images by their length to make them unit vector?
    :param x: batch of images
    :return: normalized batch of images
    """
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + 1e-10)


def spatial_average(x):
    """
     imgs have: batch_size x channels x width x height --> average over width and height channel
    :param x: batch of images
    :return: averaged images along width and height
    """
    return x.mean([2, 3], keepdim=True)

class GreyscaleLPIPS(nn.Module):
    def __init__(self, use_raw=True, clamp_output=False, robust_clamp=True, warn_on_clamp=False):
        super().__init__()
        self.use_raw = use_raw
        self.clamp_output = clamp_output
        self.robust_clamp = robust_clamp
        self.warn_on_clamp = warn_on_clamp

        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.vgg = VGG16()

        self.lins = nn.ModuleList([
            NetLinLayer(self.channels[0]),
            NetLinLayer(self.channels[1]),
            NetLinLayer(self.channels[2]),
            NetLinLayer(self.channels[3]),
            NetLinLayer(self.channels[4])
        ])

        self.load_from_pretrained()

        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "vgg_lpips")
        state_dict = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True)
        self.load_state_dict(state_dict, strict=False)

    def forward(self, real_x, fake_x):
        if self.warn_on_clamp:
            with torch.no_grad():
                if (fake_x < 0).any() or (fake_x > 1).any():
                    print(f"Warning: Generated image contains values outside [0,1] range: [{fake_x.min().item():.4f}, {fake_x.max().item():.4f}]")
                if (real_x < 0).any() or (real_x > 1).any():
                    print(f"Warning: Reference image contains values outside [0,1] range: [{real_x.min().item():.4f}, {real_x.max().item():.4f}]")

        if self.robust_clamp:
            real_x = torch.clamp(real_x, 0.0, 1.0)
            fake_x = torch.clamp(fake_x, 0.0, 1.0)

        # Convert grayscale to RGB by duplicating channel if needed
        if real_x.shape[1] == 1:
            real_x = real_x.repeat(1, 3, 1, 1)
        if fake_x.shape[1] == 1:
            fake_x = fake_x.repeat(1, 3, 1, 1)

        features_real = self.vgg(self.scaling_layer(real_x))
        features_fake = self.vgg(self.scaling_layer(fake_x))

        diffs = [(norm_tensor(fr) - norm_tensor(ff)) ** 2 for fr, ff in zip(features_real, features_fake)]

        if self.use_raw:
            loss = sum([
                spatial_average(d).mean(dim=1, keepdim=True)  # reduce channel dimension
                for d in diffs
            ])
        else:
            loss = sum([
                spatial_average(self.lins[i].model(d))
                for i, d in enumerate(diffs)
            ])

        if self.clamp_output:
            loss = torch.clamp(loss, min=0.0)

        return loss
