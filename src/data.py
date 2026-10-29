import glob
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from src.utils import find_files


def find_cache():
    hits = find_files("meta.csv")
    if not hits: raise FileNotFoundError("attach the mediscan-cache dataset")
    return Path(hits[0]).parent

class Data:
    """All images live on the GPU as uint8. Test indices exist but train/eval code must not use them."""
    def __init__(self, size=224, device="cuda"):
        c = find_cache(); self.device = device
        self.X = torch.from_numpy(np.load(c / f"X{size}.npy")).to(device)      # N,H,W uint8
        self.y = torch.tensor(np.load(c / "y.npy"), dtype=torch.float32, device=device)
        self.meta = pd.read_csv(c / "meta.csv")
        self.idx = {s: np.where(self.meta.split.values == s)[0] for s in ["train", "val", "test"]}
        M = np.stack([np.log(self.meta.area.values), self.meta.aspect.values,
                      self.meta.mean_int.values], 1).astype(np.float32)
        tr = self.idx["train"]; mu, sd = M[tr].mean(0), M[tr].std(0)           # train-only statistics
        self.M = torch.tensor((M - mu) / sd, device=device)                    # for the two-branch model

    def batch(self, ids, augment=None):
        ids = torch.as_tensor(ids, device=self.device)
        x = self.X[ids].float().div(255).unsqueeze(1)                          # B,1,H,W in [0,1]
        if augment is not None: x = augment(x)
        return x, self.y[ids], self.M[ids]

def make_augment(rot=10, zoom=0.1, bright=0.1, contrast=0.1):
    """Small rotation, zoom, brightness and contrast jitter. Flips are deliberately NOT implemented."""
    def f(x):
        B, dev = x.shape[0], x.device
        ang = (torch.rand(B, device=dev) * 2 - 1) * rot * 3.14159265 / 180
        sc = 1 + (torch.rand(B, device=dev) * 2 - 1) * zoom
        th = torch.zeros(B, 2, 3, device=dev)
        th[:, 0, 0] = torch.cos(ang) / sc; th[:, 0, 1] = -torch.sin(ang) / sc
        th[:, 1, 0] = torch.sin(ang) / sc; th[:, 1, 1] = torch.cos(ang) / sc
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False),
                          padding_mode="border", align_corners=False)
        b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * bright
        c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * contrast
        m = x.mean((1, 2, 3), keepdim=True)
        return ((x - m) * c + m + b).clamp(0, 1)
    return f
