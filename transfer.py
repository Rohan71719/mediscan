import argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from src.utils import OUT, set_seed, compute_metrics, get_logger
from src.data import Data
from src.train import train_model

log = get_logger("transfer")

class Pretrained(nn.Module):
    """ResNet18 with ImageNet weights; greyscale replicated to 3 channels, ImageNet normalisation."""
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, 1); self.m = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):                                   # x: B,1,H,W in [0,1]
        return self.m((x.repeat(1, 3, 1, 1) - self.mean) / self.std)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    data = Data(224); tr, va = data.idx["train"], data.idx["val"]
    yv = data.y[torch.as_tensor(va, device=data.device)].cpu().numpy()
    (OUT / "preds").mkdir(exist_ok=True); (OUT / "models").mkdir(exist_ok=True)
    rows, hists = [], []
    for seed in [0, 1, 2]:
        set_seed(seed)
        model = Pretrained().cuda()
        model, hist, probs, best_ep = train_model(model, data, tr, va, seed, epochs=a.epochs)
        rows.append(dict(seed=seed, best_epoch=best_ep, **compute_metrics(yv, probs, 0.5)))
        hists += [dict(seed=seed, **h) for h in hist]
        np.save(OUT / "preds" / f"transfer_s{seed}_val.npy", probs)
        torch.save(model.state_dict(), OUT / "models" / f"transfer_s{seed}.pt")
        log.info(f"seed {seed}: {rows[-1]}")
    df = pd.DataFrame(rows); df.to_csv(OUT / "transfer_runs.csv", index=False)
    pd.DataFrame(hists).to_csv(OUT / "transfer_history.csv", index=False)
    cols = ["accuracy", "balanced_acc", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
    summ = pd.DataFrame({"mean": df[cols].mean(), "sd": df[cols].std()}).round(4)
    summ.to_csv(OUT / "transfer_summary.csv"); print(summ)
