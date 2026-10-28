import argparse, glob, json
import numpy as np, torch
from sklearn.covariance import LedoitWolf
from src.utils import OUT
from src.data import Data
from src.models import ArogyaNet

@torch.no_grad()
def embed(m, d, ids, bs=256):
    out = []
    for i in range(0, len(ids), bs):
        x, _, _ = d.batch(ids[i:i + bs])
        out.append(m.features(m.stem((x - 0.5) / 0.25)).mean((2, 3)).float())     # fp32, as predict.py
    return torch.cat(out).cpu().numpy().astype(np.float64)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threshold", type=float, default=0.4158)
    ap.add_argument("--band_lo", type=float, default=0.10); ap.add_argument("--band_hi", type=float, default=0.90)
    ap.add_argument("--gate_q", type=float, default=0.99)
    a = ap.parse_args()
    d = Data(224); tr, va = d.idx["train"], d.idx["val"]                       # the test split is not used
    w = glob.glob(f"/kaggle/input/**/arogya_full_224_aug_saved_s{a.seed}.pt", recursive=True); assert w
    m = ArogyaNet().cuda(); m.load_state_dict(torch.load(w[0], map_location="cuda")); m.eval()
    Etr, Eva = embed(m, d, tr), embed(m, d, va)
    lw = LedoitWolf().fit(Etr); mu, prec = lw.location_, lw.precision_
    d2 = lambda E: np.einsum("ij,jk,ik->i", E - mu, prec, E - mu)
    thr = float(np.quantile(d2(Eva), a.gate_q))
    print(f"squared Mahalanobis distance: train median {np.median(d2(Etr)):.0f}, validation median {np.median(d2(Eva)):.0f}, "
          f"validation {a.gate_q:.0%} quantile (gate cutoff) {thr:.0f}")
    o = OUT / "model"; o.mkdir(exist_ok=True)
    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, o / "arogyanet.pt")
    np.savez(o / "gate_stats.npz", mu=mu.astype(np.float32), prec=prec.astype(np.float32))
    json.dump(dict(model=f"ArogyaNet full, default configuration, seed {a.seed}", threshold=a.threshold, band_lo=a.band_lo,
                   band_hi=a.band_hi, gate_d2=thr, gate_quantile=a.gate_q, min_std=10.0, max_colour_diff=6.0),
              open(o / "config.json", "w"), indent=2)
