import argparse, glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils import OUT, FIG, get_logger
from src.data import find_cache
from src.models import ArogyaNet

log = get_logger("external")

def find_external():
    out = []
    for f in glob.glob("/kaggle/input/**/*.png", recursive=True):
        parts = [p.lower() for p in Path(f).parts]
        if "masks" in parts: continue
        if "normal" in parts: out.append((f, 0))
        elif any(p.startswith("viral") for p in parts): out.append((f, 1))
    return sorted(out)

def load(f):                                       # same preprocessing as cache_data.py
    im = Image.open(f); w, h = im.size
    g = im.convert("L").resize((224, 224), Image.BILINEAR, reducing_gap=2.0)
    return np.asarray(g, dtype=np.uint8), h, w

def thumbs(X, dev):                                # 224 -> 32x32, mean-centred, unit norm
    out = []
    for i in range(0, len(X), 1024):
        t = torch.from_numpy(np.ascontiguousarray(X[i:i + 1024])).to(dev).float().reshape(-1, 32, 7, 32, 7).mean((2, 4)).reshape(-1, 1024)
        t = t - t.mean(1, keepdim=True); out.append(t / t.norm(dim=1, keepdim=True).clamp_min(1e-6))
    return torch.cat(out)

@torch.no_grad()
def probs_of(model, X, dev):
    out = []
    for i in range(0, len(X), 256):
        x = torch.from_numpy(X[i:i + 256]).to(dev).float().div(255).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x).float().squeeze(1))
    return torch.sigmoid(torch.cat(out)).cpu().numpy()

def ev(y, p, thr, auc=True):
    y = np.asarray(y)
    if len(y) == 0: return {}
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
    two = (tp + fn > 0) and (tn + fp > 0)
    return dict(n=len(y), n_pos=tp + fn, n_neg=tn + fp, tp=tp, fn=fn, tn=tn, fp=fp, recall=rec, specificity=spec,
                accuracy=(tp + tn) / len(y), balanced_acc=(rec + spec) / 2 if two else np.nan,
                roc_auc=roc_auc_score(y, p) if (two and auc) else np.nan,
                pr_auc=average_precision_score(y, p) if (two and auc) else np.nan)

def cp(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

def paired(y, pa, ta, pb, tb, key, B=1000):         # gain of model b over model a, same resamples
    rng = np.random.RandomState(0); d = []
    for _ in range(B):
        i = rng.randint(0, len(y), len(y))
        d.append(ev(y[i], pb[i], tb, False)[key] - ev(y[i], pa[i], ta, False)[key])
    d = np.array(d); return np.nanmean(d), np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)

def img_stats(Xa):
    m, s = np.empty(len(Xa), np.float32), np.empty(len(Xa), np.float32)
    for i in range(0, len(Xa), 1024):
        b = np.asarray(Xa[i:i + 1024], dtype=np.float32); m[i:i + 1024] = b.mean((1, 2)); s[i:i + 1024] = b.std((1, 2))
    return m, s

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr_default", type=float, default=0.4158); ap.add_argument("--thr_fix", type=float, default=0.5498)
    ap.add_argument("--dup_corr", type=float, default=0.97)
    ap.add_argument("--val_bal_default", type=float, default=0.9542); ap.add_argument("--val_bal_fix", type=float, default=0.9457)
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    assert a.confirm, "This reads the EXTERNAL dataset. Run it once, with --confirm, after everything is frozen."
    assert not (OUT / "external_done.txt").exists(), "already run once in this session"
    dev = "cuda"; THR = {"default": a.thr_default, "fix": a.thr_fix}

    files = find_external(); assert files, "attach the external dataset"
    y = np.array([c for _, c in files]); names = np.array([Path(f).name for f, _ in files])
    with ThreadPoolExecutor(16) as ex: res = list(ex.map(load, [f for f, _ in files]))
    X = np.stack([r[0] for r in res]); H = np.array([r[1] for r in res]); W = np.array([r[2] for r in res])
    print(f"external images found: {len(y)}  (Normal {int((y == 0).sum())}, Viral {int((y == 1).sum())})")

    # ---------- 1. overlap with the primary data ----------
    c = find_cache(); Xp = np.load(c / "X224.npy", mmap_mode="r"); mp = pd.read_csv(c / "meta.csv")
    Tp, Te = thumbs(Xp, dev), thumbs(X, dev)
    rs = np.random.RandomState(0).choice(len(Xp), 1000, replace=False)
    cq = Tp[rs] @ Tp.T; cq[torch.arange(len(rs)), torch.as_tensor(rs, device=dev)] = -1
    print("calibration: primary-to-nearest-other-primary correlation, quantiles 50/90/99/100%:",
          np.quantile(cq.max(1).values.cpu().numpy(), [.5, .9, .99, 1.0]).round(4))
    mx, am = [], []
    for i in range(0, len(Te), 2048):
        v, j = (Te[i:i + 2048] @ Tp.T).max(1); mx.append(v.cpu()); am.append(j.cpu())
    mx, am = torch.cat(mx).numpy(), torch.cat(am).numpy()
    for t in [0.95, 0.97, 0.98, 0.99]:
        print(f"near-duplicates of a primary image at correlation >= {t}: Normal {int(((mx >= t) & (y == 0)).sum())}, Viral {int(((mx >= t) & (y == 1)).sum())}")
    flag = mx >= a.dup_corr; msplit = mp.split.values[am]
    print(f"flagged at {a.dup_corr}: {int(flag.sum())} of {len(y)}; matched primary split: {pd.Series(msplit[flag]).value_counts().to_dict()}")

    # ---------- 2. predictions of the frozen models ----------
    def weights(name):
        h = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert h, f"missing {name}"; return h[0]
    P = {}
    for kind, pat in [("default", "arogya_full_224_aug_saved_s{}.pt"), ("fix", "arogya_fix_s{}.pt")]:
        for s in range(3):
            m = ArogyaNet().to(dev); m.load_state_dict(torch.load(weights(pat.format(s)), map_location=dev)); m.eval()
            P[(kind, s)] = probs_of(m, X, dev)

    # ---------- 3. evaluation ----------
    sets = {"ALL external": np.ones(len(y), bool), "INDEPENDENT (headline)": ~flag,
            "duplicates of primary train": flag & (msplit == "train"), "duplicates of primary val/test": flag & (msplit != "train")}
    rows = [dict(set=sn, model=k, seed=s, **ev(y[mk], P[(k, s)][mk], THR[k]))
            for sn, mk in sets.items() if mk.sum() for k in ["default", "fix"] for s in range(3)]
    R = pd.DataFrame(rows); R.to_csv(OUT / "external_runs.csv", index=False)
    show = ["n", "n_pos", "n_neg", "recall", "specificity", "balanced_acc", "roc_auc", "fp", "fn"]
    print("\n== seed 0 at the frozen thresholds ==\n", R[R.seed == 0][["set", "model"] + show].round(4).to_string(index=False))
    print("\n== mean over 3 seeds ==\n", R.groupby(["set", "model"])[show].mean().round(4).to_string())

    ind = ~flag; yi = y[ind]; pd0, pf0 = P[("default", 0)][ind], P[("fix", 0)][ind]
    h = ev(yi, pd0, a.thr_default)
    print(f"\nHEADLINE, default model, seed 0, independent subset (n={h['n']}, {h['n_pos']} positive, {h['n_neg']} negative):")
    if h["n_neg"]: print("  specificity %.4f, exact 95%% CI [%.4f, %.4f]  (internal test: 0.9076)" % (h["specificity"], *cp(h["tn"], h["n_neg"])))
    if h["n_pos"]: print("  recall %.4f, exact 95%% CI [%.4f, %.4f]  (internal test: 0.9837)" % (h["recall"], *cp(h["tp"], h["n_pos"])))
    if h["n_pos"] and h["n_neg"]: print("  balanced accuracy %.4f (internal test: 0.9456), ROC-AUC %.4f (0.9937)" % (h["balanced_acc"], h["roc_auc"]))

    key = "balanced_acc" if h.get("n_pos", 0) >= 30 and h.get("n_neg", 0) >= 30 else "specificity"
    g, lo, hi = paired(yi, pd0, a.thr_default, pf0, a.thr_fix, key)
    ok_val = (a.val_bal_default - a.val_bal_fix) <= 0.01
    print(f"\nFIX vs default (seed 0, independent subset), gain in {key}: {g:+.4f}, paired-bootstrap 95% interval [{lo:+.4f}, {hi:+.4f}]")
    print("pre-declared rule:", "FIX HELPS" if (lo > 0 and ok_val) else "FIX DOES NOT CLEARLY HELP", f"(validation condition met: {ok_val})")

    # ---------- 4. pixel statistics for the explanation ----------
    mp_, sp_ = img_stats(Xp); me_, se_ = img_stats(X); tr = (mp.split == "train").values
    G = {"primary NORMAL (train)": tr & (mp.label.values == 0), "primary PNEUMONIA (train)": tr & (mp.label.values == 1)}
    D = pd.DataFrame(dict(area=np.r_[mp.area.values, H * W], aspect=np.r_[mp.aspect.values, W / H],
                          brightness=np.r_[mp_, me_], contrast=np.r_[sp_, se_]))
    n0 = len(mp); masks = {k: np.r_[v, np.zeros(len(y), bool)] for k, v in G.items()}
    for nm, mk in [("external Normal, independent", (y == 0) & ~flag), ("external Normal, duplicates", (y == 0) & flag),
                   ("external Viral, independent", (y == 1) & ~flag), ("external Viral, duplicates", (y == 1) & flag)]:
        masks[nm] = np.r_[np.zeros(n0, bool), mk]
    S = pd.DataFrame({k: D[m].median().assign(n=int(m.sum())) for k, m in masks.items() if m.sum()}).T
    S.to_csv(OUT / "external_pixel_stats.csv"); print("\n== median pixel statistics (224x224 arrays; area/aspect from original files) ==\n", S.round(2).to_string())
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for k, col in enumerate(["area", "aspect", "brightness", "contrast"]):
        for nm, c_ in [("primary NORMAL (train)", "tab:blue"), ("primary PNEUMONIA (train)", "tab:red"), ("external Normal, independent", "tab:green")]:
            if masks[nm].sum(): ax[k].hist(D[col][masks[nm]], bins=40, alpha=0.5, density=True, color=c_, label=nm)
        ax[k].set_title(col); 
    ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(FIG / "fig11_external_pixel_stats.png", dpi=150); plt.close()
    if ind.sum():
        plt.figure(figsize=(5.5, 3.6)); plt.hist(pd0[yi == 0], bins=40, alpha=0.6, label="default"); plt.hist(pf0[yi == 0], bins=40, alpha=0.6, label="fix")
        plt.yscale("log"); plt.axvline(a.thr_default, c="C0", ls=":"); plt.axvline(a.thr_fix, c="C1", ls=":"); plt.xlabel("predicted probability of pneumonia")
        plt.title("External independent NORMAL images"); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "fig12_external_probs.png", dpi=150)

    out = pd.DataFrame(dict(fname=names, label=y, max_corr=mx, matched_split=msplit, flagged=flag))
    for (k, s), p in P.items(): out[f"p_{k}_s{s}"] = p
    out.to_csv(OUT / "external_preds.csv", index=False); (OUT / "external_done.txt").write_text("external evaluation run once\n")
