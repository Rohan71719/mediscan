import argparse, glob
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta
from src.utils import OUT, FIG, compute_metrics, get_logger
from src.data import Data
from src.models import ArogyaNet
from src.train import predict

log = get_logger("final_eval")
PI, CFN, CFP = 0.10, 20.0, 1.0
COLS = ["accuracy", "balanced_acc", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
        "camp_ppv", "camp_cost_per_1000"]

def cp(k, n, alpha=0.05):                                   # exact (Clopper-Pearson) interval
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

def camp(m):
    r, s = m["recall"], m["specificity"]
    return dict(camp_ppv=PI * r / (PI * r + (1 - PI) * (1 - s)),
                camp_cost_per_1000=1000 * (PI * (1 - r) * CFN + (1 - PI) * (1 - s) * CFP))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.4158)     # frozen in operating_point.json
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    assert a.confirm, "This evaluates the LOCKED test split. Run it once, with --confirm, after everything is frozen."
    d = Data(224); te = d.idx["test"]
    y = d.y[torch.as_tensor(te, device=d.device)].cpu().numpy().astype(int)
    fn_ = d.meta.fname.values[te]
    rows, probs = [], {}
    for s in range(3):
        h = glob.glob(f"/kaggle/input/**/arogya_full_224_aug_saved_s{s}.pt", recursive=True)
        assert h, "attach the mediscan-models dataset"
        m = ArogyaNet().to(d.device); m.load_state_dict(torch.load(h[0], map_location=d.device)); m.eval()
        probs[s] = torch.sigmoid(predict(m, d, te)).cpu().numpy()
        mt = compute_metrics(y, probs[s], a.threshold); mt.update(camp(mt))
        rows.append(dict(seed=s, **mt))
    R = pd.DataFrame(rows); R.to_csv(OUT / "test_runs.csv", index=False)
    pd.DataFrame(dict(fname=fn_, label=y, prob=probs[0])).to_csv(OUT / "test_preds_seed0.csv", index=False)

    p = probs[0]; h0 = rows[0]
    tn, fp, fn, tp = (int(h0[k]) for k in ["tn", "fp", "fn", "tp"])
    rng = np.random.RandomState(0); B = []
    for _ in range(1000):
        i = rng.randint(0, len(y), len(y)); b = compute_metrics(y[i], p[i], a.threshold); b.update(camp(b)); B.append(b)
    B = pd.DataFrame(B)
    print(f"\n== LOCKED TEST SET, seed 0, frozen threshold {a.threshold}: n = {len(y)} ({int((y==0).sum())} NORMAL, {int((y==1).sum())} PNEUMONIA) ==")
    print(f"confusion matrix: TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    for c in COLS:
        print(f"{c:20s} {h0[c]:.4f}   95% bootstrap CI [{B[c].quantile(.025):.4f}, {B[c].quantile(.975):.4f}]")
    print("recall exact 95% CI [%.4f, %.4f]; specificity exact 95%% CI [%.4f, %.4f]" % (*cp(tp, tp + fn), *cp(tn, tn + fp)))
    print("\n== robustness only (seeds 0, 1, 2) ==\n", R[["seed"] + COLS].round(4).to_string(index=False))
    print("mean ± SD:", {c: f"{R[c].mean():.4f} ± {R[c].std():.4f}" for c in COLS})
    print("for reference, seed 0 at threshold 0.5:", {k: round(v, 4) for k, v in compute_metrics(y, p, 0.5).items() if k in COLS})

    fig, ax = plt.subplots(figsize=(3.6, 3.4)); cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Blues"); ax.set_xticks([0, 1], ["pred NORMAL", "pred PNEUMONIA"]); ax.set_yticks([0, 1], ["true NORMAL", "true PNEUMONIA"])
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha="center", va="center", fontsize=12)
    ax.set_title(f"Locked test set, threshold {a.threshold}", fontsize=9); plt.tight_layout(); plt.savefig(FIG / "fig10_test_confusion.png", dpi=150)
