import argparse, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils import OUT, FIG, set_seed, compute_metrics, get_logger
from src.data import Data, make_augment
from src.models import ArogyaNet
from src.train import train_model

log = get_logger("tune")

def run(data, seed, lr, bs, dropout, wd, epochs, aug, select_best, patience=8):
    set_seed(seed)
    model = ArogyaNet(dropout=dropout).cuda()
    model, hist, probs, best_ep = train_model(model, data, data.idx["train"], data.idx["val"], seed, epochs=epochs,
                                              lr=lr, wd=wd, bs=bs, augment=aug, patience=patience,
                                              select_best=select_best, verbose=False)
    return hist, probs, best_ep

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--search_size", type=int, default=96)
    ap.add_argument("--search_epochs", type=int, default=20)
    ap.add_argument("--skip_search", action="store_true")
    a = ap.parse_args()
    COLS = ["balanced_acc", "specificity", "recall", "roc_auc", "pr_auc"]

    # ---------- 1. random search (96x96 for speed, one seed, early stopping on val loss) ----------
    if not a.skip_search:
        d = Data(a.search_size); yv = d.y[torch.as_tensor(d.idx["val"], device="cuda")].cpu().numpy()
        rng = np.random.RandomState(123); rows = []
        for t in range(a.trials):
            cfg = dict(lr=float(10 ** rng.uniform(np.log10(3e-4), np.log10(5e-3))), bs=int(rng.choice([32, 64, 128])),
                       dropout=float(rng.uniform(0, 0.5)), wd=float(10 ** rng.uniform(-6, -2)))
            hist, probs, best_ep = run(d, 0, epochs=a.search_epochs, aug=make_augment(), select_best=True, patience=6, **cfg)
            m = compute_metrics(yv, probs, 0.5)
            rows.append(dict(trial=t, **cfg, best_epoch=best_ep, val_loss=min(h["val_loss"] for h in hist),
                             roc_auc=m["roc_auc"], balanced_acc=m["balanced_acc"]))
            log.info(f"trial {t}: {cfg} -> auc {m['roc_auc']:.4f} bal_acc {m['balanced_acc']:.4f}")
        S = pd.DataFrame(rows).sort_values("roc_auc", ascending=False); S.to_csv(OUT / "tune_search.csv", index=False)
        print(f"\n== search: {a.trials} random trials at {a.search_size}px, seed 0, <= {a.search_epochs} epochs, ranked by val ROC-AUC ==")
        print(S.head(8).round(4).to_string(index=False)); del d; torch.cuda.empty_cache()
    best = pd.read_csv(OUT / "tune_search.csv").iloc[0]
    tuned = dict(lr=float(best.lr), bs=int(best.bs), dropout=float(best.dropout), wd=float(best.wd))
    print("\nbest configuration:", tuned)

    # ---------- 2. overfitting curves at 224 (30 epochs, NO early stopping, seed 0) ----------
    d = Data(224); yv = d.y[torch.as_tensor(d.idx["val"], device="cuda")].cpu().numpy()
    runs = {"unregularised (no dropout, no weight decay, no augmentation)": (dict(lr=2e-3, bs=64, dropout=0.0, wd=0.0), None),
            "regularised, default (dropout 0.2, wd 1e-4, augmentation)": (dict(lr=2e-3, bs=64, dropout=0.2, wd=1e-4), make_augment()),
            "regularised, tuned (augmentation)": (tuned, make_augment())}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4), sharey=True); summ, allh = [], []
    for k, (name, (cfg, aug)) in enumerate(runs.items()):
        hist, _, _ = run(d, 0, epochs=30, aug=aug, select_best=False, **cfg)
        h = pd.DataFrame(hist); h["run"] = name; allh.append(h)
        e = int(h.val_loss.idxmin())
        ax[k].plot(h.epoch, h.train_loss, label="train loss"); ax[k].plot(h.epoch, h.val_loss, "--", label="validation loss")
        ax[k].axvline(e, c="r", ls=":", label=f"min val loss, epoch {e}"); ax[k].set_title(name, fontsize=8)
        ax[k].set_xlabel("epoch"); ax[k].set_yscale("log"); ax[k].legend(fontsize=7)
        summ.append(dict(run=name, best_epoch=e, val_loss_at_best=h.val_loss[e], val_loss_at_epoch30=h.val_loss.iloc[-1],
                         train_loss_at_best=h.train_loss[e], train_loss_at_epoch30=h.train_loss.iloc[-1],
                         epochs_saved=29 - e))
    ax[0].set_ylabel("loss"); plt.tight_layout(); plt.savefig(FIG / "fig07_overfitting_curves.png", dpi=150)
    pd.concat(allh).to_csv(OUT / "tune_curves.csv", index=False)
    S2 = pd.DataFrame(summ); S2.to_csv(OUT / "tune_overfitting_summary.csv", index=False)
    print("\n== overfitting: where it starts and what early stopping saves ==\n", S2.round(4).to_string(index=False))

    # ---------- 3. tuned vs default, 3 seeds at 224, with early stopping ----------
    out = {}
    for name, cfg in [("default", dict(lr=2e-3, bs=64, dropout=0.2, wd=1e-4)), ("tuned", tuned)]:
        rows = []
        for s in range(3):
            _, probs, be = run(d, s, epochs=30, aug=make_augment(), select_best=True, **cfg)
            rows.append(dict(seed=s, best_epoch=be, **compute_metrics(yv, probs, 0.5)))
        R = pd.DataFrame(rows); R.to_csv(OUT / f"tune_final_{name}_runs.csv", index=False)
        out[name] = pd.Series({f"{c} mean": R[c].mean() for c in COLS}) .combine_first(pd.Series({f"{c} sd": R[c].std() for c in COLS}))
    print("\n== tuned vs default configuration (224px, 3 seeds, validation) ==\n", pd.DataFrame(out).round(4))
