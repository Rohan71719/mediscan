import json
import numpy as np, pandas as pd, torch
from src.utils import OUT, set_seed, compute_metrics, get_logger
from src.data import Data, make_augment
from src.models import ArogyaNet
from src.train import train_model
from threshold import rates_grid, cost_per_1000, pick     # the same cost-matrix rule as Task 1

log = get_logger("train_fix")
COLS = ["balanced_acc", "specificity", "recall", "roc_auc", "pr_auc"]

if __name__ == "__main__":
    d = Data(224); tr, va = d.idx["train"], d.idx["val"]                  # train and validation only
    yv = d.y[torch.as_tensor(va, device=d.device)].cpu().numpy().astype(int)
    aug = make_augment(rot=10, zoom=0.1, bright=0.30, contrast=0.40)      # the pre-declared fix
    (OUT / "models").mkdir(exist_ok=True); (OUT / "preds").mkdir(exist_ok=True)
    rows, P = [], []
    for s in range(3):
        set_seed(s); m = ArogyaNet().cuda()
        m, _, probs, be = train_model(m, d, tr, va, s, epochs=30, lr=2e-3, wd=1e-4, bs=64,
                                      augment=aug, patience=8, verbose=False)
        torch.save(m.state_dict(), OUT / "models" / f"arogya_fix_s{s}.pt")
        np.save(OUT / "preds" / f"arogya_fix_s{s}_val.npy", probs); P.append(probs)
        rows.append(dict(seed=s, best_epoch=be, **compute_metrics(yv, probs, 0.5)))
        log.info(f"seed {s}: bal_acc={rows[-1]['balanced_acc']:.4f} auc={rows[-1]['roc_auc']:.4f}")
    R = pd.DataFrame(rows); R.to_csv(OUT / "fix_runs.csv", index=False)
    print("== fixed model, validation, threshold 0.5 (default model: bal acc 0.9542 ± 0.0042, ROC-AUC 0.9919 ± 0.0009) ==")
    print(pd.DataFrame({"mean": R[COLS].mean(), "sd": R[COLS].std()}).round(4))
    rec, spec = rates_grid(np.tile(yv, 3), np.concatenate(P)); c = cost_per_1000(rec, spec)
    t = float(pick(c)); i = int(np.argmin(np.abs(np.array([float(x) for x in rates_grid.__defaults__[0]]) - t)))
    print(f"\nthreshold chosen on validation by the Task 1 rule: {t:.4f}  (recall {rec[i]:.4f}, specificity {spec[i]:.4f}, cost {c[i]:.1f} per 1,000)")
    json.dump(dict(model="ArogyaNet, strong brightness/contrast augmentation", threshold=t), open(OUT / "operating_point_fix.json", "w"), indent=2)
