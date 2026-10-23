import argparse, numpy as np, pandas as pd, torch
from src.utils import OUT, set_seed, compute_metrics, get_logger
from src.data import Data, make_augment
from src.models import ArogyaNet
from src.train import train_model

log = get_logger("variants")
VARIANTS = {"full": dict(residual=True, se=True, meta=False),
            "no_res": dict(residual=False, se=True, meta=False),
            "no_se": dict(residual=True, se=False, meta=False),
            "meta": dict(residual=True, se=True, meta=True)}
COLS = ["accuracy", "balanced_acc", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["full"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--pos_weight", type=float, default=None)
    ap.add_argument("--save_models", action="store_true")

            
    a = ap.parse_args()
    data = Data(a.size); tr, va = data.idx["train"], data.idx["val"]
    yv = data.y[torch.as_tensor(va, device=data.device)].cpu().numpy()
    aug = None if a.no_aug else make_augment()
    (OUT / "preds").mkdir(exist_ok=True)
    for v in a.variants:
        tag = (f"{v}_{a.size}_{'noaug' if a.no_aug else 'aug'}"+ (f"_pw{a.pos_weight:g}" if a.pos_weight else "") + ("_saved" if a.save_models else ""))
        rows, hists = [], []
        for seed in [0, 1, 2]:
            set_seed(seed)
            model = ArogyaNet(**VARIANTS[v]).cuda()
            model, hist, probs, best_ep = train_model(
                model, data, tr, va, seed, epochs=a.epochs, lr=a.lr, augment=aug, patience=8,
                pos_weight=a.pos_weight, use_meta=VARIANTS[v]["meta"], verbose=False)
            rows.append(dict(seed=seed, best_epoch=best_ep, **compute_metrics(yv, probs, 0.5)))
            hists += [dict(seed=seed, **h) for h in hist]
            np.save(OUT / "preds" / f"arogya_{tag}_s{seed}_val.npy", probs)
            if a.save_models:
                (OUT / "models").mkdir(exist_ok=True)
                torch.save(model.state_dict(), OUT / "models" / f"arogya_{tag}_s{seed}.pt")        
            log.info(f"{tag} seed {seed}: bal_acc={rows[-1]['balanced_acc']:.4f} auc={rows[-1]['roc_auc']:.4f} best_ep={best_ep}")
        df = pd.DataFrame(rows); df.to_csv(OUT / f"arogya_{tag}_runs.csv", index=False)
        pd.DataFrame(hists).to_csv(OUT / f"arogya_{tag}_history.csv", index=False)
        print(tag); print(pd.DataFrame({"mean": df[COLS].mean(), "sd": df[COLS].std()}).round(4))
