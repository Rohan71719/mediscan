import os, random, json, logging
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix)

# ---------- paths (works on Kaggle and locally) ----------
ON_KAGGLE = os.path.exists("/kaggle/input")
ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(".")
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def find_primary_root():
    """Locate the folder containing train/ test/ val/ of the primary dataset."""
    base = Path("/kaggle/input")
    for p in base.rglob("train"):
        if (p.parent / "test").exists() and (p / "NORMAL").exists():
            return p.parent
    raise FileNotFoundError("Primary dataset not attached")

# ---------- seeding ----------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Seed set to {seed}")
    with open(OUT / "seed_log.txt", "a") as f:
        f.write(f"{seed}\n")

# ---------- logging ----------
def get_logger(name):
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)

# ---------- metrics ----------
def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
