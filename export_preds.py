import numpy as np, pandas as pd
from src.utils import OUT
from src.data import find_cache

if __name__ == "__main__":
    meta = pd.read_csv(find_cache() / "meta.csv")
    val = meta[meta.split == "val"].reset_index(drop=True)
    out = val[["fname", "label"]].copy()
    for f in sorted((OUT / "preds").glob("*_val.npy")):      # every model/seed saved so far
        p = np.load(f)
        assert len(p) == len(val), f"{f.name}: {len(p)} rows, expected {len(val)}"
        out[f.stem.replace("_val", "")] = p
    out.to_csv(OUT / "val_preds.csv", index=False)
    print(out.shape); print(out.head())
