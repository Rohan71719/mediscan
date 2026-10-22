import argparse, re
import numpy as np, pandas as pd
from src.utils import OUT

def strict_id(f):
    m = re.match(r"(person\d+)_(bacteria|virus)_", f)
    if m: return m.group(1) + "_" + m.group(2)       # bacteria/virus numbering is separate
    m = re.match(r"(NORMAL2-IM-\d+)", f)
    if m: return m.group(1)
    m = re.match(r"(IM-\d+)", f)
    if m: return m.group(1)
    return f

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--inventory", default=str(OUT / "inventory.csv"))
    a = ap.parse_args()
    inv = pd.read_csv(a.inventory)
    inv["pid"] = inv.fname.map(strict_id)
    tr, te = inv[inv.split == "train"], inv[inv.split == "test"]
    shared = set(tr.pid) & set(te.pid)
    print("strict patient IDs in both train and test:", len(shared))
    print("test images affected: %d of %d" % (te.pid.isin(shared).sum(), len(te)))
    ex = inv[inv.duplicated("md5", keep=False)]
    ex_cross = ex.groupby("md5").split.nunique()
    print("exact-duplicate sets:", ex.md5.nunique(), "| spanning more than one split:", int((ex_cross > 1).sum()))
    cross_tt = ex.groupby("md5").split.apply(lambda s: {"train", "test"} <= set(s))
    print("exact-duplicate sets present in both train and test:", int(cross_tt.sum()))
    pd.DataFrame([{"strict_shared_ids": len(shared), "test_images_affected": int(te.pid.isin(shared).sum()),
                   "exact_dup_sets": int(ex.md5.nunique()), "exact_dup_sets_train_and_test": int(cross_tt.sum())}]
                 ).to_csv(OUT / "leakage_check.csv", index=False)
