import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from src.utils import OUT, find_primary_root, get_logger

from concurrent.futures import ThreadPoolExecutor
import os

def load(p):
    im = Image.open(p)
    im.draft("L", (448, 448))
    im = im.convert("L")
    return [np.asarray(im.resize((s, s), Image.BILINEAR, reducing_gap=2.0), dtype=np.uint8) for s in SIZES]

log = get_logger("cache")
SIZES = [96, 224]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="path to split.csv")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    split_path = a.split or str(OUT / "split.csv")
    df = pd.read_csv(split_path)
    if a.debug: df = df.groupby("split").head(20).reset_index(drop=True)
    root = find_primary_root()           # rebuild paths so they work in any notebook
    paths = [root / r.orig_split / r.cls / r.fname for r in df.itertuples()]
    missing = [p for p in paths if not p.exists()]
    assert not missing, f"{len(missing)} files not found, e.g. {missing[:2]}"

    cache = OUT / "cache"; cache.mkdir(exist_ok=True, parents=True)
    arrays = {s: np.zeros((len(df), s, s), dtype=np.uint8) for s in SIZES}
    with ThreadPoolExecutor(max_workers=16) as ex:   # threads hide slow disk reads
        for i, out in enumerate(tqdm(ex.map(load, paths), total=len(paths))):
            for s, arr in zip(SIZES, out):
                arrays[s][i] = arr

    for s in SIZES:
        np.save(cache / f"X{s}.npy", arrays[s])
    np.save(cache / "y.npy", df.label.values.astype(np.int64))
    df[["path", "fname", "cls", "label", "patient", "group", "split", "fold",
        "h", "w", "aspect", "area", "mean_int"]].to_csv(cache / "meta.csv", index=False)
    log.info(f"cached {len(df)} images -> {cache}")
    for s in SIZES:
        print(s, arrays[s].shape, "mean pixel:",
              {c: round(float(arrays[s][df.cls.values == c].mean()), 1) for c in ["NORMAL", "PNEUMONIA"]})
