import argparse, hashlib, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from tqdm import tqdm
from src.utils import OUT, FIG, find_primary_root, get_logger, set_seed

log = get_logger("audit")
NEAR_THR = 5   # Hamming distance (of 64 bits) counted as near-duplicate

def patient_id(fname, label):
    # PNEUMONIA: person123_bacteria_456.jpeg -> person123
    m = re.match(r"(person\d+)_", fname)
    if m: return m.group(1)
    # NORMAL: IM-0115-0001.jpeg or NORMAL2-IM-1427-0001.jpeg -> IM-0115 / IM-1427
    m = re.search(r"(IM-\d+)", fname)
    if m: return m.group(1)
    return fname   # unknown -> image is its own group

def build_inventory(root, debug=False):
    rows = []
    for split in ["train", "test", "val"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            files = sorted((root / split / cls).glob("*.jp*g"))
            if debug: files = files[:30]
            for f in tqdm(files, desc=f"{split}/{cls}"):
                im = Image.open(f)
                mode = im.mode
                w, h = im.size
                arr = np.asarray(im.convert("RGB"))
                true_grey = (mode == "L") or bool(
                    (arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all())
                g = np.asarray(im.convert("L"), dtype=np.float32)
                rows.append(dict(
                    path=str(f), fname=f.name, split=split, cls=cls,
                    label=int(cls == "PNEUMONIA"), patient=patient_id(f.name, cls),
                    h=h, w=w, aspect=w / h, area=w * h, mode=mode,
                    true_grey=true_grey, mean_int=g.mean(), std_int=g.std(),
                    filesize=f.stat().st_size,
                    md5=hashlib.md5(f.read_bytes()).hexdigest(),
                    phash=str(imagehash.phash(im.convert("L")))))
    return pd.DataFrame(rows)

def inventory_report(df):
    print(df.groupby(["split", "cls"]).size().unstack())
    print("\nModes:\n", df["mode"].value_counts())
    print("\nTrue greyscale:", df["true_grey"].sum(), "of", len(df))
    print("\nStats by class:\n", df.groupby("cls")[["h", "w", "aspect", "area", "mean_int", "std_int"]].describe().T)
    df.groupby(["split", "cls"]).size().unstack().to_csv(OUT / "counts.csv")
    df.groupby("cls")[["h", "w", "aspect", "area", "mean_int", "std_int"]].describe().to_csv(OUT / "pixel_stats_by_class.csv")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for i, (col, title) in enumerate([("h", "Height"), ("w", "Width"), ("aspect", "Aspect ratio"),
                                      ("area", "Area (px)"), ("mean_int", "Mean brightness"), ("std_int", "Contrast (std)")]):
        a = ax.flat[i]
        for cls, c in [("NORMAL", "tab:blue"), ("PNEUMONIA", "tab:red")]:
            a.hist(df[df.cls == cls][col], bins=40, alpha=0.55, label=cls, color=c, density=True)
        a.set_title(title); a.legend()
    plt.tight_layout(); plt.savefig(FIG / "fig01_pixel_stats_by_class.png", dpi=150); plt.close()

    ax = df.groupby(["split", "cls"]).size().unstack().plot(kind="bar", figsize=(6, 4))
    plt.title("Images per split and class"); plt.tight_layout()
    plt.savefig(FIG / "fig02_counts.png", dpi=150); plt.close()

def duplicate_report(df):
    # exact duplicates (identical bytes)
    dup = df[df.duplicated("md5", keep=False)].sort_values("md5")
    dup.to_csv(OUT / "exact_duplicates.csv", index=False)
    print("\nExact-duplicate files:", len(dup))

    # near duplicates via pHash Hamming distance
    H = np.array([[int(b) for b in format(int(h, 16), "064b")] for h in df.phash], dtype=np.float32)
    ham = H.shape[1] - (H @ H.T + (1 - H) @ (1 - H).T)      # pairwise Hamming
    np.fill_diagonal(ham, 99)
    iu = np.argwhere(np.triu(ham <= NEAR_THR, k=1))
    pairs = pd.DataFrame({
        "i": iu[:, 0], "j": iu[:, 1],
        "dist": ham[iu[:, 0], iu[:, 1]],
        "split_i": df.split.values[iu[:, 0]], "split_j": df.split.values[iu[:, 1]],
        "cls_i": df.cls.values[iu[:, 0]], "cls_j": df.cls.values[iu[:, 1]],
        "patient_i": df.patient.values[iu[:, 0]], "patient_j": df.patient.values[iu[:, 1]]})
    pairs["same_split"] = pairs.split_i == pairs.split_j
    pairs["label_conflict"] = pairs.cls_i != pairs.cls_j
    pairs.to_csv(OUT / "near_duplicate_pairs.csv", index=False)

    print(f"Near-duplicate pairs (dist<={NEAR_THR}): {len(pairs)}")
    print("  within same split:", int(pairs.same_split.sum()))
    cross = pairs[(~pairs.same_split) & (pairs[["split_i", "split_j"]].apply(set, axis=1).apply(lambda s: s == {"train", "test"}))]
    print("  train<->test:", len(cross))
    print("  label conflicts:", int(pairs.label_conflict.sum()))

    # patient overlap between train and test
    ptr = set(df[df.split == "train"].patient); pte = set(df[df.split == "test"].patient)
    print("Patients appearing in both train and test:", len(ptr & pte))

    # worst five cross-split pairs (smallest distance)
    worst = cross.sort_values("dist").head(5) if len(cross) else pairs.sort_values("dist").head(5)
        if len(worst) == 0:
            print("No near-duplicate pairs found; skipping worst-5 figure.")
            return
    fig, ax = plt.subplots(len(worst), 2, figsize=(6, 3 * len(worst)))
    ax = np.atleast_2d(ax)
    for r, (_, p) in enumerate(worst.iterrows()):
        for c, idx in enumerate([int(p.i), int(p.j)]):
            ax[r, c].imshow(Image.open(df.path.iloc[idx]).convert("L"), cmap="gray")
            ax[r, c].set_title(f"{df.split.iloc[idx]}/{df.cls.iloc[idx]}\nd={int(p.dist)}", fontsize=8)
            ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(FIG / "fig03_worst5_pairs.png", dpi=150); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    set_seed(0)
    root = find_primary_root()
    df = build_inventory(root, a.debug)
    df.to_csv(OUT / "inventory.csv", index=False)
    inventory_report(df)
    duplicate_report(df)
    log.info("audit done")
