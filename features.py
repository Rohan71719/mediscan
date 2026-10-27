import argparse, glob
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.utils import OUT, FIG, compute_metrics, get_logger
from src.data import Data
from src.models import ArogyaNet

log = get_logger("features")

def weights(seed):
    h = glob.glob(f"/kaggle/input/**/arogya_full_224_aug_saved_s{seed}.pt", recursive=True)
    assert h, f"weights for seed {seed} not found: attach the mediscan-models dataset"
    return h[0]

@torch.no_grad()
def embed(model, data, ids, bs=256):
    model.eval(); out = []
    for i in range(0, len(ids), bs):
        x, _, _ = data.batch(ids[i:i + bs])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model.features(model.stem((x - 0.5) / 0.25)).mean((2, 3)).float())
    return torch.cat(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--threshold", type=float, default=0.4158)
    a = ap.parse_args()
    d = Data(224); dev = d.device; tr, va = d.idx["train"], d.idx["val"]       # the test split is never used
    ytr = d.y[torch.as_tensor(tr, device=dev)].cpu().numpy().astype(int)
    yv = d.y[torch.as_tensor(va, device=dev)].cpu().numpy().astype(int)
    vm = d.meta.iloc[va].reset_index(drop=True)
    out = pd.DataFrame(dict(fname=vm.fname.values, label=yv)); rows = []

    # ---------- Task 4: Random Forest and AdaBoost on penultimate-layer embeddings (3 seeds) ----------
    for s in range(3):
        m = ArogyaNet().to(dev); m.load_state_dict(torch.load(weights(s), map_location=dev)); m.eval()
        Etr, Eva = embed(m, d, tr), embed(m, d, va)
        with torch.no_grad(): head = torch.sigmoid(m.fc(Eva)).squeeze(1).cpu().numpy()
        Etr, Eva = Etr.cpu().numpy(), Eva.cpu().numpy()
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=s).fit(Etr, ytr)
        ada = AdaBoostClassifier(n_estimators=200, random_state=s).fit(Etr, ytr)
        out[f"emb_cnnhead_s{s}"] = head
        out[f"emb_rf_s{s}"] = rf.predict_proba(Eva)[:, 1]
        out[f"emb_ada_s{s}"] = ada.predict_proba(Eva)[:, 1]
        for n in ["cnnhead", "rf", "ada"]:
            rows.append(dict(model=n, seed=s, **compute_metrics(yv, out[f"emb_{n}_s{s}"].values, 0.5)))
        if s == 0: E0, p0 = Eva, head
        log.info(f"seed {s} done")
    out.to_csv(OUT / "val_preds_embed.csv", index=False)
    R = pd.DataFrame(rows); R.to_csv(OUT / "embed_models_runs.csv", index=False)
    print("\n== CNN head vs Random Forest vs AdaBoost on embeddings (validation, threshold 0.5, mean ± SD over 3 seeds) ==")
    print(R.groupby("model")[["balanced_acc", "specificity", "recall", "roc_auc", "pr_auc"]].agg(["mean", "std"]).round(4).T)

    # ---------- Task 5: feature space of the seed-0 model on the validation split ----------
    pred = (p0 >= a.threshold).astype(int); wrong = pred != yv
    Zp = PCA(2, random_state=0).fit_transform(E0)
    Zt = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(E0)
    cl = KMeans(2, n_init=10, random_state=0).fit(E0).labels_
    ct = pd.crosstab(cl, yv); purity = ct.max(axis=1).sum() / len(yv)
    print("\n== k-means (k=2) on the 256-d embeddings vs true labels ==\n", ct)
    print(f"purity {purity:.3f} (a clustering that ignores the data scores {max(yv.mean(), 1 - yv.mean()):.3f}), "
          f"adjusted Rand index {adjusted_rand_score(yv, cl):.3f}, NMI {normalized_mutual_info_score(yv, cl):.3f}")
    print("misclassifications per k-means cluster:\n", pd.crosstab(cl, wrong, rownames=["cluster"], colnames=["wrong"]))

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    def scat(a_, Z, masks, names, cols, title):
        for mk, n, c in zip(masks, names, cols):
            a_.scatter(Z[mk, 0], Z[mk, 1], s=8, c=c, alpha=0.7, label=f"{n} ({mk.sum()})")
        a_.set_title(title, fontsize=9); a_.legend(fontsize=7); a_.set_xticks([]); a_.set_yticks([])
    for r, (Z, nm) in enumerate([(Zp, "PCA"), (Zt, "t-SNE")]):
        scat(ax[r, 0], Z, [yv == 0, yv == 1], ["NORMAL", "PNEUMONIA"], ["tab:blue", "tab:red"], f"{nm}: true class")
        scat(ax[r, 1], Z, [~wrong, wrong], ["correct", "incorrect"], ["lightgrey", "k"], f"{nm}: correct / incorrect (threshold {a.threshold})")
    plt.tight_layout(); plt.savefig(FIG / "fig08_embeddings.png", dpi=150); plt.close()

    outcome = np.where(~wrong, np.where(yv == 1, "correct PNEUMONIA", "correct NORMAL"),
                       np.where(yv == 1, "missed pneumonia (FN)", "false alarm (FP)"))
    T = vm.assign(prob=p0, outcome=outcome).groupby("outcome").agg(
        n=("prob", "size"), area_median=("area", "median"), aspect_median=("aspect", "median"),
        brightness_median=("mean_int", "median"), prob_median=("prob", "median"),
        share_uncertain=("prob", lambda s: ((s > 0.1) & (s < 0.9)).mean()))
    T.to_csv(OUT / "error_profile.csv"); print("\n== what the errors look like (seed 0, validation) ==\n", T.round(3))
    fp = np.where((yv == 0) & wrong)[0]; fn = np.where((yv == 1) & wrong)[0]
    fp, fn = fp[np.argsort(-p0[fp])][:6], fn[np.argsort(p0[fn])][:6]
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.4))
    for r, (idx, nm) in enumerate([(fp, "false alarm"), (fn, "missed pneumonia")]):
        for c in range(6):
            ax[r, c].axis("off")
            if c < len(idx):
                ax[r, c].imshow(d.X[int(va[idx[c]])].cpu().numpy(), cmap="gray")
                ax[r, c].set_title(f"{nm}, p={p0[idx[c]]:.2f}", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "fig09_errors.png", dpi=150); plt.close()
