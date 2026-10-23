import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from src.utils import OUT, FIG, compute_metrics, get_logger

log = get_logger("threshold")
PI_CAMP, C_FN, C_FP = 0.10, 20.0, 1.0                  # camp prevalence 1:9, a miss costs 20 false alarms
N_TRAIN_POS, N_TRAIN = 3006, 4120                      # training prevalence 0.73 (from split.csv)
STD, WGT = "arogya_full_224_aug_saved_s{}", "arogya_full_224_aug_pw0.371_saved_s{}"
GRID = 1 / (1 + np.exp(-np.linspace(-9, 9, 901)))      # thresholds, dense near 0 and 1

def rates_grid(y, p, grid=GRID):
    pos, neg = (y == 1), (y == 0)
    pred = p[:, None] >= grid[None, :]
    return (pred & pos[:, None]).sum(0) / pos.sum(), (~pred & neg[:, None]).sum(0) / neg.sum()   # recall, specificity

def cost_per_1000(rec, spec, pi=PI_CAMP, cfn=C_FN, cfp=C_FP):
    return 1000 * (pi * (1 - rec) * cfn + (1 - pi) * (1 - spec) * cfp)

def pick(cost, tol=0.02):
    """Middle of the thresholds within 2% of the minimum cost (avoids chasing noise in a flat minimum)."""
    ok = np.where(cost <= cost.min() * (1 + tol) + 1e-9)[0]
    return GRID[ok[len(ok) // 2]]

def prec(rec, spec, pi):
    with np.errstate(invalid="ignore", divide="ignore"):
        return pi * rec / (pi * rec + (1 - pi) * (1 - spec))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--preds", default="outputs/val_preds.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.preds)
    need = [c.format(s) for c in (STD, WGT) for s in range(3)]
    miss = [c for c in need if c not in df.columns]
    assert not miss, f"columns missing in {a.preds}: {miss}\nfound: {list(df.columns)}"
    y = df.label.values.astype(int); pi_val = y.mean()

    # ================= PART A: class weights change the model, thresholds change only the decision ==========
    rows, match = [], []
    for s in range(3):
        for name, tag in [("standard", STD), ("class-weighted", WGT)]:
            p = df[tag.format(s)].values
            r, sp = rates_grid(y, p, np.array([0.5]))
            rows.append(dict(seed=s, model=name, roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                             mean_prob=p.mean(), recall_at_0p5=r[0], spec_at_0p5=sp[0]))
        fpr, tpr, _ = roc_curve(y, df[STD.format(s)].values)
        rw, spw = rates_grid(y, df[WGT.format(s)].values, np.array([0.5]))
        match.append(dict(seed=s, weighted_recall=rw[0], weighted_spec=spw[0],
                          standard_spec_at_same_recall=1 - np.interp(rw[0], tpr, fpr)))
    A, M = pd.DataFrame(rows), pd.DataFrame(match)
    A.to_csv(OUT / "classweight_vs_threshold_runs.csv", index=False); M.to_csv(OUT / "classweight_matched_recall.csv", index=False)
    print("\n== A1. standard vs class-weighted model (validation, mean ± SD over 3 seeds) ==")
    print(A.groupby("model")[["roc_auc", "pr_auc", "mean_prob", "recall_at_0p5", "spec_at_0p5"]].agg(["mean", "std"]).round(4).T)
    print("\n== A2. does the class-weighted operating point lie on the standard model's ROC curve? ==")
    print(M.round(4))

    p0 = df[STD.format(0)].values; sw = []
    for t in [0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]:
        r, sp = rates_grid(y, p0, np.array([t]))
        sw.append(dict(threshold=t, recall=r[0], specificity=sp[0], roc_auc=roc_auc_score(y, p0)))
    SW = pd.DataFrame(sw); SW.to_csv(OUT / "threshold_sweep_standard_seed0.csv", index=False)
    print("\n== A3. one trained model, moving the threshold (ROC-AUC never changes) ==\n", SW.round(4))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, name, c in [(STD, "standard", "tab:blue"), (WGT, "class-weighted", "tab:red")]:
        p = df[tag.format(0)].values; fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, c=c, label=f"{name} (AUC {roc_auc_score(y, p):.4f})")
        r, sp = rates_grid(y, p, np.array([0.5])); ax[0].scatter(1 - sp, r, c=c, s=60, marker="o", zorder=5)
    r, sp = rates_grid(y, p0, np.array([0.05, 0.1, 0.9, 0.95]))
    ax[0].scatter(1 - sp, r, c="tab:blue", s=40, marker="x", zorder=5, label="standard, thresholds 0.05/0.1/0.9/0.95")
    ax[0].set_xlim(0, 0.3); ax[0].set_ylim(0.8, 1.005); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("recall")
    ax[0].set_title("Seed 0. Dots: threshold 0.5. Crosses: moved thresholds"); ax[0].legend(fontsize=7)
    ax[1].hist(p0, bins=40, alpha=0.6, label="standard", color="tab:blue")
    ax[1].hist(df[WGT.format(0)].values, bins=40, alpha=0.6, label="class-weighted", color="tab:red")
    ax[1].set_yscale("log"); ax[1].set_xlabel("predicted probability of pneumonia"); ax[1].set_title("Class weights change the scores"); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIG / "fig05_classweight_vs_threshold.png", dpi=150); plt.close()

    # ================= PART B: cost matrix at camp prevalence, pooled standard model over 3 seeds =============
    P = np.concatenate([df[STD.format(s)].values for s in range(3)]); Y = np.tile(y, 3)
    rec, spec = rates_grid(Y, P); cost = cost_per_1000(rec, spec)
    t_star = float(pick(cost))
    odds_shift = (PI_CAMP / (1 - PI_CAMP)) / ((N_TRAIN_POS / N_TRAIN) / (1 - N_TRAIN_POS / N_TRAIN))
    o = (C_FP / C_FN) / odds_shift; t_bayes = o / (1 + o)       # theory, if probabilities were calibrated
    t_naive = C_FP / (C_FP + C_FN)                               # ignores the prevalence shift
    named = {"0.5 (default)": 0.5, f"cost-only {t_naive:.4f} (ignores prevalence)": t_naive,
             f"calibrated theory {t_bayes:.3f}": t_bayes, f"chosen {t_star:.4f} (min expected cost)": t_star}
    print(f"\n== B1. thresholds, expected cost per 1000 camp children (10% sick, miss = {C_FN:g} x false alarm) ==")
    for k, t in named.items():
        r, sp = rates_grid(Y, P, np.array([t])); print(f"{k:50s} recall {r[0]:.4f} spec {sp[0]:.4f} cost {cost_per_1000(r[0], sp[0]):.1f}")

    S = []
    for pi in [0.05, 0.10, 0.20]:
        for ratio in [10, 20, 40]:
            c = cost_per_1000(rec, spec, pi=pi, cfn=ratio); t = float(pick(c)); i = np.argmin(np.abs(GRID - t))
            S.append(dict(prevalence=pi, cost_ratio=ratio, threshold=round(t, 4), recall=rec[i], specificity=spec[i], cost_per_1000=c[i]))
    S = pd.DataFrame(S); S.to_csv(OUT / "threshold_sensitivity.csv", index=False)
    print("\n== B2. sensitivity of the chosen threshold to prevalence and cost ratio ==\n", S.round(4))

    per = []
    for s in range(3):
        p = df[STD.format(s)].values; m = compute_metrics(y, p, t_star)
        r, sp = rates_grid(y, p, np.array([t_star]))
        m.update(seed=s, camp_ppv=float(prec(r, sp, PI_CAMP)[0]), camp_cost_per_1000=float(cost_per_1000(r[0], sp[0])))
        per.append(m)
    PER = pd.DataFrame(per); PER.to_csv(OUT / "chosen_threshold_val_runs.csv", index=False)
    cols = ["accuracy", "balanced_acc", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "camp_ppv", "camp_cost_per_1000"]
    print(f"\n== B3. validation metrics at chosen threshold {t_star:.4f} (mean ± SD over 3 seeds) ==")
    print(pd.DataFrame({"mean": PER[cols].mean(), "sd": PER[cols].std()}).round(4))

    def row(name, r, sp):
        tp, fn, fp, tn = 100 * r, 100 * (1 - r), 900 * (1 - sp), 900 * sp
        return dict(strategy=name, TP=tp, FN=fn, FP=fp, TN=tn, accuracy=(tp + tn) / 1000,
                    PPV=tp / (tp + fp) if tp + fp > 0 else np.nan, cost=C_FN * fn + C_FP * fp)
    rs = lambda t: np.mean([rates_grid(y, df[STD.format(s)].values, np.array([t])) for s in range(3)], axis=0).ravel()
    r5, s5, rc, sc = *rs(0.5), *rs(t_star)
    T = pd.DataFrame([row("Everyone healthy (never refer)", 0, 1), row("Everyone pneumonia (refer all)", 1, 0),
                      row("Hypothetical 99% accuracy, all 10 errors are misses", 0.90, 1.0),
                      row("Hypothetical 99% accuracy, all 10 errors are false alarms", 1.0, 890 / 900),
                      row("ArogyaNet, threshold 0.5", r5, s5), row(f"ArogyaNet, chosen threshold {t_star:.3f}", rc, sc)])
    T.to_csv(OUT / "camp_arithmetic.csv", index=False)
    print("\n== B4. 1,000 camp children (100 sick, 900 healthy): accuracy vs expected cost ==\n", T.round(3).to_string(index=False))

    fpr, tpr, _ = roc_curve(Y, P); i = np.argmin(np.abs(GRID - t_star)); j = np.argmin(np.abs(GRID - 0.5))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(fpr, tpr, label=f"pooled val ROC (AUC {roc_auc_score(Y, P):.4f})")
    ax[0].scatter(1 - spec[i], rec[i], c="r", s=70, zorder=5, label=f"chosen t={t_star:.3f}")
    ax[0].scatter(1 - spec[j], rec[j], c="k", s=50, marker="s", zorder=5, label="t=0.5")
    ax[0].set_xlim(0, 0.3); ax[0].set_ylim(0.8, 1.005); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("recall"); ax[0].set_title("ROC"); ax[0].legend(fontsize=8)
    for pi, lab, c in [(pi_val, f"validation prevalence {pi_val:.2f}", "tab:blue"), (PI_CAMP, "camp prevalence 0.10", "tab:orange")]:
        pr = prec(rec, spec, pi); ax[1].plot(rec, pr, c=c, label=lab); ax[1].scatter(rec[i], pr[i], c="r", s=70, zorder=5)
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("PR (red dot = chosen threshold)"); ax[1].legend(fontsize=8)
    ax[2].plot(GRID, cost); ax[2].set_xscale("logit"); ax[2].axvline(t_star, c="r", label=f"chosen {t_star:.3f}")
    ax[2].axvline(0.5, c="k", ls="--", label="0.5"); ax[2].axvline(t_naive, c="g", ls=":", label=f"cost-only {t_naive:.3f}")
    ax[2].set_xlabel("threshold"); ax[2].set_ylabel("expected cost per 1,000 children"); ax[2].set_title("Cost at camp prevalence"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "fig06_roc_pr_cost_chosen.png", dpi=150); plt.close()

    json.dump(dict(model="ArogyaNet full 224 aug, standard loss", threshold=t_star, camp_prevalence=PI_CAMP,
                   cost_fn=C_FN, cost_fp=C_FP, chosen_on="validation split, pooled over seeds 0-2"),
              open(OUT / "operating_point.json", "w"), indent=2)
    log.info(f"chosen threshold {t_star:.4f}")
