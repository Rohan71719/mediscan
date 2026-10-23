## Task 1: Operating point at camp prevalence
*Script: `threshold.py`. Outputs: `outputs/threshold_sensitivity.csv`, `chosen_threshold_val_runs.csv`,
`camp_arithmetic.csv`, `operating_point.json`; figure `fig06_roc_pr_cost_chosen.png`.*

**Method.** Expected cost per 1,000 children at the camp (10% sick, 100 sick and 900 healthy) is
20 × missed pneumonia + 1 × false alarm. Recall and specificity do not depend on prevalence, so the
validation rates of the standard ArogyaNet (3 seeds pooled) were converted to camp costs without
retraining. The chosen threshold is the middle of the thresholds whose cost is within 2% of the
minimum.

| Threshold | Recall | Specificity | Cost per 1,000 |
|---|---|---|---|
| 0.5 (default) | 0.980 | 0.928 | 104.2 |
| 0.048 (cost ratio only, ignores prevalence) | 0.996 | 0.799 | 189.6 |
| 0.548 (theory, assumes calibrated probabilities) | 0.976 | 0.933 | 108.8 |
| **0.416 (chosen)** | **0.984** | **0.925** | **99.4** |

**Findings.** At the chosen threshold, validation recall is 0.984 ± 0.002 and specificity
0.925 ± 0.004 (3 seeds). At camp prevalence about 59% of flagged children are truly sick (PPV
0.594 ± 0.015): per 1,000 children, 98 sick children are found, 1.6 are missed and 67 healthy
children are flagged. The cost curve is flat between about 0.3 and 0.55 (99 to 109 per 1,000, seed
SD 7), so the choice of 0.416 over 0.5 is not statistically distinguishable. A threshold derived from the
cost ratio alone (1/21) ignores the prevalence shift and nearly doubles the cost, because specificity
falls to 0.80. The best threshold also depends strongly on the assumptions (0.17 to 0.89 for
prevalence 5 to 20% and cost ratio 10 to 40), so the cost ratio and prevalence need clinical
confirmation. Moving the threshold on one trained model changes recall (1.00 to 0.81) and specificity
(0.66 to 0.99) but not ROC-AUC (0.9928 throughout), so it changes the decision, not the model.
[Class-weight comparison: add after the rerun.]

**Why accuracy is the wrong headline.** Per 1,000 camp children, a tool that refers nobody scores
90% accuracy and misses all 100 sick children (cost 2,000). Two hypothetical models with the same 99%
accuracy cost 200 (all errors are misses) and 10 (all are false alarms), a 20-fold difference.
ArogyaNet has lower accuracy at the camp (93.1%) but a lower cost (99) than the first hypothetical
model. Accuracy also changes with prevalence: 96.8% on the validation split (73% sick) and 93.1% at
10% sick.

[Class-weight comparison: add after the rerun.]

**Class weights versus threshold moving.** A positive-class weight of 0.371 (equal total weight for the
two classes) changes the trained network: its scores and per-seed results differ from the standard
model (validation ROC-AUC 0.9925 ± 0.0009 vs 0.9919 ± 0.0009; recall at 0.5 0.979 vs 0.980;
specificity at 0.5 0.922 ± 0.018 vs 0.928 ± 0.005). The effect is small and within the seed
spread, and the weighted model is less stable (specificity SD 3.6 times larger). Placing the weighted
model's operating point against the standard model's ROC curve shows no consistent gain
(specificity at equal recall differs by +0.022, −0.009, −0.018 across seeds), so class weighting moved the operating
point rather than improving the ranking. A plausible reason is that the classes are nearly separable
(AUC 0.99), so reweighting affects only images near the boundary. Moving the threshold on one trained
model changes recall (1.00 to 0.81) and specificity (0.66 to 0.99) but not ROC-AUC (0.9928 throughout):
it changes only the decision. The training-side treatment needs a retraining for each setting and
gives one operating point, while threshold moving costs nothing and can be reset for any prevalence.
We therefore use the standard loss and a moved threshold.

**Caveats.** The threshold was chosen on the validation split, which was also used for early
stopping, so the cost is optimistic; the locked test set is used once, in task 2, with this
threshold frozen. Validation holds only 227 healthy images.


## Task 3: Hyperparameter tuning and regularisation
*Script: `tune.py`. Outputs: `outputs/tune_search.csv`, `tune_curves.csv`,
`tune_overfitting_summary.csv`, `tune_final_default_runs.csv`, `tune_final_tuned_runs.csv`;
figure `fig07_overfitting_curves.png`.*

**Search and budget.** Random search, 20 trials, over learning rate (3e-4 to 5e-3, log scale), batch
size (32, 64, 128), dropout (0 to 0.5) and weight decay (1e-6 to 1e-2, log scale). Each trial trained
ArogyaNet at 96×96 with seed 0 and augmentation, up to 20 epochs with early stopping on validation loss
(patience 6), and was ranked by validation ROC-AUC; the search took about 9 minutes on a T4. The best
trial (lr 0.0033, batch size 64, dropout 0.37, weight decay 2.7e-5, ROC-AUC 0.9932 at 96 px) was
retrained at 224×224 over three seeds and compared with the default (lr 2e-3, batch size 64, dropout 0.2,
weight decay 1e-4). Decision rule fixed beforehand: adopt the tuned configuration only if its mean
validation ROC-AUC exceeds the default's by more than 0.002.

| 224×224, 3 seeds, validation | Default | Tuned |
|---|---|---|
| ROC-AUC | 0.9919 ± 0.0009 | 0.9929 ± 0.0005 |
| PR-AUC | 0.9969 ± 0.0004 | 0.9973 ± 0.0002 |
| Balanced accuracy | 0.9542 ± 0.0042 | 0.9544 ± 0.0040 |
| Specificity | 0.9280 ± 0.0051 | 0.9280 ± 0.0092 |
| Recall | 0.9803 ± 0.0051 | 0.9808 ± 0.0016 |

**Decision.** The gain in ROC-AUC (+0.0010) is below the margin, so the default configuration is kept.
The search spread (0.983 to 0.993) is about three times the single-seed noise at 96 px (SD about
0.003), so the ranking of the best trials is not reliable and the best trial's score is probably
inflated by selection on the validation split. One pattern is clear: all eight trials with learning rate
≥ 1.5e-3 reached ROC-AUC ≥ 0.989, and the two worst had lr ≈ 3e-4 to 4e-4 (under-training within the
20-epoch budget); the default lr of 2e-3 lies in the good region. Dropout, weight decay and batch size
showed no visible effect. Limits: one seed per trial, search at 96 px with transfer to 224 px assumed,
and the validation split used for both the search and the early stopping.

**Overfitting curves (seed 0, 30 epochs, no early stopping; figure `fig07`).**

| Run | Epoch of lowest val loss | Val loss at best | Val loss at epoch 30 | Train loss at best → at 30 | Epochs saved by early stopping |
|---|---|---|---|---|---|
| Unregularised (no dropout, no weight decay, no augmentation) | 4 | 0.171 | 0.240 | 0.078 → 0.0005 | 26 |
| Regularised, default | 19 | 0.094 | 0.104 | 0.080 → 0.059 | 11 |
| Regularised, tuned | 26 | 0.096 | 0.102 | 0.049 → 0.044 | 4 |

Without regularisation the network overfits from epoch 4: training loss falls to 0.0005 while
validation loss rises 40% above its minimum; early stopping saves 26 of 30 epochs and 0.069 of
validation loss. With dropout, weight decay and augmentation the validation minimum moves to epoch 19
and the penalty for training on to epoch 30 is small (+0.010), so early stopping mainly saves compute.
One seed; validation loss fluctuates between epochs (Section 6), so the best epochs are approximate.
