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
