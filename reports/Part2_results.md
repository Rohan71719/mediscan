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

## Task 2: Evaluation
*Scripts: `cv_eval.py` (cross-validation); the locked-test evaluation is added after the final
freeze. Outputs: `outputs/cv_folds.csv`, `cv_summary_thr0.4158.csv`, `cv_summary_thr0.5000.csv`,
`cv_oof_preds.csv`.*

**5-fold stratified, grouped cross-validation on the training pool** (train + validation, 4,972
images; folds from `split.csv`, grouped by patient and near-duplicate cluster). Each fold trains
ArogyaNet (default configuration) from scratch on the other four folds for 30 epochs and is scored
with its final-epoch weights, so nothing is selected on the held-out fold.

| Metric (mean ± SD over 5 folds) | Threshold 0.416 (chosen) | Threshold 0.5 |
|---|---|---|
| Accuracy | 0.9655 ± 0.0038 | 0.9660 ± 0.0037 |
| Balanced accuracy | 0.9545 ± 0.0078 | 0.9575 ± 0.0058 |
| Precision | 0.9748 ± 0.0071 | 0.9777 ± 0.0059 |
| Recall (sensitivity) | 0.9780 ± 0.0084 | 0.9755 ± 0.0084 |
| Specificity | 0.9311 ± 0.0219 | 0.9394 ± 0.0176 |
| F1 | 0.9764 ± 0.0032 | 0.9766 ± 0.0032 |
| ROC-AUC | 0.9934 ± 0.0021 | 0.9934 ± 0.0021 |
| PR-AUC | 0.9975 ± 0.0009 | 0.9975 ± 0.0009 |
| Camp precision (10% sick) | 0.619 ± 0.071 | 0.647 ± 0.064 |
| Camp cost per 1,000 children | 106.0 ± 10.9 | 103.4 ± 9.5 |

Pooled over all out-of-fold images, threshold 0.416: TN 1,249, FP 92, FN 79, TP 3,552 (threshold
0.5: TN 1,260, FP 81, FN 88, TP 3,543).

**Reading.** Cross-validation agrees with the single validation split (balanced accuracy 0.955 vs
0.954, ROC-AUC 0.993 vs 0.992). Specificity ranges from 0.898 to 0.955 across folds, so one
validation split with 227 healthy images is more uncertain than its seed spread suggests. Expected
camp cost is 106 ± 11 per 1,000 children, slightly above the validation estimate (99). The chosen
threshold is not distinguishable from 0.5 (cost 106.0 vs 103.4), as expected from the flat cost
curve in Task 1; we keep it because it was fixed before cross-validation. Because the threshold was
selected on the validation split, which is part of the pool, the figures at 0.416 are slightly
optimistic.

**Why accuracy alone is the wrong headline.** At the camp's 10% prevalence a tool that refers
nobody is 90% accurate while missing every sick child, so accuracy rewards ignoring the class we
care about. A miss costs about 20 false alarms, so two models with the same accuracy can differ
twenty-fold in expected cost; we therefore report recall, specificity and expected cost at a stated
threshold.

**Locked test set:** [added after the final freeze].


## Task 4: Ensembles
*Scripts: `ensemble.py`, `features.py`. Outputs: `outputs/ensemble_table.csv`, `embed_models_runs.csv`.*

**Method.** Soft voting averages the predicted probabilities of the members. Members are the Part 1
models (ArogyaNet full, without residual connections, without SE, two-branch, and fine-tuned ResNet18),
combined seed by seed (seed s of every model), so each ensemble row is a mean ± SD over three seeds. Random
Forest (300 trees) and AdaBoost (200 stumps) were trained on the 256-dimensional penultimate-layer
embeddings of ArogyaNet (training split only). Everything is evaluated on the validation split at
threshold 0.5.


| Model / Variant | Balanced Acc. | Specificity | Recall | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ArogyaNet full** | 0.9542 ± 0.0042 | 0.9280 ± 0.0051 | 0.9803 ± 0.0051 | 0.9919 ± 0.0009 | 0.9969 ± 0.0004 |
| **ArogyaNet no residual** | 0.9544 ± 0.0073 | 0.9280 ± 0.0067 | 0.9808 ± 0.0085 | 0.9920 ± 0.0006 | 0.9969 ± 0.0003 |
| **ArogyaNet no SE** | 0.9479 ± 0.0048 | 0.9192 ± 0.0092 | 0.9765 ± 0.0024 | 0.9914 ± 0.0010 | 0.9967 ± 0.0004 |
| **ArogyaNet two-branch** | 0.9532 ± 0.0070 | 0.9266 ± 0.0102 | 0.9797 ± 0.0040 | 0.9926 ± 0.0006 | 0.9973 ± 0.0002 |
| **ResNet18 (ImageNet)** | 0.9630 ± 0.0028 | 0.9398 ± 0.0111 | 0.9861 ± 0.0067 | 0.9958 ± 0.0006 | 0.9984 ± 0.0003 |
| **Random Forest on embeddings** | 0.9530 ± 0.0035 | 0.9236 ± 0.0051 | 0.9824 ± 0.0028 | 0.9840 ± 0.0026 | 0.9903 ± 0.0018 |
| **AdaBoost on embeddings** | 0.9465 ± 0.0036 | 0.9148 ± 0.0025 | 0.9781 ± 0.0046 | 0.9854 ± 0.0030 | 0.9919 ± 0.0037 |
| **Soft vote, 4 ArogyaNet variants** | 0.9574 ± 0.0027 | 0.9339 ± 0.0044 | 0.9808 ± 0.0016 | 0.9934 ± 0.0006 | 0.9975 ± 0.0003 |
| **Soft vote, 4 ArogyaNet + ResNet18** | 0.9602 ± 0.0034 | 0.9354 ± 0.0067 | 0.9851 ± 0.0024 | 0.9949 ± 0.0009 | 0.9981 ± 0.0003 |
| **Soft vote, 3 seeds of ArogyaNet (1 run)** | 0.9544 | 0.9295 | 0.9792 | 0.9932 | 0.9975 |
| **Soft vote, 3 seeds of ResNet18 (1 run)** | 0.9664 | 0.9471 | 0.9856 | 0.9967 | 0.9987 |

> **Note:** Values represent Validation metrics at a threshold of 0.5, reported as mean ± SD over 3 seeds (unless specified as a single 1-run ensemble).


**Does the ensemble beat its best member?** The ArogyaNet-only vote improves on its best member by 0.003
balanced accuracy (0.9574 vs 0.9544) and 0.0008 ROC-AUC, within the seed spread. The vote that adds
ResNet18 does not beat ResNet18 alone (0.9602 vs 0.9630 balanced accuracy; ROC-AUC 0.9949 vs 0.9958).
Reasons: (1) the members are unequal, since ResNet18 made 20 errors on seed 0 against 31 to 34 for the
others, so an unweighted average pulls it toward weaker models; (2) the errors are largely shared: the mean
pairwise error overlap (Jaccard) is 0.56, probabilities correlate at 0.83, and 15 images are wrong in
every member, of 51 that are wrong in at least one, so voting can only change the 36 images on which the
members disagree, and the 15 shared errors (1.8% of the validation set) are a floor that no average can
remove. Averaging three seeds of the same model gains little for the same reason (ResNet18: +0.003 balanced
accuracy; ArogyaNet: +0.0002).

**Random Forest and AdaBoost on embeddings.** They do not beat the network's own classifier: balanced
accuracy 0.953 (Random Forest) and 0.947 (AdaBoost) against 0.954, and ROC-AUC 0.984 and 0.985 against
0.992. The embedding was trained jointly with the final linear layer, so the classes are already nearly
linearly separable and trees add nothing; their probabilities are also coarser, which costs ranking
resolution. [Trees were fitted on training embeddings, which are cleaner than validation embeddings
because the network has partly memorised the training set; this is a plausible explanation that we did
not test.]

**Decision.** We keep a single ArogyaNet (seed 0). A vote of three ArogyaNet seeds or four variants improves
validation ROC-AUC by 0.0013 to 0.0015, below the margin of 0.002 fixed in advance, and costs three to four
times the inference time. Limits: unweighted votes, validation split only, and "best member" is itself
chosen on the validation split.
