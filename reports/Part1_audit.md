# Part 1: Audit of the MediScan data

All numbers below come from scripts in this repo (script and output file named in each section).

## 1. Inventory and pixel statistics
*Script: `audit.py`. Outputs: `outputs/inventory.csv`, `counts.csv`, `pixel_stats_by_class.csv`; figures `fig01`, `fig02`.*

**Counts.** The primary dataset holds 5,856 images (train 5,216, test 624, val 16). Class balance is 1,583 NORMAL and 4,273 PNEUMONIA (73.0% pneumonia); the test set is 62.5% pneumonia, so the splits are not even drawn from the same class ratio.

**Greyscale.** All 5,856 images are truly greyscale: 5,573 are stored as single-channel and 283 are stored as RGB with identical channels.

**Size and shape.** [Insert fig01.] PNEUMONIA images are smaller and wider than NORMAL ones: mean area 1.05M vs 2.42M pixels (2.3x), mean height 820 vs 1,379, mean aspect ratio 1.51 vs 1.25.

**Brightness and contrast.** Mean brightness is the same (122.8 vs 122.6). PNEUMONIA images vary more in brightness across images (SD 19.9 vs 13.5) and have slightly lower contrast (55.4 vs 61.3).

**What this implies.** The two classes differ in acquisition geometry (image size and aspect), which has nothing to do with lungs. A network can therefore reach a good score by learning "small and wide means pneumonia". Such a model would fail at a camp whose equipment produces different image sizes. Tasks 4 and 8 test this directly.
<p align="center">
  <img src="../figures/fig01_pixel_stats_by_class.png" alt="Pixel stats by class" width="70%">
  <br>
  <em>Figure 1: Pixel stats by class.</em>
</p>
<p align="center">
  <img src="../figures/fig02_counts.png" alt="counts" width="50%">
  <br>
  <em>Figure 2: Counts.</em>
</p>


## 2. Duplicates and leakage
*Scripts: `audit.py`, `leakage_check.py`. Outputs: `outputs/exact_duplicates.csv`,
`near_duplicate_pairs.csv`, `leakage_check.csv`; figure `fig03`.*

**Method.** Perceptual hash (pHash, 64 bits) on greyscale images; pairs with Hamming distance
<= 5 are near-duplicates. Patient identity was parsed from filenames. A first parse (person
number only) merged different children, because `personN_bacteria_*` and `personN_virus_*`
are numbered separately, and the NORMAL series `IM-XXXX` (train) and `NORMAL2-IM-XXXX` (test)
reuse the same numbers. It reported 264 shared IDs, which inspection showed to be false. The
strict ID used for the leakage check keeps the bacteria/virus tag and the NORMAL2 prefix.

**Results.**
| Check | Count |
|---|---|
| Files in exact-duplicate sets (identical bytes) | 62 in 30 sets (32 redundant copies) |
| ...sets spanning more than one split | 0 |
| Near-duplicate pairs (distance <= 5), all | 265 |
| ...within the same split | 238 |
| ...between train and test | 26 (1 at distance 2, 25 at distance 4) |
| ...between val and another split | 1 |
| Near-duplicate pairs with conflicting labels | 40 (all at distance 4; none at distance <= 2) |
| Pair distances (all 265) | 0: 34, 2: 16, 4: 215 |
| Strict patient IDs present in both train and test | 0 (0 of 624 test images) |


<p align="center">
  <img src="../figures/fig03_worst5_pairs.png" alt="Five closest train/test pairs" width="40%">
  <br>
  <em>Figure 3: The five closest train/test pairs.</em>
</p>


**Interpretation.** We found no evidence that the previous engineer's test set leaked from the
training set: no exact duplicate and no strict patient ID crosses train and test, and only 26
of 265 near-duplicate pairs do, almost all at distance 4, where look-alike chest X-rays are
common. All 40 label-conflicting pairs are at distance 4 [and the closest four are visibly
different X-rays], so we treat them as hash false positives rather than label noise. Patient
identity is inferred from filenames only, so hidden overlap cannot be excluded. The 98.4% is
therefore not explained by simple leakage, and the remaining explanations (a class mix in
the test set that differs from training, a single hospital and age group, and the size and shape
differences between the classes in Section 1) are examined in Sections 4 and 8.


## 3. The validation set
*Script: `split.py`. Outputs: `outputs/val16_ci.csv`, `split.csv`, `split_summary.csv`, `split.md5`.*

**Why 16 images is useless.** A model scoring 16/16 has an observed accuracy of 100%, but the
95% Clopper-Pearson interval is [0.794, 1.000]. The true accuracy could be as low as 79%, so a
validation set of this size cannot detect overfitting or tell two models apart.

**Replacement design.** The three original folders were pooled (5,856 images). The 32 exact
duplicate files were removed, leaving 5,824 images. Images were then grouped so that no group
crosses a split: images with the same patient ID, and images whose perceptual hashes differ by
at most 2 bits and share a label, are placed in the same group (2,773 groups). A stratified
grouped split then produced a locked test set (about 15%, not touched until the final
evaluation), a validation set (about 15%) and a training set (about 70%). A 5-fold stratified
grouped assignment on the non-test pool (train + val) is also stored for Part 2. The script
asserts that no group or patient appears in more than one of train, val and test.

**A grouping problem we found and fixed.** A first attempt linked images with a hash distance
of up to 5 (the audit threshold). This chained 188 images from 105 different patients (154
NORMAL, 34 PNEUMONIA) into a single group, because chest X-rays resemble each other and
similarity links form chains (A~B, B~C...). A group cannot be split, so this made the class
balance uneven (test set 0.81 pneumonia). We tightened the grouping threshold to 2 bits and
linked only images with the same label. The largest group is now 31 images and the class
balance is even across train, validation and test. No model performance was used in any of
these choices.Patient IDs used for grouping were parsed from filenames without the bacteria/virus tag or the NORMAL2 prefix, so some different children share an ID. This makes groups larger than necessary, which can only reduce leakage across splits.

| Split | NORMAL | PNEUMONIA | Total | Pneumonia fraction |
|---|---|---|---|---|
| Train | 1,114 | 3,006 | 4,120 | 0.730 |
| Validation | 227 | 625 | 852 | 0.734 |
| Test (locked) | 238 | 614 | 852 | 0.721 |

Split table: `outputs/split_summary.csv`. Frozen split file hash (MD5):
`d7a8298d0b456ed277a60be0d6ca3934`. Split seed: 42 (test), 43 (validation), 44 (CV folds),
set in `split.py`.

| Fold (train + val pool) | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| NORMAL | 253 | 260 | 288 | 275 | 265 |
| PNEUMONIA | 768 | 717 | 675 | 739 | 732 |

**Why a locked test plus cross-validation, and not either alone.**
- *k-fold alone* leaves no untouched final estimate, because every image is used in some
  fold while choices (thresholds, hyperparameters) are being made. It is also five times as
  expensive for deep networks.
- *A single split alone* gives one validation score with no estimate of its own spread, so
  it cannot say whether a difference between two models is real.
- The locked test set gives one honest final number; the 5 folds give a mean ± SD for
  Part 2 and show how stable the result is.

  **Preprocessing and caching.** *Script: `cache_data.py`. Output: `outputs/cache/` (`X96.npy`,
`X224.npy`, `y.npy`, `meta.csv`).* All 5,824 pooled images were converted to greyscale and
resized to 96×96 and 224×224 pixels (JPEG decoded at reduced size with PIL, then bilinear
resize; aspect ratio is not preserved) and stored as uint8 arrays in the row order of
`split.csv`. The original height, width, aspect ratio, area and mean intensity are kept in
`meta.csv`, because resizing to a square removes the size and shape differences found in
Section 1; the two-branch model (task 8) and the shortcut test (task 4) use them. All
experiments read these arrays, fit on the train split and are evaluated on the validation
split; the test split is not used until the final evaluation.

## 4. Shortcut hunt: 
*Script: `shortcut.py`. Output: `outputs/shortcut_results.csv`.*

**Method.** Three deliberately crippled models were trained on the training split (4,120 images)
and evaluated on the validation split (852 images: 625 PNEUMONIA, 227 NORMAL). Each is a
standardised logistic regression with threshold 0.5:
1. the image downsampled to 16×16 pixels (256 features);
2. the image with the central region blanked out (pixels 16 to 80 of a 96×96 image set to zero),
   leaving only the border, using a 32×32 block-averaged version;
3. three handcrafted numbers only (aspect ratio, image area, mean intensity), with no pixels.

**Results (validation set, seed 0).**

| Model | Accuracy | Balanced acc. | Recall | Specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Majority class (always PNEUMONIA) | 0.734 | 0.500 | 1.000 | 0.000 | 0.500 | 0.734 |
| 16×16 downsample | 0.950 | 0.925 | 0.978 | 0.872 | 0.980 | 0.991 |
| Lungs masked (border only) | 0.903 | 0.859 | 0.952 | 0.767 | 0.961 | 0.985 |
| Metadata only (aspect, area, mean) | 0.832 | 0.778 | 0.894 | 0.661 | 0.905 | 0.961 |

**What it means.** All three crippled models beat chance by a wide margin (balanced accuracy 0.78
to 0.93 against 0.50). Two results are direct evidence of a shortcut:
- *Metadata only.* Three numbers that contain no lung appearance at all give ROC-AUC 0.905. Image
  size and shape differ by class (Section 1: area 2.3× larger for NORMAL, aspect ratio 1.25 vs
  1.51), so a classifier can separate the classes without looking at the lungs.
- *Lungs masked.* With the central lung region removed, balanced accuracy is still 0.859, so
  information outside the lungs (borders, shoulders, positioning, markers) separates the classes.

The 16×16 result (balanced accuracy 0.925) shows how little detail is needed. We do not count it
as proof of a shortcut by itself, because coarse opacity is a genuine pneumonia sign that
survives heavy downsampling.

**Consequence.** The dataset contains information that is not lung appearance, and a full network
can use it as well. High accuracy on this single-hospital dataset therefore does not show that a
model has learned pneumonia, and a camp whose equipment produces different image sizes or framing
could break it. Section 8 tests whether giving the network the metadata helps, and Part 2 tests
a different hospital.

**Limits.** One seed and one split. With 227 NORMAL validation images, a specificity estimate has
a 95% interval of roughly ±4 to 5 percentage points, so small differences between the models
above (for example between 0.872 and 0.767) should be read with that in mind.


## 5. Baselines: 
*Script: `baselines.py`. Output: `outputs/baselines.csv`.*

**Method.** (a) Majority class: predict PNEUMONIA for every image. (b) HOG features (96×96
greyscale input, 9 orientations, 8×8-pixel cells, 2×2 blocks; 4,356 features), standardised, then
logistic regression, trained on the training split and evaluated on the validation split at
threshold 0.5.

**Results (validation set, seed 0).**

| Model | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | PR-AUC |
|---|---|---|---|---|---|---|---|
| Majority class | 0.734 | 0.500 | 0.734 | 1.000 | 0.000 | 0.846 | 0.734 |
| HOG + logistic regression | 0.942 | 0.917 | 0.951 | 0.971 | 0.863* | 0.961 | 0.994 |

*Specificity for HOG is computed as 2 × balanced accuracy − recall; the value in
`outputs/baselines.csv` is the reference.

**What it means.** A classical pipeline with no deep learning reaches 94.2% accuracy and a PR-AUC
of 0.994. This is the bar a deep model has to clear. It also puts the previous engineer's 98.4%
in context: it is only about four points above a handcrafted-feature baseline, on a single
hospital's data where Section 4 shows non-lung cues are enough to score well.

**How to read these numbers.** The validation set is 73% pneumonia, so accuracy, precision, F1 and
PR-AUC are flattering; the majority-class row alone scores 0.734 accuracy and F1 0.846.
Balanced accuracy and specificity are the honest columns: the HOG model misses about 3% of sick
children but still wrongly flags about 14% of healthy ones. Results come from one seed and one
split.


## 6. Transfer baseline:
*Script: `transfer.py` (uses `src/data.py`, `src/train.py`). Outputs: `outputs/transfer_runs.csv`,
`transfer_summary.csv`, `transfer_history.csv`. Prediction written before the run:
`reports/predictions.md`.*

**Setup.** ResNet18 with ImageNet weights, fully fine-tuned; the greyscale image is replicated to
three channels and normalised with ImageNet statistics; 224×224 input; AdamW (lr 3e-4, weight decay
1e-4), cosine schedule, batch size 64, up to 12 epochs, no augmentation, early stopping on
validation loss (patience 4). Seeds 0, 1, 2. Trained on the training split, evaluated on the
validation split at threshold 0.5.

**Prediction (committed before running).** Early and middle layers (edges, textures) should
transfer; colour sensitivity and object-level late layers should not. Expected ROC-AUC 0.98 to 0.99,
balanced accuracy 0.92 to 0.95, seed SD about 1 point.

**Results (validation set, mean ± SD over 3 seeds).**

| Metric | ResNet18 (ImageNet) | HOG + LR (Section 5) |
|---|---|---|
| Accuracy | 0.974 ± 0.002 | 0.942 |
| Balanced accuracy | 0.963 ± 0.003 | 0.917 |
| Precision | 0.978 ± 0.004 | 0.951 |
| Recall | 0.986 ± 0.007 | 0.971 |
| Specificity | 0.940 ± 0.011 | 0.863 |
| F1 | 0.982 ± 0.002 | 0.961 |
| ROC-AUC | 0.9958 ± 0.0006 | n/a |
| PR-AUC | 0.9984 ± 0.0003 | 0.994 |

Per seed (false positives / false negatives out of 227 NORMAL / 625 PNEUMONIA): seed 0: 16 / 4,
seed 1: 14 / 10, seed 2: 11 / 12. Best epoch (0-indexed): 4, 4, 3.

**Did the numbers agree with the prediction?** Partly. The direction agreed: the pretrained network
beats the HOG baseline, and generic early-layer features evidently transfer to radiographs despite
the greyscale input. The size of the gain did not: balanced accuracy (0.963) and ROC-AUC (0.996)
were above the predicted ranges, and the seed spread (0.003) was smaller than the predicted 0.01.
The prediction was too cautious. The largest gain over HOG is in specificity (0.86 to 0.94), i.e.
fewer healthy children flagged.

**Observations.** Validation AUC is already 0.993 to 0.996 after the first epoch, so pretrained
features separate the classes almost immediately. Training loss falls to about 0.0005 while
validation loss stops improving after epoch 3 to 4, so the network overfits quickly. Validation
loss is also erratic from epoch to epoch while AUC is stable, which means the ranking is good but
the predicted probabilities are poorly calibrated; this matters for the threshold choice in Part 2.

**Caveats.** The validation split was used for early stopping, so these numbers are slightly
optimistic; the test split has not been used. Seeds vary initialisation and shuffling only, on one
fixed split, so the SD understates the true uncertainty (with 227 NORMAL validation images, a
specificity estimate has a 95% interval of about ±3 points). Section 4 shows that non-lung cues
alone give high scores on this dataset, so a high score here does not prove that the network has
learned pneumonia.


## 7. ArogyaNet
*Scripts: `src/models.py`, `arch_table.py`, `run_variants.py`. Outputs: `outputs/arch_table.txt`,
`arogya_*_runs.csv`, `arogya_*_history.csv`. Hypotheses committed before any result:
`reports/hypotheses.md`.*

**Design.** A 300,641-parameter network trained from scratch (no pretrained weights) on 224×224
greyscale input. Design decisions: (1) residual connections around each block, (2) squeeze-and-
excitation (SE) channel attention in each block, (3) depthwise-separable convolutions, (4) global
average pooling instead of flatten + dense. Decisions 1 and 2 are ablated.

| Stage | Output shape | Parameters |
|---|---|---|
| Stem: conv 3×3, stride 2, BN, ReLU | 24×112×112 | 264 |
| Stage 1: 2 × DS-block, 32 ch | 32×56×56 | 4,472 |
| Stage 2: 2 × DS-block, 64 ch | 64×28×28 | 13,888 |
| Stage 3: 3 × DS-block, 128 ch | 128×14×14 | 78,752 |
| Stage 4: 2 × DS-block, 256 ch | 256×7×7 | 203,008 |
| Global average pool, dropout 0.2, linear | 1 | 257 |
| **Total** | | **300,641** (1.2 MB, float32) |

DS-block = depthwise 3×3 → pointwise 1×1 → SE → (+ skip) → ReLU. The per-layer table is in
`outputs/arch_table.txt`. 

**Hypotheses (written before the results, `reports/hypotheses.md`).** Residual connections:
removing them lowers balanced accuracy by about 1 point and makes training less stable. SE blocks:
removing them has a smaller effect (under 1 point), possibly inside the seed noise.

**Training.** AdamW, lr 2e-3, weight decay 1e-4, cosine schedule, batch size 64, up to 30 epochs,
early stopping on validation loss (patience 8), mild augmentation (rotation ±10°, zoom ±10%,
brightness and contrast ±10%; no flips). Seeds 0, 1, 2. Same settings for every variant. Not tuned.

**Results (validation set, mean ± SD over 3 seeds).**

| Variant | Params | Balanced acc. | Specificity | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Full ArogyaNet | 300,641 | 0.954 ± 0.004 | 0.928 ± 0.005 | 0.980 ± 0.005 | 0.9919 ± 0.0009 | 0.9969 ± 0.0004 |
| No residual connections | 255,905 | 0.954 ± 0.007 | 0.928 ± 0.007 | 0.981 ± 0.009 | 0.9920 ± 0.0006 | 0.9969 ± 0.0003 |
| No SE blocks | 204,049 | 0.948 ± 0.005 | 0.919 ± 0.009 | 0.977 ± 0.002 | 0.9914 ± 0.0010 | 0.9967 ± 0.0004 |
| ResNet18, ImageNet (Section 6) | ~11.2 M | 0.963 ± 0.003 | 0.940 ± 0.011 | 0.986 ± 0.007 | 0.9958 ± 0.0006 | 0.9984 ± 0.0003 |

**Ablation 1: residual connections. Hypothesis not supported.** Removing them changes balanced
accuracy by +0.0002 (full 0.9542, without 0.9544), far inside the seed spread (SD 0.004 to 0.007),
and ROC-AUC is identical. In a network this shallow (9 blocks) with batch normalisation,
gradients evidently reach every layer without skips, so the connections add nothing measurable.

**Ablation 2: SE blocks. Inconclusive.** Removing them lowers balanced accuracy by 0.006
(0.9542 to 0.9479), in the predicted direction and with a size under the 1 point we predicted, but
comparable to the spread across seeds, and in one of three seeds the no-SE model scored higher. With
three seeds this cannot be separated from noise. Both ablations also change the parameter count
(−44,736 and −96,592), so they are not pure tests of the component.

**Comparison.** ArogyaNet is about 0.9 points below fine-tuned ResNet18 in balanced accuracy
(0.954 vs 0.963) with 37 times fewer parameters, and clearly above the HOG baseline (0.917). Its
1.2 MB size fits the 25 MB update channel; ResNet18's weights (about 45 MB) do not. In three runs
the best epoch was the last of 30, so the networks were still improving; the schedule is not tuned
(Part 2). As in Section 6, the validation split was used for early stopping, so numbers are slightly
optimistic, and Section 4 shows non-lung cues alone give high scores on this dataset.



## 8. The two-branch experiment
*Script: `run_variants.py --variants meta`. Outputs: `outputs/arogya_meta_224_aug_runs.csv`,
`arogya_meta_224_aug_history.csv`.*

**Design.** A second version of ArogyaNet takes the image through the convolutional branch and
three handcrafted numbers through a small dense branch (3 → 16 → 16, ReLU): the log of the original
image area, the aspect ratio and the mean intensity, standardised with training-set statistics only.
The 16-dimensional output is concatenated with the 256-dimensional pooled image features before the
classifier (+352 parameters, 300,993 in total). Training settings are identical to Section 7.
Because images are resized to a square, the image-only network does not see the original size or
shape directly; this branch gives them back.

**Results (validation set, mean ± SD over 3 seeds).**

| Model | Balanced acc. | Specificity | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Image-only ArogyaNet | 0.954 ± 0.004 | 0.928 ± 0.005 | 0.980 ± 0.005 | 0.9919 ± 0.0009 | 0.9969 ± 0.0004 |
| Two-branch | 0.953 ± 0.007 | 0.927 ± 0.010 | 0.980 ± 0.004 | 0.9926 ± 0.0006 | 0.9973 ± 0.0002 |
| Metadata only, logistic regression (Section 4) | 0.778 | 0.661 | 0.894 | 0.905 | 0.961 |

**Did the metadata branch help?** No. Balanced accuracy differs by −0.001 and ROC-AUC by +0.0007;
both are within the seed-to-seed spread, and the sign of the difference changes between seeds.

**Connection to Section 4.** The three numbers alone separate the classes well above chance (balanced
accuracy 0.778), so they carry information that has nothing to do with lungs. Yet adding them to the
CNN gains nothing, so either the image already holds that information (framing, borders, texture) or
it adds little next to a model that scores 0.99 AUC. This does not show that the network ignores
shortcuts: the lung-masked model in Section 4 reaches balanced accuracy 0.859 from non-lung regions,
and a network can use such cues without a metadata branch. Had the branch helped, it would have
meant the model exploits properties of this hospital's archive (image size, framing, brightness),
which describe the scanner and file export, not the child, and would not carry over to a camp with
different equipment; area and aspect ratio would also depend on how the camp's software saves
images. Whether the model has learned pneumonia rather than hospital artefacts is tested on the
external hospital in Part 2. **We use the image-only model**: the branch gives no benefit and adds
a dependency on file properties.

**Caveats.** Three seeds on one split; early stopping used the validation split.


## 9. Two ablations: input resolution and augmentation
*Script: `run_variants.py --variants full --size 96` and `--no_aug`. Outputs:
`outputs/arogya_full_96_aug_*.csv`, `arogya_full_224_noaug_*.csv`. The 224 + augmentation
reference is the full model of Section 7.*

**Setup.** Full ArogyaNet, three seeds, all other settings identical to Section 7. Resolution: 96×96
vs 224×224 (both with augmentation). Augmentation: none vs a policy of rotation ±10°, zoom ±10%,
brightness and contrast jitter ±10% (both at 224×224). No other ablations were run.

**Results (validation set, mean ± SD over 3 seeds).**

| Setting | Balanced acc. | Specificity | Recall | ROC-AUC | PR-AUC | Best epochs |
|---|---|---|---|---|---|---|
| 224, augmentation (reference) | 0.954 ± 0.004 | 0.928 ± 0.005 | 0.980 ± 0.005 | 0.9919 ± 0.0009 | 0.9969 ± 0.0004 | 18, 24, 29 |
| 96, augmentation | 0.945 ± 0.004 | 0.918 ± 0.007 | 0.972 ± 0.005 | 0.9900 ± 0.0030 | 0.9962 ± 0.0013 | 6, 8, 27 |
| 224, no augmentation | 0.926 ± 0.019 | 0.897 ± 0.053 | 0.954 ± 0.015 | 0.9814 ± 0.0036 | 0.9924 ± 0.0019 | 1, 3, 1 |

**Resolution.** Reducing the input from 224 to 96 lowers balanced accuracy by 0.009 (lower in all
three seeds) and ROC-AUC by 0.002 (mixed sign across seeds, within the spread). Ranking quality is
therefore similar, and the difference appears mainly at the fixed 0.5 threshold. Pneumonia is
visible as coarse opacity, which survives heavy downsampling (consistent with the 16×16 result in
Section 4). The cost of the 96×96 input is about 0.18 times that of 224×224 in pixels, so it is a
fallback if inference is too slow on the camp laptop (latency is measured in Part 2).

**Augmentation.** Removing augmentation lowers balanced accuracy by 0.029 and ROC-AUC by 0.011; all
three no-augmentation runs are below all three augmented runs on both metrics. Training is also less
stable (balanced-accuracy SD 0.019 vs 0.004). The best epoch is 1 to 3 in every no-augmentation run,
i.e. the network memorises the training set within a few epochs. Augmentation is the more
consequential of the two choices. The policy is a bundle, so we cannot attribute the gain to rotation,
zoom or brightness and contrast individually.

**The medically wrong augmentation: horizontal flip.** Flipping a chest X-ray left to right is a
routine augmentation for natural images, but it is wrong here (we did not use it). The heart sits on
the left of the body; a mirrored film shows it on the right, which is the rare condition
dextrocardia, an anatomy that does not occur in the children this tool will see and that the label
was never meant to describe. Laterality also carries diagnostic meaning: which lung is affected, a
left lower-lobe consolidation hidden behind the heart, the stomach bubble under the left
hemidiaphragm, and the L/R marker, which is mirrored into unreadable text. After a flip the image no
longer matches the clinical reading that produced its label. (For a bare pneumonia-present label, a
flip does not change whether pneumonia is present, so the harm is that it teaches the network
anatomy and laterality cues that are false; for any side- or lobe-specific label it would corrupt the
label itself.) A vertical flip is worse: it puts the diaphragm and stomach bubble at the top, an
anatomically impossible image. Our policy therefore uses only small rotations, zoom and brightness
and contrast jitter.

<p align="center">
  <img src="../figures/fig04_arogyanet.png" alt="Five closest train/test pairs" width="80%">
  <br>
  <em>Figure 4: ArogyaNet.</em>
</p>

**Choice for the final model.** 224×224 with augmentation, because it scores best on validation
(a selection made without touching the test split). The validation split was used for early stopping
and for this choice, so the numbers are slightly optimistic. Section 4 still applies: a good
validation score does not show that the network has learned pneumonia rather than hospital cues.
