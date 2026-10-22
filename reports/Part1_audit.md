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


## 6. Transfer baseline: [TODO; prediction goes in reports/predictions.md BEFORE results]
## 7. ArogyaNet: [TODO; hypotheses go in reports/hypotheses.md BEFORE ablations]
## 8. Two-branch experiment: [TODO]
## 9. Resolution and augmentation ablations: [TODO]
