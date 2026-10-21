# Part 1: Audit of the MediScan data

All numbers below come from scripts in this repo (script and output file named in each section).

## 1. Inventory and pixel statistics
*Script: `audit.py`. Outputs: `outputs/inventory.csv`, `counts.csv`, `pixel_stats_by_class.csv`; figures `fig01`, `fig02`.*

**Counts.** The primary dataset holds 5,856 images (train 5,216, test 624, val 16). Class balance is 1,583 NORMAL and 4,273 PNEUMONIA (73.0% pneumonia); the test set is 62.5% pneumonia, so the splits are not even drawn from the same class ratio.

**Greyscale.** All 5,856 images are truly greyscale: 5,573 are stored as single-channel and 283 are stored as RGB with identical channels.

**Size and shape.** [Insert fig01.] PNEUMONIA images are smaller and wider than NORMAL ones: mean area 1.05M vs 2.42M pixels (2.3x), mean height 820 vs 1,379, mean aspect ratio 1.51 vs 1.25.

**Brightness and contrast.** Mean brightness is the same (122.8 vs 122.6). PNEUMONIA images vary more in brightness across images (SD 19.9 vs 13.5) and have slightly lower contrast (55.4 vs 61.3).

**What this implies.** The two classes differ in acquisition geometry (image size and aspect), which has nothing to do with lungs. A network can therefore reach a good score by learning "small and wide means pneumonia". Such a model would fail at a camp whose equipment produces different image sizes. Tasks 4 and 8 test this directly.

## 2. Duplicates and leakage
*Script: `audit.py`. Outputs: `outputs/exact_duplicates.csv`, `near_duplicate_pairs.csv`; figure `fig03`.*

**Method.** Perceptual hash (pHash, 64 bits) on greyscale images; pairs with Hamming distance <= 5 are near-duplicates. Patient identity is parsed from the filename.

**Results.**
| Check | Count |
|---|---|
| Exact duplicate files (identical bytes) | 62 |
| Near-duplicate pairs, all | 265 |
| ...within the same split | 238 |
| ...between train and test | 26 |
| Near-duplicate pairs with conflicting labels | 40 [verified: TODO] |
| Patient IDs present in both train and test | [TODO after check] |

[Insert fig03: the five closest train/test pairs.]

**Interpretation.** [Write after the checks: say how many of the shared patients are real, what share of test images belong to patients seen in training, and why that makes the test score optimistic. Say plainly if some pHash matches are false positives, and give the distance below which the conflicts look like true duplicates.] The 98.4% reported by the previous engineer was measured on a test set that [is / may be] not independent of the training set, so it cannot be trusted as an estimate of performance on new children.

## 3. The validation set
*Script: `split.py`. Outputs: `outputs/val16_ci.csv`, `split.csv`, `split_summary.csv`.*

**Why 16 images is useless.** A model scoring 16/16 has an observed accuracy of 100%, but the 95% Clopper-Pearson interval is [0.794, 1.000]. The true accuracy could be as low as 79%, so the validation set cannot detect overfitting or compare models.

**Replacement design.** All images are pooled, exact duplicates removed, and images are grouped by patient and near-duplicate cluster so no group crosses a split. A stratified grouped split gives: locked test (about 15%, not touched until final evaluation), validation (about 15%) and training (about 70%). A 5-fold stratified grouped assignment on the non-test pool is also stored for Part 2.

**Why a locked test plus CV, not either alone.**
- *k-fold alone* gives no untouched final estimate (the folds are used for choices), and is expensive for deep networks.
- *A single split alone* gives a validation score that is noisy and tells you nothing about its own uncertainty.

[Insert the split table from `split_summary.csv`: counts and pneumonia fraction per split, and the group-size check.]

## 4. Shortcut hunt: [TODO after shortcut.py]
## 5. Baselines: [TODO after baselines.py]
## 6. Transfer baseline: [TODO; prediction goes in reports/predictions.md BEFORE results]
## 7. ArogyaNet: [TODO; hypotheses go in reports/hypotheses.md BEFORE ablations]
## 8. Two-branch experiment: [TODO]
## 9. Resolution and augmentation ablations: [TODO]
