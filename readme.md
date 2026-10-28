# MediScan Audit: reproduction guide

**Contents.** `reports/Part1_audit.md` (Part 1 sections 1 to 9), `reports/Part2_results.md` (Part 2), `reports/CEO_report.md`,
the technical report, all scripts, `model/` (shipped model: 1.5 MB), `outputs/` (tables), `figures/`.
`reports/predictions.md` and `reports/hypotheses.md` were committed before the experiments they describe.

## Environment
Python 3.13 (3.10+ should work), PyTorch with a CUDA GPU for training (free Kaggle T4 is enough).
`pip install -r requirements.txt` (versions pinned from the Kaggle environment the results were produced in).

## Data (not in the repo)
- Primary: Kaggle "Chest X-Ray Images (Pneumonia)".
- External: Kaggle "COVID-19 Radiography Database" (Normal and Viral Pneumonia only; used once).
The scripts search `/kaggle/input` recursively and write to `/kaggle/working`. On a fresh Linux machine:
`sudo mkdir -p /kaggle/input /kaggle/working && sudo chown -R $USER /kaggle`, then unzip each dataset into its own folder under `/kaggle/input/`.

## Reproduce everything
`bash run_all.sh` (about 2 hours on a T4). Add `RUN_EXTERNAL=1` for the external evaluation (once only).
All seeds are set and logged (`outputs/seed_log.txt`); GPU kernels are not bit-exact, so numbers can differ in the third or fourth decimal.

## Which script produced what
| Report part | Script | Main outputs (`outputs/`, `figures/`) |
|---|---|---|
| P1 §1 Inventory, pixel statistics | `audit.py` | `inventory.csv`, `counts.csv`, `pixel_stats_by_class.csv`, fig01, fig02 |
| P1 §2 Duplicates and leakage | `audit.py`, `leakage_check.py` | `near_duplicate_pairs.csv`, `exact_duplicates.csv`, `leakage_check.csv`, fig03 |
| P1 §3 Validation design | `split.py`, `cache_data.py` | `split.csv`, `split.md5`, `split_summary.csv`, `val16_ci.csv` |
| P1 §4 Shortcut hunt | `shortcut.py` | `shortcut_results.csv` |
| P1 §5 Baselines | `baselines.py` | `baselines.csv` |
| P1 §6 Transfer baseline | `transfer.py` | `transfer_runs.csv`, `transfer_summary.csv`, `transfer_history.csv` |
| P1 §7 ArogyaNet and ablations | `models` in `src/models.py`, `arch_table.py`, `arch_diagram.py`, `run_variants.py` | `arch_table.txt`, `arogya_*_runs.csv`, fig04 |
| P1 §8 Two-branch | `run_variants.py --variants meta` | `arogya_meta_224_aug_*` |
| P1 §9 Resolution, augmentation | `run_variants.py --size 96`, `--no_aug` | `arogya_full_96_aug_*`, `arogya_full_224_noaug_*` |
| P2 T1 Threshold and costs | `threshold.py`, `export_preds.py` | `threshold_*.csv`, `camp_arithmetic.csv`, `operating_point.json`, fig05, fig06 |
| P2 T2 Evaluation | `cv_eval.py`, `final_eval.py` | `cv_*.csv`, `test_runs.csv`, `test_preds_seed0.csv`, fig10 |
| P2 T3 Tuning, overfitting | `tune.py` | `tune_*.csv`, fig07 |
| P2 T4 Ensembles | `ensemble.py`, `features.py` | `ensemble_table.csv`, `embed_models_runs.csv` |
| P2 T5 Feature space | `features.py` | `error_profile.csv`, fig08, fig09 |
| P2 T6 External test and fix | `train_fix.py`, `external_eval.py`, `external_posthoc.py` | `fix_runs.csv`, `external_*.csv`, fig11, fig12 |
| P2 T7 Shipping | `build_gate.py`, `predict.py`, `app.py`, `band_eval.py`, `ood_tests.py`, `latency.py` | `model/`, `ood_tests.txt`, `latency.txt`, fig13 |

## Use the tool
`pip install -r requirements-demo.txt`, then `python predict.py <image>` or `python app.py`. Screening aid for children aged 1 to 5; not a diagnosis.
