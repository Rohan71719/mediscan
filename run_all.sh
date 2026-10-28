#!/usr/bin/env bash
# Reproduces every table and figure in the report. Needs a CUDA GPU; about 2 hours on a Kaggle T4.
# Data layout (searched recursively, folder names do not matter):
#   /kaggle/input/<any>/chest_xray/{train,val,test}/...          primary dataset
#   /kaggle/input/<any>/.../{Normal,Viral Pneumonia}/images/*.png external dataset (only with RUN_EXTERNAL=1)
# Results go to /kaggle/working/outputs and /kaggle/working/figures, then are copied to ./outputs and ./figures.
set -euo pipefail
cd "$(dirname "$0")"
[ -d /kaggle/input ] || { echo "Create /kaggle/input and /kaggle/working and put the datasets there (see README)"; exit 1; }
export PYTHONHASHSEED=0
OUT=$(python -c "from src.utils import OUT; print(OUT.resolve())")
FIG=$(python -c "from src.utils import FIG; print(FIG.resolve())")
step() { echo; echo "=== $* ==="; }

step "1. audit: inventory, pixel statistics, duplicates (Part 1, tasks 1-2)"; python audit.py
step "2. validation design: grouped stratified split (task 3)";               python split.py
step "3. strict leakage check (task 2)";                                      python leakage_check.py
step "4. cache 96 and 224 px arrays";                                         python cache_data.py
rm -rf /kaggle/input/mediscan-cache; cp -r "$OUT/cache" /kaggle/input/mediscan-cache

step "5. shortcut hunt and baselines (tasks 4-5)"; python shortcut.py; python baselines.py
step "6. transfer baseline, ResNet18 (task 6; needs internet once for ImageNet weights)"; python transfer.py
step "7. ArogyaNet: architecture, ablations, two-branch, resolution, augmentation (tasks 7-9)"
python arch_table.py; python arch_diagram.py
python run_variants.py --variants full no_res no_se meta
python run_variants.py --variants full --size 96
python run_variants.py --variants full --no_aug
python run_variants.py --variants full --save_models
python run_variants.py --variants full --pos_weight 0.371 --save_models
python train_fix.py
python export_preds.py
rm -rf /kaggle/input/mediscan-models; cp -r "$OUT/models" /kaggle/input/mediscan-models

step "8. threshold and cost matrix (Part 2, task 1)";            python threshold.py --preds "$OUT/val_preds.csv"
step "9. tuning and overfitting curves (task 3)";               python tune.py
step "10. 5-fold cross-validation (task 2)";                    python cv_eval.py
step "11. embeddings, forests, boosting, PCA/t-SNE/k-means (tasks 4-5)"; python features.py
step "12. ensembles (task 4)";                                  python ensemble.py --preds "$OUT/val_preds.csv" "$OUT/val_preds_embed.csv"
step "13. locked test set, evaluated once (task 2)";            python final_eval.py --confirm
step "14. shipping artefacts and checks (task 7)"
python build_gate.py
python band_eval.py --preds "$OUT/val_preds.csv" --col arogya_full_224_aug_saved_s0
PRIMARY=$(python -c "from src.utils import find_primary_root; print(find_primary_root())")
IMG=$(find "$PRIMARY" -name 'person108_bacteria_504.jpeg' | head -1)
python ood_tests.py --xray_dir "$PRIMARY" | tee "$OUT/ood_tests.txt"
python latency.py "$IMG" | tee "$OUT/latency.txt"

if [ "${RUN_EXTERNAL:-0}" = "1" ]; then
  step "15. external dataset, evaluated once (task 6)"; python external_eval.py --confirm; python external_posthoc.py
else
  echo; echo "Skipping the external evaluation (the external data may be used only once). Run: RUN_EXTERNAL=1 bash run_all.sh"
fi

mkdir -p outputs figures
if [ "$OUT" != "$(realpath outputs)" ]; then
  cp -r "$OUT"/*.csv "$OUT"/*.txt "$OUT"/*.json outputs/ 2>/dev/null || true
  cp -r "$FIG"/* figures/ 2>/dev/null || true
fi
echo; echo "done: tables in ./outputs, figures in ./figures"
