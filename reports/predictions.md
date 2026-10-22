# Prediction for the transfer baseline (written before running transfer.py)

**Model:** ResNet18 with ImageNet weights, fine-tuned end to end on the training split
(224×224 greyscale, replicated to 3 channels), evaluated on the validation split. Three seeds
(0, 1, 2), reported as mean ± standard deviation.

**What I expect to transfer.** The early and middle layers detect edges, textures, gradients and
blobs. These are generic and should help on X-rays, where lung borders, rib edges and opacity
patterns are texture-like. I also expect pretraining to make training converge faster and more
stably than training from scratch.

**What I expect not to transfer.** Colour sensitivity (the input is greyscale, so the three
channels are identical) and the object-level features in the late layers (animals, vehicles),
which will be changed during fine-tuning. ImageNet pixel statistics also differ from X-ray
statistics, so the first layers will need to adapt.

**Numeric expectation (validation set).** ROC-AUC about 0.98 to 0.99 and balanced accuracy about
0.92 to 0.95. That is at least as good as the HOG baseline (balanced accuracy 0.917, PR-AUC
0.994), but only by a small margin, because HOG is already strong and the validation set has just
227 NORMAL images. I expect a seed-to-seed standard deviation in balanced accuracy of about
1 percentage point.

**Caveat.** Part of any gain may come from the shortcuts found in Section 4 (image size, shape and
non-lung regions) rather than from better lung features, so a high score does not prove that the
model has learned pneumonia.
