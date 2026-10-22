# ArogyaNet hypotheses (written before any ArogyaNet result exists)

**Decision 1: residual (skip) connections around each depthwise-separable block.**
Purpose: let gradients and low-level detail (edges, rib borders) reach the later layers of a
network trained from scratch. Expected change if removed: balanced accuracy drops by about
1 percentage point and training is less stable (noisier validation loss).

**Decision 2: squeeze-and-excitation (channel attention) in each block.**
Purpose: let the network re-weight feature channels per image, for example emphasising texture
channels that signal opacity over background channels. Expected change if removed: a smaller
effect than decision 1 (under 1 percentage point in balanced accuracy), possibly inside the
seed noise, because the dataset is easy and the network small.

**Other choices (not ablated):** depthwise-separable convolutions (to keep the parameter count
low for the camp laptop) and global average pooling instead of flatten + dense (fewer
parameters, less overfitting, no dependence on exact input size).
