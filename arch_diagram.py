import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.utils import FIG

boxes = ["Input\n1×224×224", "Stem conv 3×3, s2\n24 ch", "Stage 1: 2× DS-block\n32 ch, SE, skip",
         "Stage 2: 2× DS-block\n64 ch, SE, skip", "Stage 3: 3× DS-block\n128 ch, SE, skip",
         "Stage 4: 2× DS-block\n256 ch, SE, skip", "Global average pool\n256", "Dropout 0.2 + Linear\n1 logit"]
fig, ax = plt.subplots(figsize=(12, 2.6)); ax.axis("off"); ax.set_xlim(0, len(boxes) * 1.5); ax.set_ylim(0, 2)
for i, t in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((i * 1.5 + 0.05, 0.6), 1.3, 0.9, boxstyle="round,pad=0.03", fc="#dbe8f5", ec="k"))
    ax.text(i * 1.5 + 0.7, 1.05, t, ha="center", va="center", fontsize=7)
    if i: ax.annotate("", xy=(i * 1.5 + 0.05, 1.05), xytext=(i * 1.5 - 0.15, 1.05), arrowprops=dict(arrowstyle="->"))
ax.text(len(boxes) * 0.75, 0.2, "DS-block = depthwise 3×3 → pointwise 1×1 → SE → (+ skip) → ReLU", ha="center", fontsize=8)
plt.savefig(FIG / "fig04_arogyanet.png", dpi=200, bbox_inches="tight")
