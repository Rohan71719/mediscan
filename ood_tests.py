import argparse, tempfile, glob, random
from pathlib import Path
import numpy as np
from PIL import Image
from predict import analyze
from scipy.stats import beta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="your own photos, e.g. a dog photo")
    ap.add_argument("--xray_dir", default=None, help="folder of real chest X-rays (the primary dataset) for a false-rejection check")
    a = ap.parse_args(); rng = np.random.RandomState(0); items = []
    items += [("blank white", Image.fromarray(np.full((512, 512, 3), 255, np.uint8))),
              ("blank black", Image.fromarray(np.zeros((512, 512, 3), np.uint8))),
              ("uniform grey", Image.fromarray(np.full((512, 512), 128, np.uint8))),
              ("noise, grey", Image.fromarray(rng.randint(0, 256, (512, 512), dtype=np.uint8))),
              ("noise, colour", Image.fromarray(rng.randint(0, 256, (512, 512, 3), dtype=np.uint8))),
              ("gradient, grey", Image.fromarray(np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))))]
    try:
        from skimage import data
        for n in ["chelsea", "astronaut", "coffee"]:
            im = Image.fromarray(getattr(data, n)()); items += [(f"photo {n}, colour", im), (f"photo {n}, greyscale", im.convert("L"))]
    except Exception as e: print("skimage photos unavailable:", e)
    for p in a.extra:
        im = Image.open(p).convert("RGB"); items += [(f"{Path(p).name}, colour", im), (f"{Path(p).name}, greyscale", im.convert("L"))]
    fails = []
    with tempfile.TemporaryDirectory() as td:
        for k, (name, im) in enumerate(items):
            f = Path(td) / f"{k}.png"; im.save(f); r = analyze(str(f))
            print(f"{name:32s} {r['status'][:42]:42s}" + (f" p={r['prob']:.3f}" if "prob" in r else "") + (f" d2={r['d2']:.0f}" if "d2" in r else "") + (f"  ({r['reason']})" if "reason" in r else ""))
            if r["status"] != "REJECTED": fails.append(name)
    print("\nNOT rejected (these are failures):", fails if fails else "none")
    if a.xray_dir:
        fs = sorted(glob.glob(f"{a.xray_dir}/**/*.jp*g", recursive=True)); random.Random(0).shuffle(fs); fs = fs[:1000]
        rej = sum(analyze(f)["status"] == "REJECTED" for f in fs); n = len(fs)
        lo = beta.ppf(0.025, rej, n - rej + 1) if rej else 0.0; hi = beta.ppf(0.975, rej + 1, n - rej) if rej < n else 1.0
        print(f"real chest X-rays from the primary dataset (fixed sample, mixed train/val/test): {rej} of {n} rejected "
              f"({rej / n:.1%}, exact 95% interval {lo:.1%} to {hi:.1%})")
