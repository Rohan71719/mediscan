import argparse, time, statistics as st
from pathlib import Path
import numpy as np, torch
from predict import analyze, MODEL_DIR

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    print("torch", torch.__version__, "| logical CPUs:", torch.get_num_threads())
    for th in [1, 2, 4]:
        torch.set_num_threads(th)
        for _ in range(15): analyze(a.image)                                   # warm-up
        t = []
        for _ in range(a.n):
            s = time.perf_counter(); analyze(a.image); t.append((time.perf_counter() - s) * 1000)
        print(f"{th} thread(s): median {st.median(t):.1f} ms, 95th percentile {np.percentile(t, 95):.1f} ms, max {max(t):.1f} ms "
              f"(limit 200 ms, includes reading and preprocessing the image)")
    for f in sorted(MODEL_DIR.iterdir()): print(f"{f.name}: {f.stat().st_size / 1e6:.2f} MB")
    print(f"total model folder: {sum(f.stat().st_size for f in MODEL_DIR.iterdir()) / 1e6:.2f} MB (limit 25 MB)")
