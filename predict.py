import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from src.models import ArogyaNet

MODEL_DIR = Path(__file__).parent / "model"; _S = {}

def load():
    if not _S:
        m = ArogyaNet(); m.load_state_dict(torch.load(MODEL_DIR / "arogyanet.pt", map_location="cpu")); m.eval()
        g = np.load(MODEL_DIR / "gate_stats.npz")
        _S.update(model=m, cfg=json.load(open(MODEL_DIR / "config.json")), mu=torch.tensor(g["mu"]), prec=torch.tensor(g["prec"]))
    return _S

def prep(path):                                      # identical to cache_data.py / external_eval.py
    im = Image.open(path); fmt = im.format; colour = 0.0
    if im.mode not in ("L", "I;16", "1"):
        a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32)
        colour = max(np.abs(a[..., i] - a[..., j]).mean() for i, j in [(0, 1), (1, 2), (0, 2)])
    im = Image.open(path)
    if fmt == "JPEG": im.draft("L", (448, 448))
    return np.asarray(im.convert("L").resize((224, 224), Image.BILINEAR, reducing_gap=2.0), dtype=np.uint8), colour

@torch.no_grad()
def analyze(path):
    S = load(); c = S["cfg"]; m = S["model"]
    try: arr, colour = prep(path)
    except Exception as e: return dict(status="REJECTED", reason=f"cannot read image ({type(e).__name__})")
    if colour > c["max_colour_diff"]: return dict(status="REJECTED", reason="colour photograph, not a greyscale chest X-ray")
    if arr.std() < c["min_std"]: return dict(status="REJECTED", reason="blank or almost uniform image")
    x = torch.from_numpy(arr).float().div(255)[None, None]
    f = m.features(m.stem((x - 0.5) / 0.25)).mean((2, 3))
    z = f[0] - S["mu"]; d2 = float(z @ S["prec"] @ z)
    if d2 > c["gate_d2"]: return dict(status="REJECTED", reason="does not resemble the chest X-rays this tool was built for", d2=d2)
    p = float(torch.sigmoid(m.fc(f))[0, 0])
    if p >= c["band_hi"]: status = "PNEUMONIA SUSPECTED: refer to a clinician"
    elif p >= c["band_lo"]: status = "LOW CONFIDENCE: refer to a clinician"
    else: status = "NO PNEUMONIA SUSPECTED"
    return dict(status=status, prob=p, d2=d2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("image"); r = analyze(ap.parse_args().image)
    print(r["status"] if r["status"] == "REJECTED" else f"{r['status']}  (probability of pneumonia {r['prob']:.3f})")
    if "reason" in r: print("reason:", r["reason"])
    print("Screening aid for children aged 1 to 5; not a diagnosis.")
