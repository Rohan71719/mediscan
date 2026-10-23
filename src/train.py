import copy, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def predict(model, data, ids, bs=256, use_meta=False):
    model.eval(); out = []
    for i in range(0, len(ids), bs):
        x, _, m = data.batch(ids[i:i + bs])
        with torch.autocast("cuda", dtype=torch.float16):
            lo = model(x, m) if use_meta else model(x)
        out.append(lo.float().squeeze(1))
    return torch.cat(out)                                   # logits

def train_model(model, data, train_ids, val_ids, seed, epochs=12, lr=3e-4, wd=1e-4, bs=64,
                augment=None, pos_weight=None, patience=4, use_meta=False, verbose=True,
                select_best=True):
    """Trains on train_ids. select_best=True: early-stop on VALIDATION loss and restore the best epoch.
    select_best=False: train all epochs, keep the final weights (used for overfitting curves).
    Call set_seed(seed) BEFORE building the model."""
    dev = data.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    pw = None if pos_weight is None else torch.tensor([pos_weight], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    rng = np.random.RandomState(seed)
    yv = data.y[torch.as_tensor(val_ids, device=dev)]
    best_loss, best_state, best_ep, bad, hist = 1e9, None, -1, 0, []
    for ep in range(epochs):
        model.train(); perm = rng.permutation(train_ids); tl = n = 0
        for i in range(0, len(perm), bs):
            ids = perm[i:i + bs]
            if len(ids) < 2: continue
            x, y, m = data.batch(ids, augment)
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(x, m) if use_meta else model(x)
            loss = lossf(lo.float().squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(ids); n += len(ids)
        sched.step()
        vlog = predict(model, data, val_ids, use_meta=use_meta)
        vl = F.binary_cross_entropy_with_logits(vlog, yv).item()
        auc = roc_auc_score(yv.cpu().numpy(), vlog.cpu().numpy())
        hist.append(dict(epoch=ep, train_loss=tl / n, val_loss=vl, val_auc=auc))
        if verbose: print(f"ep{ep:02d} train {tl/n:.4f} val {vl:.4f} auc {auc:.4f}")
        if not select_best:
            continue                      # no early stopping, no checkpoint selection: full curve
        if vl < best_loss - 1e-4:
            best_loss, best_state, best_ep, bad = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience: break
    if select_best:
        model.load_state_dict(best_state)
    probs = torch.sigmoid(predict(model, data, val_ids, use_meta=use_meta)).cpu().numpy()
    return model, hist, probs, best_ep
