"""
Evaluate the GRU baseline checkpoint -> test MAE.
=================================================
Builds the GRU model exactly as in training (so the state_dict matches),
loads the checkpoint, and computes test-set MAE/RMSE in consumption (kWh)
units. Does NOT use v5b_segmented_eval.py (which expects the transformer
layer names).

USAGE:
    python eval_gru.py --checkpoint checkpoints_gru/best_ema.pt
    python eval_gru.py --checkpoint checkpoints_gru/best.pt
"""
import argparse
from pathlib import Path

import torch
import numpy as np

import run_paper_experiments as R
import run_gru_baseline as G   # reuses build_gru_model + GRUEncoder


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    abs_err, sq_err, n = 0.0, 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch)
        mu = out["mu"]
        gate = out["gate"]
        target = out["target"]
        mask = out["mask"].bool()
        pred = (1.0 - gate) * mu          # expected consumption
        e = (pred - target)[mask]
        abs_err += e.abs().sum().item()
        sq_err += (e * e).sum().item()
        n += int(mask.sum().item())
    mae = abs_err / max(n, 1)
    rmse = (sq_err / max(n, 1)) ** 0.5
    return mae, rmse, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = R.build_default_config()
    device, _ = R.setup_run(cfg)
    cards, train_ds, _, _, test_loader = R.load_splits(cfg)

    model = G.build_gru_model(cfg, cards, train_ds.rate_mean, device)
    state = torch.load(args.checkpoint, map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    print(f"[eval_gru] loaded {args.checkpoint}")

    mae, rmse, n = evaluate(model, test_loader, device)
    print("=" * 50)
    print(f"  GRU baseline — test set ({n:,} readings)")
    print(f"  MAE  = {mae:.2f} kWh")
    print(f"  RMSE = {rmse:.2f} kWh")
    print("=" * 50)


if __name__ == "__main__":
    main()
