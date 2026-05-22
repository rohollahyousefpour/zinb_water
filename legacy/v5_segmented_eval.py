"""
v5 Segmented Evaluation + Baselines
====================================
Two analyses on the trained v5 model, no retraining:

  (A) WELL-CONDITIONED SUBSET REFRAMING
      Partition val/test readings into "easy" vs "hard" by:
        - gap length (days since previous reading)
        - per-meter rate volatility (std of historical rate)
        - history length (number of prior readings)
      Report MAE on each subset, and on the well-conditioned core
      (small gap + low volatility + enough history).

  (B) NAIVE BASELINES on the SAME readings
        - persistence: predict last rate × current dt
        - peer_avg:    predict peer_avg rate × current dt
        - tariff-mean: predict the train mean consumption for that tariff
      So we can see how much the transformer actually adds.

Usage:
    python v5_segmented_eval.py \
        --checkpoint checkpoints_paper/best_ema.pt \
        --npz meters_electricity_test.npz \
        --cards static_cardinalities_ramz.json \
        --agg peer_avg_aggregates.pkl

If --checkpoint is omitted, only the baselines (B) and the data-driven
partition (A, using persistence as the "model") are computed — useful
as a quick sanity check without loading torch weights.
"""

import argparse
import json
import pickle
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np


# ════════════════════════════════════════════════════════════
# Value-channel indices (v5 schema)
# ════════════════════════════════════════════════════════════
VAL_IDX_RATE    = 0
VAL_IDX_DT      = 1
VAL_IDX_PEERAVG = 2
VAL_IDX_SINDOY  = 3
VAL_IDX_COSDOY  = 4


# ════════════════════════════════════════════════════════════
# Build a flat table of (per-reading) records from the NPZ
# ════════════════════════════════════════════════════════════
def build_records(npz_path, cards, tariff_idx):
    """
    Walk every meter, every valid reading t (t >= 1, so a previous
    reading exists), and emit a record dict with everything needed for
    both the partition and the baselines.

    Returns a dict of numpy arrays (columnar).
    """
    data = np.load(npz_path, allow_pickle=True)
    values = data["values"]
    masks  = data["masks"]
    static = data["static"]
    N = len(values)

    tariff_col = list(cards.keys()).index("tariff_code")

    rec = defaultdict(list)
    for i in range(N):
        v = values[i]
        if v.ndim != 2 or v.shape[0] < 2:
            continue
        m = masks[i].astype(bool)[:, 0]
        T = v.shape[0]

        rate    = v[:, VAL_IDX_RATE]
        dt      = v[:, VAL_IDX_DT]
        peer    = v[:, VAL_IDX_PEERAVG]
        tariff  = int(static[i, tariff_col])

        # Per-meter rate volatility (std over valid readings)
        valid_rates = rate[m]
        if valid_rates.size >= 2:
            meter_vol = float(np.std(valid_rates))
        else:
            meter_vol = 0.0

        # Walk readings t >= 1
        n_valid_so_far = int(m[0])
        last_valid_rate = rate[0] if m[0] else np.nan
        for t in range(1, T):
            if not m[t]:
                continue
            # Need a previous valid reading for persistence
            if not np.isfinite(last_valid_rate):
                n_valid_so_far += 1
                last_valid_rate = rate[t]
                continue

            cons_actual = rate[t] * dt[t]      # true consumption (target)
            gap = dt[t]                        # days since prev reading

            rec["meter"].append(i)
            rec["t"].append(t)
            rec["tariff"].append(tariff)
            rec["dt"].append(float(dt[t]))
            rec["gap"].append(float(gap))
            rec["history_len"].append(n_valid_so_far)
            rec["meter_vol"].append(meter_vol)
            rec["target"].append(float(cons_actual))
            # Baselines (consumption space)
            rec["pred_persist"].append(float(last_valid_rate * dt[t]))
            rec["pred_peer"].append(float(peer[t] * dt[t]))
            # rate features for reference
            rec["rate"].append(float(rate[t]))
            rec["peer_rate"].append(float(peer[t]))

            n_valid_so_far += 1
            last_valid_rate = rate[t]

    out = {k: np.asarray(v) for k, v in rec.items()}
    print(f"   built {len(out['target']):,} reading-records from {N:,} meters")
    return out


# ════════════════════════════════════════════════════════════
# Tariff-mean baseline (computed on TRAIN)
# ════════════════════════════════════════════════════════════
def compute_tariff_mean_consumption(train_npz, cards):
    """Mean consumption (rate*dt) per tariff, from train readings."""
    data = np.load(train_npz, allow_pickle=True)
    values = data["values"]
    masks  = data["masks"]
    static = data["static"]
    tariff_col = list(cards.keys()).index("tariff_code")

    sums = defaultdict(float)
    counts = defaultdict(int)
    global_sum = 0.0
    global_n = 0
    for i in range(len(values)):
        v = values[i]
        if v.ndim != 2 or v.shape[0] == 0:
            continue
        m = masks[i].astype(bool)[:, 0]
        tariff = int(static[i, tariff_col])
        cons = v[m, VAL_IDX_RATE] * v[m, VAL_IDX_DT]
        sums[tariff] += float(cons.sum())
        counts[tariff] += int(cons.size)
        global_sum += float(cons.sum())
        global_n += int(cons.size)

    means = {t: sums[t] / max(counts[t], 1) for t in sums}
    global_mean = global_sum / max(global_n, 1)
    return means, global_mean


# ════════════════════════════════════════════════════════════
# Model predictions (optional, requires torch + patches)
# ════════════════════════════════════════════════════════════
def compute_model_predictions(checkpoint, npz_path, cards_path, agg_path):
    """
    Load the v5 model and produce per-reading predictions aligned to the
    same records as build_records. Returns dict keyed by (meter, t) → pred.
    """
    import torch
    import Improved_embeding as orig          # noqa: F401
    import electricity_zinb_patches as p
    from torch.utils.data import DataLoader

    with open(cards_path) as f:
        cards = json.load(f, object_pairs_hook=OrderedDict)

    ds = p.ImprovedElectricityMeterDataset(npz_path, cards)
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        collate_fn=orig.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = p.ImprovedZINBElectricityMeterEncoder(
        d_model=192, n_heads=6, n_layers=5,
        static_cardinalities=cards, dropout=0.05,
        use_time_aware_attention=True, default_rate=4.6,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    preds_by_key = {}
    meter_offset = 0
    with torch.no_grad():
        for batch in loader:
            bsz = batch["values"].shape[0]
            batch_g = {k: (v.to(device) if torch.is_tensor(v) else v)
                       for k, v in batch.items()}
            out = model(batch_g)
            expected = ((1 - out["gate"]) * out["mu"]).cpu().numpy()
            mask = batch["mask"].numpy()
            for b in range(bsz):
                meter_i = meter_offset + b
                T = mask[b].sum()
                for t in range(mask.shape[1]):
                    if mask[b, t]:
                        preds_by_key[(meter_i, t)] = float(expected[b, t])
            meter_offset += bsz
    print(f"   model produced {len(preds_by_key):,} predictions")
    return preds_by_key


# ════════════════════════════════════════════════════════════
# Reporting helpers
# ════════════════════════════════════════════════════════════
def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def report_partition(rec, model_pred=None,
                     gap_easy=70, vol_easy=3.0, hist_easy=5):
    """Print MAE by partition. model_pred is array aligned to rec, or None."""
    target = rec["target"]
    gap    = rec["gap"]
    vol    = rec["meter_vol"]
    hist   = rec["history_len"]

    print("\n" + "=" * 64)
    print("(A) WELL-CONDITIONED SUBSET REFRAMING")
    print("=" * 64)
    print(f"Easy thresholds: gap<{gap_easy}d, meter_vol<{vol_easy}, "
          f"history>={hist_easy}")

    def show(name, sel, pred):
        n = int(sel.sum())
        if n == 0:
            print(f"  {name:<34} n=0")
            return
        m = mae(pred[sel], target[sel])
        frac = 100 * n / len(target)
        mean_t = float(target[sel].mean())
        print(f"  {name:<34} n={n:>8,} ({frac:4.1f}%)  "
              f"MAE={m:>7.1f}  mean_target={mean_t:>6.0f}")

    use_pred = model_pred if model_pred is not None else rec["pred_persist"]
    pred_label = "MODEL" if model_pred is not None else "persistence(proxy)"
    print(f"\nPredictor: {pred_label}")

    # Single-axis partitions
    print("\n-- by gap length --")
    for lo, hi in [(0, 40), (40, 70), (70, 100), (100, 200), (200, 1e9)]:
        sel = (gap >= lo) & (gap < hi)
        show(f"gap [{lo},{hi})", sel, use_pred)

    print("\n-- by meter rate volatility --")
    for lo, hi in [(0, 1), (1, 3), (3, 10), (10, 30), (30, 1e9)]:
        sel = (vol >= lo) & (vol < hi)
        show(f"vol [{lo},{hi})", sel, use_pred)

    print("\n-- by history length --")
    for lo, hi in [(1, 3), (3, 5), (5, 10), (10, 20), (20, 1e9)]:
        sel = (hist >= lo) & (hist < hi)
        show(f"hist [{lo},{hi})", sel, use_pred)

    # The well-conditioned core
    print("\n-- well-conditioned core vs the rest --")
    core = (gap < gap_easy) & (vol < vol_easy) & (hist >= hist_easy)
    show("WELL-CONDITIONED core", core, use_pred)
    show("HARD (everything else)", ~core, use_pred)
    show("ALL readings", np.ones_like(core, dtype=bool), use_pred)

    # Contribution to total error
    err = np.abs(use_pred - target)
    total = err.sum()
    print(f"\n  Core contributes {100*err[core].sum()/total:4.1f}% of total "
          f"abs error from {100*core.mean():4.1f}% of readings")
    print(f"  Hard contributes {100*err[~core].sum()/total:4.1f}% of total "
          f"abs error from {100*(~core).mean():4.1f}% of readings")


def report_baselines(rec, tariff_means, global_mean, model_pred=None):
    target = rec["target"]
    tariff = rec["tariff"]

    print("\n" + "=" * 64)
    print("(B) BASELINE COMPARISON (same readings)")
    print("=" * 64)

    pred_tariff_mean = np.array(
        [tariff_means.get(t, global_mean) for t in tariff])

    rows = [
        ("persistence (last rate × dt)", rec["pred_persist"]),
        ("peer_avg (shrunk rate × dt)",  rec["pred_peer"]),
        ("tariff-mean consumption",      pred_tariff_mean),
    ]
    if model_pred is not None:
        rows.append(("v5 TRANSFORMER",            model_pred))

    print(f"  {'predictor':<34} {'MAE':>9} {'RMSE':>9}")
    print("  " + "-" * 54)
    for name, pred in rows:
        m = mae(pred, target)
        r = float(np.sqrt(np.mean((pred - target) ** 2)))
        print(f"  {name:<34} {m:>9.1f} {r:>9.1f}")

    if model_pred is not None:
        best_baseline = min(mae(p, target) for _, p in rows[:-1])
        model_mae = mae(model_pred, target)
        gain = best_baseline - model_mae
        print(f"\n  Transformer beats best baseline by {gain:.1f} MAE "
              f"({100*gain/best_baseline:.1f}%)")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="meters_electricity_test.npz")
    ap.add_argument("--train_npz", default="meters_electricity_train.npz")
    ap.add_argument("--cards", default="static_cardinalities_ramz.json")
    ap.add_argument("--agg", default="peer_avg_aggregates.pkl")
    ap.add_argument("--checkpoint", default=None,
                    help="v5 model checkpoint (best_ema.pt). "
                         "If omitted, baselines + proxy partition only.")
    ap.add_argument("--gap_easy", type=float, default=70)
    ap.add_argument("--vol_easy", type=float, default=3.0)
    ap.add_argument("--hist_easy", type=int, default=5)
    args = ap.parse_args()

    with open(args.cards) as f:
        cards = json.load(f, object_pairs_hook=OrderedDict)

    tariff_idx = list(cards.keys()).index("tariff_code")

    print("Building reading records ...")
    rec = build_records(args.npz, cards, tariff_idx)

    print("Computing tariff-mean baseline from train ...")
    tariff_means, global_mean = compute_tariff_mean_consumption(
        args.train_npz, cards)

    # Optional model predictions
    model_pred = None
    if args.checkpoint:
        print(f"Loading model from {args.checkpoint} ...")
        preds_by_key = compute_model_predictions(
            args.checkpoint, args.npz, args.cards, args.agg)
        model_pred = np.array([
            preds_by_key.get((m, t), np.nan)
            for m, t in zip(rec["meter"], rec["t"])
        ])
        # Drop records the model didn't cover (shouldn't happen)
        ok = np.isfinite(model_pred)
        if not ok.all():
            print(f"   WARNING: {int((~ok).sum())} records lack model pred; "
                  f"dropping")
            for k in rec:
                rec[k] = rec[k][ok]
            model_pred = model_pred[ok]

    report_partition(rec, model_pred,
                     gap_easy=args.gap_easy,
                     vol_easy=args.vol_easy,
                     hist_easy=args.hist_easy)
    report_baselines(rec, tariff_means, global_mean, model_pred)

    print("\nDone.")


if __name__ == "__main__":
    main()
