"""
Train / Validation / Test Split — Electricity Meters
====================================================
Two strategies (configurable):

  (A) temporal_per_meter   — for each meter, the EARLIEST 80% of its
                             readings → train, next 10% → val,
                             last 10% → test.   ✔ default; preferred
                             for forecasting evaluation; this is what
                             reviewers will expect at TMLR.

  (B) by_meter             — random split of meters into 70/15/15;
                             each meter goes entirely to one split.
                             Useful only as a robustness check.

Output: one NPZ per split with the same schema as the preprocessor's
output, so the existing ImprovedWaterMeterDataset (rename!) can load
them directly.

Usage:
    python split_dataset.py meters_electricity_ready.npz --mode temporal
"""

import argparse
from pathlib import Path

import numpy as np


def temporal_per_meter_split(values, times, masks, abs_dates,
                             frac_train=0.80, frac_val=0.10):
    """Split EACH meter's reading sequence chronologically."""
    out = {k: ([], [], []) for k in ("times", "values", "masks", "abs_dates")}
    keep_idx = [[], [], []]   # train, val, test  → meter indices
    keep_lens = [[], [], []]  # readings kept per split per meter

    for i in range(len(values)):
        T = values[i].shape[0]
        if T < 5:
            continue
        n_tr = max(1, int(np.floor(T * frac_train)))
        n_va = max(1, int(np.floor(T * frac_val)))
        n_te = T - n_tr - n_va
        if n_te < 1:
            continue

        splits = [
            slice(0,             n_tr),
            slice(n_tr,          n_tr + n_va),
            slice(n_tr + n_va,   T),
        ]
        for s, sl in enumerate(splits):
            out["times"][s].append(times[i][sl])
            out["values"][s].append(values[i][sl])
            out["masks"][s].append(masks[i][sl])
            if abs_dates is not None:
                out["abs_dates"][s].append(abs_dates[i][sl])
            keep_idx[s].append(i)
            keep_lens[s].append(sl.stop - sl.start)

    return out, keep_idx


def by_meter_split(N, seed=0,
                   frac_train=0.70, frac_val=0.15):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_tr = int(frac_train * N); n_va = int(frac_val * N)
    return {
        "train": perm[:n_tr].tolist(),
        "val":   perm[n_tr:n_tr + n_va].tolist(),
        "test":  perm[n_tr + n_va:].tolist(),
    }


def save_split(npz_in, mode="temporal", out_prefix="split",
               seed=0, frac_train=0.80, frac_val=0.10):
    data = np.load(npz_in, allow_pickle=True)
    values    = data["values"]
    times     = data["times"]
    masks     = data["masks"]
    static    = data["static"]
    ramz      = data["ramz"]
    abs_dates = data["abs_dates"] if "abs_dates" in data.files else None
    N = len(values)

    print(f"⏳ splitting N={N:,} meters in mode={mode} ...")
    split_names = ["train", "val", "test"]

    if mode == "temporal":
        bundles, keep_idx = temporal_per_meter_split(
            values, times, masks, abs_dates,
            frac_train=frac_train, frac_val=frac_val)

        for s, name in enumerate(split_names):
            idx = np.asarray(keep_idx[s], dtype=np.int64)
            payload = {
                "times":  np.array(bundles["times"][s],  dtype=object),
                "values": np.array(bundles["values"][s], dtype=object),
                "masks":  np.array(bundles["masks"][s],  dtype=object),
                "static": static[idx],
                "ramz":   ramz[idx],
            }
            if abs_dates is not None:
                payload["abs_dates"] = np.array(bundles["abs_dates"][s],
                                                dtype=object)
            out_file = f"{out_prefix}_{name}.npz"
            np.savez_compressed(out_file, **payload)
            print(f"   {name:5s}: {len(idx):,} meters → {out_file}")

    elif mode == "by_meter":
        idx_map = by_meter_split(N, seed=seed,
                                 frac_train=0.70, frac_val=0.15)
        for name in split_names:
            idx = np.asarray(idx_map[name], dtype=np.int64)
            payload = {
                "times":  values_array_subset(times, idx),
                "values": values_array_subset(values, idx),
                "masks":  values_array_subset(masks, idx),
                "static": static[idx],
                "ramz":   ramz[idx],
            }
            if abs_dates is not None:
                payload["abs_dates"] = values_array_subset(abs_dates, idx)
            out_file = f"{out_prefix}_{name}.npz"
            np.savez_compressed(out_file, **payload)
            print(f"   {name:5s}: {len(idx):,} meters → {out_file}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("✅ split complete.")


def values_array_subset(arr, idx):
    return np.array([arr[i] for i in idx], dtype=object)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npz_in")
    p.add_argument("--mode", default="temporal",
                   choices=["temporal", "by_meter"])
    p.add_argument("--out_prefix", default="split")
    p.add_argument("--frac_train", type=float, default=0.80)
    p.add_argument("--frac_val",   type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    save_split(args.npz_in, mode=args.mode, out_prefix=args.out_prefix,
               seed=args.seed,
               frac_train=args.frac_train, frac_val=args.frac_val)
