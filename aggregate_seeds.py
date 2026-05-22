"""
Aggregate Multi-Seed Results → mean ± std
==========================================
Reads the best MAE from each seeded run and reports mean, std, and the
individual values, so you can write "MAE = X ± Y over N seeds" in the paper.

It tries three sources, in order:
  1. results_paper_s{seed}/seed_result.json   (written by the patched main)
  2. checkpoints_paper_s{seed}/*.json log history (best val MAE)
  3. manual entry — pass --vals 129.5 130.2 128.9

USAGE:
    # automatic (after running the seeded trainings):
    python aggregate_seeds.py --seeds 42 43 44

    # or just hand it the numbers you read off the console:
    python aggregate_seeds.py --vals 129.5 130.2 128.9
"""

import argparse
import json
import glob
from pathlib import Path

import numpy as np


def from_seed_result(seed):
    p = Path(f"results_paper_s{seed}") / "seed_result.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    if "test_mae" in d:
        return float(d["test_mae"])
    # fall back to any mae-like key in all_metrics
    am = d.get("all_metrics", {})
    for k in ("mae", "test_mae", "MAE"):
        if k in am:
            try:
                return float(am[k])
            except (TypeError, ValueError):
                pass
    return None


def from_log_history(seed):
    """
    Look for a training-history JSON in the checkpoint dir and pull the
    best (minimum) val MAE. The exact filename/format depends on your
    TrainingLogger; we scan all JSONs and look for a 'val' MAE series.
    """
    cdir = Path(f"checkpoints_paper_s{seed}")
    if not cdir.exists():
        return None
    best = None
    for jf in glob.glob(str(cdir / "*.json")):
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception:
            continue
        # Try a few plausible shapes
        candidates = []
        if isinstance(data, dict):
            # e.g. {"val": [{"mae": ...}, ...]} or {"history": {...}}
            for key in ("val", "validation", "val_metrics"):
                seq = data.get(key)
                if isinstance(seq, list):
                    for row in seq:
                        if isinstance(row, dict) and "mae" in row:
                            candidates.append(float(row["mae"]))
            # flat {"val_mae": [...]}
            for key in ("val_mae", "val_MAE"):
                seq = data.get(key)
                if isinstance(seq, list):
                    candidates += [float(x) for x in seq]
        if candidates:
            m = min(candidates)
            best = m if best is None else min(best, m)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--vals", type=float, nargs="*", default=None,
                    help="manually supply the best MAE per seed; "
                         "overrides automatic lookup")
    args = ap.parse_args()

    if args.vals:
        vals = list(args.vals)
        labels = [f"manual[{i}]" for i in range(len(vals))]
    else:
        vals, labels = [], []
        for s in args.seeds:
            v = from_seed_result(s)
            src = "seed_result.json"
            if v is None:
                v = from_log_history(s)
                src = "log_history"
            if v is None:
                print(f"  seed {s}: NO RESULT FOUND "
                      f"(looked in results_paper_s{s}/ and "
                      f"checkpoints_paper_s{s}/)")
                continue
            vals.append(v)
            labels.append(f"seed {s} ({src})")

    if not vals:
        print("\nNo results found. Either run the seeded trainings first, "
              "or pass the numbers directly:")
        print("    python aggregate_seeds.py --vals 129.5 130.2 128.9")
        return

    vals = np.array(vals, dtype=float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    print("\n" + "=" * 50)
    print("MULTI-SEED RESULT")
    print("=" * 50)
    for lab, v in zip(labels, vals):
        print(f"  {lab:<28} MAE = {v:.2f}")
    print("-" * 50)
    print(f"  n runs    : {len(vals)}")
    print(f"  mean MAE  : {mean:.2f}")
    print(f"  std  MAE  : {std:.2f}  (sample std, ddof=1)")
    print(f"  min / max : {vals.min():.2f} / {vals.max():.2f}")
    print("-" * 50)
    print(f"  REPORT AS:  MAE = {mean:.1f} ± {std:.1f}  "
          f"(n={len(vals)} seeds)")
    print("=" * 50)


if __name__ == "__main__":
    main()
