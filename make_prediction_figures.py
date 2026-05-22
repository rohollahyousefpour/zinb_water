"""
Prediction Visualisations for the Paper
========================================
Generates two qualitative figures from the self-contained prediction
dump (xfmr_test_pred.npz, written by v5b_segmented_eval.py --save_pred):

  1. pred_vs_actual.png  — hexbin of predicted vs actual consumption
     (log-log), with the ideal diagonal. Shows calibration across the
     full range and where the model over/under-predicts.

  2. example_trajectories.png — a small multi-panel figure of individual
     meters' actual vs predicted consumption over their reading sequence,
     chosen to span easy -> hard regimes (low/medium/high volatility,
     and one with a long gap). Makes the irregular-sampling story
     concrete.

The npz must contain at least: pred, target, vol, gap, meter, t.
(meter and t let us reconstruct per-meter sequences for figure 2.)

Usage:
    python make_prediction_figures.py \
        --pred_npz xfmr_test_pred.npz \
        --out_dir results_paper
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_pred_vs_actual(d, out_dir):
    pred = d["pred"].astype(float)
    tgt = d["target"].astype(float)

    # work in log space for the heavy tail; keep strictly positive
    m = (pred > 0.5) & (tgt > 0.5)
    p, t = pred[m], tgt[m]

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    hb = ax.hexbin(t, p, gridsize=55, bins="log",
                   xscale="log", yscale="log", cmap="viridis", mincnt=1)
    lim_lo = max(min(t.min(), p.min()), 0.5)
    lim_hi = max(t.max(), p.max())
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            "r--", lw=1.5, label="ideal ($\\hat c = c$)")
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Actual consumption (kWh, log scale)")
    ax.set_ylabel("Predicted consumption (kWh, log scale)")
    ax.set_title("Predicted vs. actual (test set)")
    ax.legend(loc="upper left", fontsize=9)
    cb = fig.colorbar(hb, ax=ax, shrink=0.85)
    cb.set_label("count (log)")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(Path(out_dir) / f"pred_vs_actual.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote pred_vs_actual.png/.svg  ({m.sum():,} points)")


def fig_trajectories(d, out_dir, n_per_class=1, seed=0):
    """Reconstruct per-meter sequences and plot actual vs predicted."""
    needed = {"pred", "target", "meter", "t", "vol", "gap"}
    if not needed.issubset(set(d.files)):
        print("  (skipping trajectories: npz lacks meter/t/vol/gap keys)")
        return

    pred = d["pred"].astype(float)
    tgt = d["target"].astype(float)
    meter = d["meter"].astype(int)
    t = d["t"].astype(int)
    vol = d["vol"].astype(float)
    gap = d["gap"].astype(float)

    rng = np.random.default_rng(seed)

    # group reading indices by meter
    order = np.argsort(meter, kind="stable")
    meter_s = meter[order]
    uniq, starts = np.unique(meter_s, return_index=True)
    groups = np.split(order, starts[1:])

    # per-meter summary for class selection
    def meter_vol(g):
        return float(np.median(vol[g]))
    def meter_len(g):
        return len(g)
    def meter_maxgap(g):
        return float(gap[g].max())

    # candidate meters with enough readings to be interesting
    cand = [(u, g) for u, g in zip(uniq, groups) if meter_len(g) >= 8]

    picks = []
    # low volatility
    low = [ (u,g) for u,g in cand if meter_vol(g) < 1.0 ]
    med = [ (u,g) for u,g in cand if 1.0 <= meter_vol(g) < 5.0 ]
    high= [ (u,g) for u,g in cand if meter_vol(g) >= 10.0 ]
    longgap = [ (u,g) for u,g in cand if meter_maxgap(g) >= 120 ]

    def pick_one(pool, label):
        if pool:
            u, g = pool[rng.integers(len(pool))]
            picks.append((u, g, label))

    pick_one(low,  "low volatility (easy)")
    pick_one(med,  "medium volatility")
    pick_one(high, "high volatility (hard)")
    pick_one(longgap, "long-gap meter")

    if not picks:
        print("  (skipping trajectories: no suitable meters found)")
        return

    n = len(picks)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.4 * n), squeeze=False)
    for ax, (u, g, label) in zip(axes[:, 0], picks):
        gg = g[np.argsort(t[g])]
        x = np.arange(len(gg))
        ax.plot(x, tgt[gg], "o-", color="#1565c0", ms=4, lw=1.4,
                label="actual")
        ax.plot(x, pred[gg], "s--", color="#ad1457", ms=4, lw=1.4,
                label="predicted")
        ax.set_title(f"Meter (id {int(u)}) — {label}", fontsize=10)
        ax.set_ylabel("kWh")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    axes[-1, 0].set_xlabel("Reading index (chronological)")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(Path(out_dir) / f"example_trajectories.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote example_trajectories.png/.svg  ({n} meters)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_npz", default="xfmr_test_pred.npz")
    ap.add_argument("--out_dir", default="results_paper")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    d = np.load(args.pred_npz)
    print(f"loaded {args.pred_npz}: keys = {list(d.files)}")

    fig_pred_vs_actual(d, args.out_dir)
    fig_trajectories(d, args.out_dir, seed=args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
