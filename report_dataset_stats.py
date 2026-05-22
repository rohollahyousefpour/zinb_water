
"""
Dataset Characterisation for the Paper
======================================
Computes every descriptive statistic the Problem Formulation needs, and
produces a two-panel data-characterisation figure (gap distribution +
consumption distribution / zero-inflation). Reads the preprocessed NPZ
files directly — no model, no training.

Outputs:
  - prints a block of numbers you paste into the paper's \todo slots
  - writes data_characterisation.png / .svg  (Figure for the paper)
  - writes dataset_stats.json (machine-readable copy)

Usage:
    python report_dataset_stats.py \
        --train meters_electricity_train.npz \
        --val   meters_electricity_val.npz \
        --test  meters_electricity_test.npz \
        --cards static_cardinalities_ramz.json \
        --out_dir results_paper

Schema assumed (v5/v5b): values[:, :, 0]=consumption target (kWh over
the interval), values[:, :, 1]=dt (days). masks[:, :, 0]=validity.
Adjust TARGET_IDX / DT_IDX if your channel order differs.
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_IDX = 0   # consumption (kWh) per reading
DT_IDX     = 1   # gap in days


# ════════════════════════════════════════════════════════════
def collect(npz_path):
    """Flatten all valid readings into 1-D arrays of consumption, gap, rate."""
    d = np.load(npz_path, allow_pickle=True)
    values, masks = d["values"], d["masks"]
    cons, gap = [], []
    n_meters = len(values)
    seq_lengths = []
    for i in range(n_meters):
        v = values[i]
        if v.ndim != 2 or v.shape[0] == 0:
            continue
        m = masks[i].astype(bool)[:, 0]
        seq_lengths.append(int(m.sum()))
        c = v[m, TARGET_IDX].astype(np.float64)
        g = v[m, DT_IDX].astype(np.float64)
        cons.append(c); gap.append(g)
    cons = np.concatenate(cons) if cons else np.array([])
    gap  = np.concatenate(gap)  if gap  else np.array([])
    rate = cons / np.clip(gap, 0.5, None)
    return {
        "consumption": cons, "gap": gap, "rate": rate,
        "n_meters": n_meters, "seq_lengths": np.array(seq_lengths),
    }


def pct(a, qs):
    return {q: float(np.percentile(a, q)) for q in qs}


def describe(name, S, zero_thresh=0.5):
    c, g, r = S["consumption"], S["gap"], S["rate"]
    n = len(c)
    zero_rate = float((c < zero_thresh).mean())
    stats = {
        "split": name,
        "n_meters": int(S["n_meters"]),
        "n_readings": int(n),
        "median_seq_len": float(np.median(S["seq_lengths"])),
        "zero_rate_pct": 100 * zero_rate,
        "consumption": {
            "mean": float(c.mean()), "median": float(np.median(c)),
            "std": float(c.std()),
            "pctiles": pct(c, [50, 90, 95, 99, 99.5, 99.9]),
            "max": float(c.max()),
        },
        "gap_days": {
            "mean": float(g.mean()), "median": float(np.median(g)),
            "pctiles": pct(g, [5, 25, 50, 75, 95]),
            "bands_pct": {
                "<40":      100 * float((g < 40).mean()),
                "40-70":    100 * float(((g >= 40) & (g < 70)).mean()),
                "70-100":   100 * float(((g >= 70) & (g < 100)).mean()),
                "100-200":  100 * float(((g >= 100) & (g < 200)).mean()),
                ">=200":    100 * float((g >= 200).mean()),
            },
        },
    }
    return stats


def print_block(stats):
    s = stats
    print("\n" + "=" * 64)
    print(f"  {s['split'].upper()} SPLIT  —  paste these into the paper")
    print("=" * 64)
    print(f"  meters                 : {s['n_meters']:,}")
    print(f"  readings               : {s['n_readings']:,}")
    print(f"  median sequence length : {s['median_seq_len']:.0f}")
    print(f"  zero/near-zero rate    : {s['zero_rate_pct']:.1f}%")
    c = s["consumption"]
    print(f"  consumption mean/median: {c['mean']:.0f} / {c['median']:.0f} kWh")
    print(f"  consumption p90/p99    : {c['pctiles']['90']:.0f} / "
          f"{c['pctiles']['99']:.0f} kWh"
          if False else
          f"  consumption p90/p99    : {c['pctiles'][90]:.0f} / "
          f"{c['pctiles'][99]:.0f} kWh")
    print(f"  consumption p99.5/max  : {c['pctiles'][99.5]:.0f} / "
          f"{c['max']:.0f} kWh")
    g = s["gap_days"]
    print(f"  gap median (days)      : {g['median']:.0f}")
    print(f"  gap bands:")
    for k, v in g["bands_pct"].items():
        print(f"     {k:>8} days : {v:5.1f}%")


# ════════════════════════════════════════════════════════════
def make_figure(S_all, out_dir):
    """Two-panel figure: gap distribution + consumption distribution."""
    g = S_all["gap"]
    c = S_all["consumption"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- Panel A: gap distribution ----
    bins = np.arange(0, 220, 10)
    ax1.hist(np.clip(g, 0, 215), bins=bins, color="#2563eb",
             edgecolor="white", alpha=0.85)
    ax1.axvspan(40, 70, color="#22c55e", alpha=0.12,
                label="typical 40–70 d")
    ax1.set_xlabel("Reading gap $\\Delta t$ (days)")
    ax1.set_ylabel("Number of readings")
    ax1.set_title("(a) Irregular reading intervals")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # ---- Panel B: consumption distribution (log-x, with zero spike) ----
    zero_frac = float((c < 0.5).mean())
    nz = c[c >= 0.5]
    # log-spaced bins for the non-zero tail
    if nz.size:
        lo = max(nz.min(), 0.5)
        bins2 = np.logspace(np.log10(lo), np.log10(nz.max() + 1), 50)
        ax2.hist(nz, bins=bins2, color="#ad1457",
                 edgecolor="white", alpha=0.85)
        ax2.set_xscale("log")
    ax2.set_xlabel("Per-day rate (kWh/day, log scale)")
    ax2.set_ylabel("Number of readings")
    ax2.set_title(f"(b) Zero-inflation ({100*zero_frac:.1f}% near-zero) "
                  f"+ heavy tail")
    # annotate the zero mass
    ax2.annotate(f"{100*zero_frac:.0f}% near-zero\n(structural zeros)",
                 xy=(0.02, 0.95), xycoords="axes fraction",
                 fontsize=9, va="top",
                 bbox=dict(boxstyle="round", fc="#fde2ec", ec="#ad1457"))
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    png = Path(out_dir) / "data_characterisation.png"
    svg = Path(out_dir) / "data_characterisation.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure written: {png}\n                  {svg}")


# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="meters_electricity_train.npz")
    ap.add_argument("--val",   default="meters_electricity_val.npz")
    ap.add_argument("--test",  default="meters_electricity_test.npz")
    ap.add_argument("--cards", default="static_cardinalities_ramz.json")
    ap.add_argument("--out_dir", default="results_paper")
    ap.add_argument("--zero_thresh", type=float, default=0.5)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_stats = {}
    pooled = {"consumption": [], "gap": []}
    for name, path in [("train", args.train), ("val", args.val),
                       ("test", args.test)]:
        if not Path(path).exists():
            print(f"  (skipping {name}: {path} not found)")
            continue
        S = collect(path)
        st = describe(name, S, args.zero_thresh)
        print_block(st)
        all_stats[name] = st
        pooled["consumption"].append(S["consumption"])
        pooled["gap"].append(S["gap"])

    # combined dataset-wide totals (paper usually quotes the full archive)
    if all_stats:
        tot_meters = sum(s["n_meters"] for s in all_stats.values())
        tot_reads = sum(s["n_readings"] for s in all_stats.values())
        print("\n" + "=" * 64)
        print("  WHOLE ARCHIVE (all splits)")
        print("=" * 64)
        print(f"  total meters   : {tot_meters:,}")
        print(f"  total readings : {tot_reads:,}")

    # figure from pooled data
    S_all = {
        "consumption": np.concatenate(pooled["consumption"]),
        "gap": np.concatenate(pooled["gap"]),
    }
    make_figure(S_all, args.out_dir)

    with open(Path(args.out_dir) / "dataset_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  JSON written: {Path(args.out_dir) / 'dataset_stats.json'}")
    print("\nDone. Paste the printed numbers into the paper's \\todo slots,")
    print("and \\includegraphics the figure in the Problem Formulation.")


if __name__ == "__main__":
    main()