"""
EDA Diagnostic v2 — Outlier Detection + Paper-Quality Figures
==============================================================
Works on an existing NPZ produced by electricity_preprocessor_v2.py.
No need to re-preprocess.

What it does:
  1. Reports OUTLIER fractions for gap and consumption (the broken
     parts of the v1 EDA charts).
  2. Re-plots with:
       - log-spaced bins for heavy-tailed quantities
       - log-y axis where needed
       - x-axis clipped to the physically-meaningful range
  3. Prints percentile tables for the paper.
  4. Saves clean PNG figures sized for direct paper inclusion.

Usage:
    python eda_diagnostic.py meters_electricity_peer_loo.npz

Output:
    diagnostic_summary.txt       — printable summary (percentiles + outliers)
    paper_figs/gap_hist.png      — gap dist, clipped to [0, 200] days
    paper_figs/consumption_hist.png  — log-binned, log-y
    paper_figs/rate_hist.png         — log-binned, log-y
    paper_figs/reads_per_meter.png   — linear, count
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
MAX_REASONABLE_GAP_DAYS = 365   # anything > this is utility data error
GAP_VIEW_MAX            = 200   # x-axis cap for the readable plot
OUT_DIR                 = "paper_figs"


def pct(arr, ps=(0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)):
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def fmt_table(name, p):
    return (
        f"\n=== {name} ===\n"
        + "\n".join(f"  {k:>6}: {v:>14,.3f}" for k, v in p.items())
    )


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main(npz_path):
    print(f"⏳ loading {npz_path} ...")
    d = np.load(npz_path, allow_pickle=True)
    values = d["values"]
    N = len(values)
    print(f"   N meters = {N:,}")

    Path(OUT_DIR).mkdir(exist_ok=True)

    rates, dts, cons, n_reads = [], [], [], np.zeros(N, dtype=np.int64)
    for i in range(N):
        v = values[i]
        if v.shape[0] == 0:
            continue
        n_reads[i] = v.shape[0]
        rates.append(v[:, 0])
        dts.append(v[:, 1])
        cons.append(v[:, 0] * v[:, 1])
    rates = np.concatenate(rates).astype(np.float64)
    dts   = np.concatenate(dts).astype(np.float64)
    cons  = np.concatenate(cons).astype(np.float64)
    total = len(rates)
    print(f"   total readings = {total:,}")

    # ─── (1) OUTLIER REPORT ───────────────────────────────────
    out_gap_365 = (dts > 365).sum()
    out_gap_180 = (dts > 180).sum()
    out_gap_1k  = (dts > 1000).sum()
    cons_max    = cons.max()
    rate_max    = rates.max()

    txt = []
    txt.append(f"COHORT")
    txt.append(f"  meters         : {N:,}")
    txt.append(f"  total readings : {total:,}")
    txt.append(f"  per meter      : median {np.median(n_reads):.0f}, "
               f"mean {n_reads.mean():.1f}, max {n_reads.max()}")

    txt.append(f"\nOUTLIERS (utility data errors)")
    txt.append(f"  gap > 180 days : {out_gap_180:>12,} "
               f"({100*out_gap_180/total:.3f}% of readings)")
    txt.append(f"  gap > 365 days : {out_gap_365:>12,} "
               f"({100*out_gap_365/total:.3f}% of readings)")
    txt.append(f"  gap > 1000 days: {out_gap_1k:>12,} "
               f"({100*out_gap_1k/total:.3f}% of readings)")
    txt.append(f"  max gap        : {dts.max():,.0f} days "
               f"(={dts.max()/365:.1f} years — physically impossible)")
    txt.append(f"  max consumption: {cons_max:,.0f} kWh")
    txt.append(f"  max daily-rate : {rate_max:,.0f} kWh/day")

    txt.append(fmt_table("GAP (days) — all readings",      pct(dts)))
    txt.append(fmt_table("GAP (days) — gap ≤ 365 only",   pct(dts[dts <= 365])))
    txt.append(fmt_table("CONSUMPTION (kWh) — gap ≤ 365",
                         pct(cons[dts <= 365])))
    txt.append(fmt_table("DAILY RATE (kWh/day) — gap ≤ 365",
                         pct(rates[dts <= 365])))

    txt.append(f"\nZERO-INFLATION")
    zero_rate_all   = float((cons == 0).mean())
    zero_rate_clean = float((cons[dts <= 365] == 0).mean())
    txt.append(f"  zero rate (all readings)        : {100*zero_rate_all:.3f}%")
    txt.append(f"  zero rate (after gap filter)    : {100*zero_rate_clean:.3f}%")

    txt.append(f"\nOVERDISPERSION")
    di_all   = cons.var() / cons.mean() if cons.mean() > 0 else float("nan")
    di_clean = (cons[dts <= 365].var() / cons[dts <= 365].mean()
                if cons[dts <= 365].mean() > 0 else float("nan"))
    txt.append(f"  dispersion index (all)          : {di_all:.1f}")
    txt.append(f"  dispersion index (gap ≤ 365)    : {di_clean:.1f}")

    summary = "\n".join(txt)
    print("\n" + summary)
    Path("diagnostic_summary.txt").write_text(summary, encoding="utf-8")

    # ─── (2) PAPER-QUALITY FIGURES ────────────────────────────
    # Clean subset (drop the utility-error outliers)
    clean = dts <= MAX_REASONABLE_GAP_DAYS

    # 2a. Gap distribution clipped
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(dts[clean & (dts <= GAP_VIEW_MAX)], bins=60,
            color="#3b82c4", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("inter-reading gap (days)")
    ax.set_ylabel("count")
    ax.set_title("Inter-reading gap distribution "
                 f"({100*clean.mean():.2f}% kept after $\\leq 365$ filter)")
    ax.axvline(60, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.text(62, ax.get_ylim()[1]*0.92, "nominal bimonthly\n(60 days)",
            fontsize=8, alpha=0.7)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/gap_hist.png", dpi=160)
    plt.close(fig)

    # 2b. Consumption — log-spaced bins, log-y
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    cons_pos = cons[(cons > 0) & clean]
    bins = np.logspace(0, np.log10(cons_pos.max()), 70)
    ax.hist(cons_pos, bins=bins, color="#c44d3b",
            edgecolor="white", linewidth=0.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("consumption per reading period (kWh)")
    ax.set_ylabel("count (log)")
    ax.set_title("Consumption distribution (positive readings)")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/consumption_hist.png", dpi=160)
    plt.close(fig)

    # 2c. Daily rate — log-spaced bins, log-y
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    r_pos = rates[(rates > 0) & clean]
    bins = np.logspace(np.log10(r_pos.min().clip(1e-3, None)),
                       np.log10(r_pos.max()), 70)
    ax.hist(r_pos, bins=bins, color="#2e8b57",
            edgecolor="white", linewidth=0.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("daily rate (kWh / day)")
    ax.set_ylabel("count (log)")
    ax.set_title("Daily-rate distribution (positive readings)")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/rate_hist.png", dpi=160)
    plt.close(fig)

    # 2d. Readings per meter — linear (already fine)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(n_reads, bins=60, color="#8b3bc4",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("readings per meter")
    ax.set_ylabel("number of meters")
    ax.set_title("Readings per meter")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/reads_per_meter.png", dpi=160)
    plt.close(fig)

    print(f"\n✅ Figures saved to {OUT_DIR}/")
    print(f"✅ Summary saved to diagnostic_summary.txt")


if __name__ == "__main__":
    npz = (sys.argv[1] if len(sys.argv) > 1
           else "meters_electricity_peer_loo.npz")
    main(npz)
