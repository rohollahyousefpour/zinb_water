"""
Exploratory Data Analysis — Electricity-Meter Dataset
=====================================================
Reads the NPZ produced by electricity_preprocessor.py and computes
every statistic needed for Paper 1 Section 5.1 (Dataset).

Outputs:
  - eda_summary.json     : machine-readable stats
  - eda_summary.md       : human-readable report (paste into paper)
  - eda_figures/         : histograms + diagnostic plots (PNG)

What it computes (all needed for the TMLR submission):
  1. Cohort   : N customers, total readings, time range, geography
  2. Sampling : reading-gap distribution (mean, median, percentiles)
  3. Targets  : consumption distribution, log-scale, dynamic range
  4. Zeros    : empirical zero rate (overall, per tariff, per region)
  5. Disp.    : dispersion index Var/Mean (overall and segmented)
  6. Static   : tariff / urban / region / phase / amper breakdowns

Usage:
    python eda_electricity.py  path/to/meters_electricity_ready.npz
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════
def percentiles(arr, ps=(1, 5, 25, 50, 75, 95, 99)):
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def dispersion_index(consumption):
    """Var / Mean for non-negative count-like data."""
    consumption = np.asarray(consumption, dtype=np.float64)
    m = consumption.mean()
    if m <= 0:
        return float("nan")
    return float(consumption.var() / m)


def save_hist(arr, path, bins=80, title="", xlabel="", logx=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if logx:
        a = a[a > 0]
        ax.set_xscale("log")
    ax.hist(a, bins=bins)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def run_eda(npz_path, fig_dir="eda_figures",
            out_json="eda_summary.json", out_md="eda_summary.md"):
    print(f"⏳ loading {npz_path} ...")
    data = np.load(npz_path, allow_pickle=True)

    values    = data["values"]
    times     = data["times"]
    masks     = data["masks"]
    static    = data["static"]
    has_abs   = "abs_dates" in data.files
    abs_dates = data["abs_dates"] if has_abs else None

    N = len(values)
    print(f"   N meters = {N:,}")

    Path(fig_dir).mkdir(exist_ok=True)

    # ─── (1) flatten everything for global stats
    all_rates, all_dts, all_cons, all_peers = [], [], [], []
    n_reads_per_meter = np.zeros(N, dtype=np.int64)
    timeline_start, timeline_end = None, None

    for i in range(N):
        v = values[i]
        if v.shape[0] == 0:
            continue
        n_reads_per_meter[i] = v.shape[0]
        rate = v[:, 0]; dt = v[:, 1]
        peer = v[:, 2] if v.shape[1] >= 3 else np.zeros_like(rate)
        cons = rate * dt
        all_rates.append(rate); all_dts.append(dt)
        all_cons.append(cons);  all_peers.append(peer)
        if has_abs and len(abs_dates[i]) > 0:
            lo, hi = abs_dates[i][0], abs_dates[i][-1]
            timeline_start = lo if timeline_start is None else min(timeline_start, lo)
            timeline_end   = hi if timeline_end   is None else max(timeline_end,   hi)

    all_rates = np.concatenate(all_rates)
    all_dts   = np.concatenate(all_dts)
    all_cons  = np.concatenate(all_cons)
    all_peers = np.concatenate(all_peers)
    total_reads = len(all_rates)

    # ─── (2) zero rate
    zero_mask_cons = all_cons == 0.0
    zero_rate = float(zero_mask_cons.mean())

    # ─── (3) dispersion index
    DI_global = dispersion_index(all_cons)

    # static features by column (order must match STATIC_CAT_COLS in
    # the preprocessor)
    STATIC_NAMES = ["meter_type", "tariff_code", "is_urban",
                    "region_in", "phase", "amper"]

    per_segment_DI, per_segment_zero = {}, {}
    if static.shape[1] >= 4:
        # by tariff
        tariffs = static[:, 1]
        unique_t = np.unique(tariffs)
        for t in unique_t:
            sel = np.zeros_like(all_cons, dtype=bool)
            # rebuild a per-reading meter index quickly
            offset = 0
            for mi in range(N):
                k = n_reads_per_meter[mi]
                if tariffs[mi] == t:
                    sel[offset:offset + k] = True
                offset += k
            if sel.sum() < 50:
                continue
            per_segment_DI[f"tariff_{int(t)}"]   = dispersion_index(all_cons[sel])
            per_segment_zero[f"tariff_{int(t)}"] = float((all_cons[sel] == 0).mean())

    # ─── (4) summary dict
    summary = {
        "cohort": {
            "n_meters": int(N),
            "n_readings_total": int(total_reads),
            "reads_per_meter": {
                "mean": float(n_reads_per_meter.mean()),
                "median": float(np.median(n_reads_per_meter)),
                "min": int(n_reads_per_meter.min()),
                "max": int(n_reads_per_meter.max()),
                **percentiles(n_reads_per_meter),
            },
            "timeline": {
                "start": str(timeline_start) if timeline_start is not None else None,
                "end":   str(timeline_end)   if timeline_end   is not None else None,
            },
        },
        "sampling_gaps_days": {
            "mean":   float(all_dts.mean()),
            "median": float(np.median(all_dts)),
            "std":    float(all_dts.std()),
            **percentiles(all_dts),
        },
        "consumption": {
            "mean":   float(all_cons.mean()),
            "median": float(np.median(all_cons)),
            "std":    float(all_cons.std()),
            "max":    float(all_cons.max()),
            **percentiles(all_cons),
        },
        "rate_per_day": {
            "mean":   float(all_rates.mean()),
            "median": float(np.median(all_rates)),
            "std":    float(all_rates.std()),
            **percentiles(all_rates),
        },
        "zero_inflation": {
            "overall_zero_rate": zero_rate,
            "per_tariff": per_segment_zero,
        },
        "overdispersion": {
            "global_dispersion_index": DI_global,
            "per_tariff_DI": per_segment_DI,
        },
        "static_breakdowns": {},
    }

    # static breakdowns
    for j, name in enumerate(STATIC_NAMES):
        if j >= static.shape[1]:
            break
        vals, counts = np.unique(static[:, j], return_counts=True)
        summary["static_breakdowns"][name] = {
            int(v): int(c) for v, c in zip(vals, counts)
        }

    # ─── (5) figures
    save_hist(n_reads_per_meter, f"{fig_dir}/reads_per_meter.png",
              title="Readings per meter", xlabel="# readings")
    save_hist(all_dts, f"{fig_dir}/gap_distribution.png",
              title="Inter-reading gap (days)", xlabel="days")
    save_hist(all_cons, f"{fig_dir}/consumption_hist.png",
              title="Consumption per period", xlabel="kWh", logx=True)
    save_hist(all_rates, f"{fig_dir}/rate_hist.png",
              title="Daily-rate distribution", xlabel="kWh/day", logx=True)

    # zero-share bar by tariff
    if per_segment_zero:
        keys = list(per_segment_zero.keys())
        vals = [per_segment_zero[k] for k in keys]
        fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.3), 4))
        ax.bar(range(len(keys)), vals)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=90, fontsize=7)
        ax.set_ylabel("zero rate"); ax.set_title("Zero rate by tariff")
        fig.tight_layout(); fig.savefig(f"{fig_dir}/zero_by_tariff.png", dpi=130)
        plt.close(fig)

    # ─── (6) save outputs
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    md = _render_markdown(summary)
    with open(out_md, "w") as f:
        f.write(md)

    print(f"✅ EDA complete:")
    print(f"   {out_json}")
    print(f"   {out_md}")
    print(f"   figures in {fig_dir}/")
    print(f"\nHeadline numbers for the paper:")
    print(f"   N meters          : {summary['cohort']['n_meters']:,}")
    print(f"   total readings    : {summary['cohort']['n_readings_total']:,}")
    print(f"   median gap (days) : {summary['sampling_gaps_days']['median']:.1f}")
    print(f"   zero rate (%)     : {100*zero_rate:.2f}")
    print(f"   dispersion index  : {DI_global:.1f}")


def _render_markdown(s):
    c = s["cohort"]; g = s["sampling_gaps_days"]
    cn = s["consumption"]; z = s["zero_inflation"]; o = s["overdispersion"]
    lines = []
    lines.append("# Electricity-Meter Dataset — EDA Summary\n")
    lines.append("## Cohort\n")
    lines.append(f"- **Meters**: {c['n_meters']:,}")
    lines.append(f"- **Total readings**: {c['n_readings_total']:,}")
    lines.append(f"- **Timeline**: {c['timeline']['start']} → {c['timeline']['end']}")
    rpm = c["reads_per_meter"]
    lines.append(f"- **Readings per meter**: median {rpm['median']:.0f}, "
                 f"p5 {rpm['p5']:.0f}, p95 {rpm['p95']:.0f}, max {rpm['max']}\n")

    lines.append("## Sampling Irregularity\n")
    lines.append(f"- **Median gap**: {g['median']:.1f} days")
    lines.append(f"- **IQR**: [{g['p25']:.1f}, {g['p75']:.1f}] days")
    lines.append(f"- **5th–95th pct**: [{g['p5']:.1f}, {g['p95']:.1f}] days\n")

    lines.append("## Consumption (per reading period)\n")
    lines.append(f"- **Median**: {cn['median']:.1f} kWh")
    lines.append(f"- **Mean**: {cn['mean']:.1f} kWh")
    lines.append(f"- **p99**: {cn['p99']:.0f} kWh, **max**: {cn['max']:.0f} kWh\n")

    lines.append("## Zero-Inflation\n")
    lines.append(f"- **Overall zero rate**: **{100*z['overall_zero_rate']:.2f}%**")
    lines.append("- Per-tariff zero rates: see `eda_summary.json`\n")

    lines.append("## Overdispersion\n")
    lines.append(f"- **Global dispersion index** (Var/Mean): "
                 f"**{o['global_dispersion_index']:.1f}**")
    lines.append("  (Poisson would give 1; values >> 1 confirm overdispersion → ZINB.)\n")

    lines.append("## Static Feature Breakdowns\n")
    for k, v in s["static_breakdowns"].items():
        total = sum(v.values())
        top = sorted(v.items(), key=lambda kv: -kv[1])[:5]
        lines.append(f"### `{k}` ({len(v)} categories, top 5):")
        for cat, cnt in top:
            lines.append(f"  - {cat}: {cnt:,} ({100*cnt/total:.1f}%)")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else "meters_electricity_ready.npz"
    run_eda(npz)
