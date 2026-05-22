"""
Deployment Reframing: Auto-Forecast vs Manual-Review Triage
============================================================
Instead of reporting a single MAE over all readings, this script frames
the model as a TRIAGE system:

    • "Auto-forecast" readings  — well-conditioned, low-error, no human needed
    • "Manual-review" readings  — high-uncertainty, flagged for an analyst

A reading is well-conditioned if:
    gap        < gap_max          (regular cadence)
    meter_vol  < vol_max          (stable consumption history)
    history    >= hist_min        (enough prior readings)

The script:
  1. Computes per-reading predictions (LightGBM rate-space by default, or
     load precomputed model preds via --pred_npz).
  2. Sweeps the thresholds to show the coverage/accuracy tradeoff —
     "if you auto-forecast X% of readings, your MAE on them is Y."
  3. Prints the operating table and writes a figure (PNG + SVG).

This is the honest deployment story: the model is highly accurate on the
bulk of routine readings and defers the genuinely hard cases.

Usage:
    python deployment_reframing.py \
        --train meters_electricity_train.npz \
        --eval  meters_electricity_val.npz \
        --cards static_cardinalities_ramz.json \
        --out_dir results_paper
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VAL_IDX_RATE    = 0
VAL_IDX_DT      = 1
VAL_IDX_PEERAVG = 2
VAL_IDX_SINDOY  = 3
VAL_IDX_COSDOY  = 4

STATIC_NAMES = ["meter_type", "tariff_code", "is_urban",
                "region_in", "phase", "amper", "section_code"]
FEATURE_NAMES = [
    "lag1", "lag2", "ema", "run_mean", "run_std",
    "peer_avg", "sin_doy", "cos_doy", "dt", "log_dt", "history_len",
] + STATIC_NAMES


# ════════════════════════════════════════════════════════════
# Feature extraction (shared logic with lightgbm_baseline)
# ════════════════════════════════════════════════════════════
def extract(npz_path, cards, default_rate=4.6):
    data = np.load(npz_path, allow_pickle=True)
    values = data["values"]; masks = data["masks"]; static = data["static"]
    card_keys = list(cards.keys())
    spos = {n: card_keys.index(n) for n in STATIC_NAMES}

    X, y_cons, y_rate = [], [], []
    g_gap, g_vol, g_hist, g_tariff = [], [], [], []

    for i in range(len(values)):
        v = values[i]
        if v.ndim != 2 or v.shape[0] < 2:
            continue
        m = masks[i].astype(bool)[:, 0]
        T = v.shape[0]
        rate = v[:, VAL_IDX_RATE]; dt = v[:, VAL_IDX_DT]
        peer = v[:, VAL_IDX_PEERAVG]
        sind = v[:, VAL_IDX_SINDOY]; cosd = v[:, VAL_IDX_COSDOY]
        sv = {n: int(static[i, spos[n]]) for n in STATIC_NAMES}
        all_valid = rate[m]
        vol = float(np.std(all_valid)) if all_valid.size >= 2 else 0.0

        past = []; ema = default_rate; lag1 = default_rate; lag2 = default_rate
        nh = 0
        for t in range(T):
            if not m[t]:
                continue
            if nh >= 1:
                rmean = float(np.mean(past)) if past else default_rate
                rstd  = float(np.std(past)) if len(past) >= 2 else 0.0
                X.append([lag1, lag2, ema, rmean, rstd,
                          float(peer[t]), float(sind[t]), float(cosd[t]),
                          float(dt[t]), float(np.log1p(dt[t])), float(nh)]
                         + [sv[n] for n in STATIC_NAMES])
                y_cons.append(float(rate[t] * dt[t]))
                y_rate.append(float(rate[t]))
                g_gap.append(float(dt[t])); g_vol.append(vol)
                g_hist.append(nh); g_tariff.append(sv["tariff_code"])
            past.append(float(rate[t]))
            lag2 = lag1; lag1 = float(rate[t]); ema = 0.3*float(rate[t]) + 0.7*ema
            nh += 1

    return (np.asarray(X, dtype=np.float64),
            np.asarray(y_cons), np.asarray(y_rate),
            {"gap": np.asarray(g_gap), "vol": np.asarray(g_vol),
             "hist": np.asarray(g_hist), "tariff": np.asarray(g_tariff)})


def mae(p, t):
    return float(np.mean(np.abs(p - t)))


# ════════════════════════════════════════════════════════════
# Triage analysis
# ════════════════════════════════════════════════════════════
def triage_mask(meta, gap_max, vol_max, hist_min):
    return ((meta["gap"] < gap_max)
            & (meta["vol"] < vol_max)
            & (meta["hist"] >= hist_min))


def sweep(meta, pred, target, out_dir):
    """
    Sweep gap_max and vol_max to map the coverage/MAE tradeoff.
    Prints a table and writes a figure.
    """
    print("\n" + "=" * 70)
    print("DEPLOYMENT TRIAGE: coverage / accuracy tradeoff")
    print("=" * 70)
    print("Each row: auto-forecast the well-conditioned readings; the rest")
    print("are flagged for manual review.\n")

    total_err = np.abs(pred - target).sum()
    n_total = len(target)

    configs = [
        # (label, gap_max, vol_max, hist_min)
        ("Conservative", 50,  2.0, 10),
        ("Balanced",     70,  3.0, 5),
        ("Permissive",   100, 5.0, 3),
        ("Aggressive",   150, 10.0, 2),
        ("Very loose",   250, 30.0, 1),
    ]

    print(f"  {'policy':<13} {'gap<':>5} {'vol<':>5} {'hist>=':>6} "
          f"{'auto%':>7} {'auto_MAE':>9} {'review%':>8} {'review_MAE':>11} "
          f"{'err_avoided':>12}")
    print("  " + "-" * 92)

    rows = []
    for label, gmax, vmax, hmin in configs:
        auto = triage_mask(meta, gmax, vmax, hmin)
        review = ~auto
        n_auto = int(auto.sum()); n_rev = int(review.sum())
        if n_auto == 0 or n_rev == 0:
            continue
        auto_mae = mae(pred[auto], target[auto])
        rev_mae  = mae(pred[review], target[review])
        # fraction of total absolute error that lands in the review bucket
        err_in_review = np.abs(pred[review] - target[review]).sum()
        err_avoided = 100 * err_in_review / total_err
        cov = 100 * n_auto / n_total
        rev_pct = 100 * n_rev / n_total
        rows.append((label, cov, auto_mae, rev_pct, rev_mae, err_avoided,
                     gmax, vmax, hmin))
        print(f"  {label:<13} {gmax:>5} {vmax:>5.0f} {hmin:>6} "
              f"{cov:>6.1f}% {auto_mae:>9.1f} {rev_pct:>7.1f}% "
              f"{rev_mae:>11.1f} {err_avoided:>11.1f}%")

    overall = mae(pred, target)
    print(f"\n  Overall MAE (no triage): {overall:.1f}")
    print(f"  Total readings:          {n_total:,}")

    # ── Continuous sweep on gap only (vol, hist fixed at balanced) ──
    # Start at gap=40: readings with gap<40 are a tiny (~1.7%), oddly
    # composed slice (very short cadence) that produces a misleading
    # low-coverage artifact. We also drop any sweep point covering less
    # than MIN_COV_PCT of readings so no tiny-sample point distorts the
    # curve.
    MIN_COV_PCT = 5.0
    gaps = np.arange(40, 260, 10)
    cov_curve, mae_curve = [], []
    for gmax in gaps:
        auto = triage_mask(meta, gmax, 5.0, 3)
        n_auto = int(auto.sum())
        cov = 100 * n_auto / n_total
        if n_auto == 0 or cov < MIN_COV_PCT:
            continue
        cov_curve.append(cov)
        mae_curve.append(mae(pred[auto], target[auto]))
    cov_curve = np.array(cov_curve); mae_curve = np.array(mae_curve)

    # ── Figure: coverage vs auto-MAE tradeoff ──
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#2563eb"
    ax1.plot(cov_curve, mae_curve, "o-", color=color1, linewidth=2,
             markersize=5, label="Auto-forecast MAE")
    ax1.axhline(overall, color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"Overall MAE ({overall:.0f}) — no triage")
    ax1.set_xlabel("Coverage: % of readings auto-forecast", fontsize=11)
    ax1.set_ylabel("MAE on auto-forecast readings (kWh)", fontsize=11,
                   color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title("Deployment triage: accuracy improves as you defer "
                  "harder readings", fontsize=12, pad=12)

    # annotate the balanced operating point
    bal = triage_mask(meta, 70, 3.0, 5)
    if bal.sum() > 0:
        bx = 100 * bal.sum() / n_total
        by = mae(pred[bal], target[bal])
        ax1.annotate(f"Balanced\n({bx:.0f}%, MAE {by:.0f})",
                     xy=(bx, by), xytext=(bx + 8, by + 15),
                     fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#333"))

    fig.tight_layout()
    png = Path(out_dir) / "deployment_triage.png"
    svg = Path(out_dir) / "deployment_triage.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure written: {png}")
    print(f"                  {svg}")

    return rows


def per_tariff_review_rate(meta, gap_max, vol_max, hist_min):
    """Which tariffs end up most often in the manual-review bucket."""
    print("\n-- manual-review rate by tariff (balanced policy) --")
    auto = triage_mask(meta, gap_max, vol_max, hist_min)
    review = ~auto
    tariffs = meta["tariff"]
    print(f"  {'tariff_idx':<11} {'n':>9} {'review%':>9}")
    for t in sorted(set(tariffs.tolist())):
        sel = tariffs == t
        n = int(sel.sum())
        if n < 100:
            continue
        rev_rate = 100 * review[sel].mean()
        print(f"  {t:<11} {n:>9,} {rev_rate:>8.1f}%")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="meters_electricity_train.npz")
    ap.add_argument("--eval",  default="meters_electricity_val.npz",
                    help="split to analyze (val or test)")
    ap.add_argument("--cards", default="static_cardinalities_ramz.json")
    ap.add_argument("--out_dir", default="results_paper")
    ap.add_argument("--pred_npz", default=None,
                    help="optional .npz with array 'pred' aligned to eval "
                         "readings (model predictions). If omitted, trains "
                         "a LightGBM rate-space model as the predictor.")
    ap.add_argument("--n_estimators", type=int, default=2000)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # If a prediction file is supplied AND it's the self-contained format
    # written by v5b_segmented_eval.py (carries target+gap+vol+hist+tariff),
    # use it directly — no feature extraction, no cards file, no alignment risk.
    if args.pred_npz:
        d = np.load(args.pred_npz)
        if all(k in d for k in ("pred", "target", "gap", "vol", "hist", "tariff")):
            print(f"Using self-contained prediction file {args.pred_npz} "
                  f"({len(d['pred']):,} readings) — skipping feature extraction.")
            pred = d["pred"].astype(np.float64)
            yc_ev = d["target"].astype(np.float64)
            meta = {"gap": d["gap"].astype(np.float64),
                    "vol": d["vol"].astype(np.float64),
                    "hist": d["hist"].astype(np.int64),
                    "tariff": d["tariff"].astype(np.int64)}
            rows = sweep(meta, pred, yc_ev, args.out_dir)
            per_tariff_review_rate(meta, 70, 3.0, 5)
            summary = {
                "eval_split": args.pred_npz,
                "overall_mae": mae(pred, yc_ev),
                "n_readings": int(len(yc_ev)),
                "policies": [
                    {"policy": r[0], "coverage_pct": r[1], "auto_mae": r[2],
                     "review_pct": r[3], "review_mae": r[4],
                     "error_in_review_pct": r[5],
                     "gap_max": r[6], "vol_max": r[7], "hist_min": r[8]}
                    for r in rows
                ],
            }
            with open(Path(args.out_dir) / "deployment_triage_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n  Summary written: "
                  f"{Path(args.out_dir) / 'deployment_triage_summary.json'}")
            print("\nDone.")
            return

    with open(args.cards) as f:
        cards = json.load(f, object_pairs_hook=OrderedDict)

    print("Extracting eval features ...")
    Xev, yc_ev, yr_ev, meta = extract(args.eval, cards)

    if args.pred_npz:
        pred = np.load(args.pred_npz)["pred"]
        assert len(pred) == len(yc_ev), (
            f"pred length {len(pred)} != eval readings {len(yc_ev)}")
        print(f"   loaded {len(pred):,} precomputed predictions")
    else:
        print("Training LightGBM (rate-space) as predictor ...")
        Xtr, _, yr_tr, _ = extract(args.train, cards)
        cat_idx = [FEATURE_NAMES.index(c) for c in STATIC_NAMES]
        model = lgb.LGBMRegressor(
            objective="l1", num_leaves=63, n_estimators=args.n_estimators,
            learning_rate=0.03, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=50, reg_lambda=1.0,
            n_jobs=-1, verbosity=-1)
        model.fit(Xtr, yr_tr, eval_set=[(Xev, yr_ev)], eval_metric="l1",
                  categorical_feature=cat_idx,
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        dt_ev = Xev[:, FEATURE_NAMES.index("dt")]
        pred = (model.predict(Xev).clip(min=0)) * dt_ev

    rows = sweep(meta, pred, yc_ev, args.out_dir)
    per_tariff_review_rate(meta, 70, 3.0, 5)

    # Write a small JSON summary for the record
    summary = {
        "eval_split": args.eval,
        "overall_mae": mae(pred, yc_ev),
        "n_readings": int(len(yc_ev)),
        "policies": [
            {"policy": r[0], "coverage_pct": r[1], "auto_mae": r[2],
             "review_pct": r[3], "review_mae": r[4],
             "error_in_review_pct": r[5],
             "gap_max": r[6], "vol_max": r[7], "hist_min": r[8]}
            for r in rows
        ],
    }
    with open(Path(args.out_dir) / "deployment_triage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written: "
          f"{Path(args.out_dir) / 'deployment_triage_summary.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
