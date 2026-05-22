"""
LightGBM Baseline for Electricity-Meter Forecasting
====================================================
Trains a LightGBM regressor on the SAME information the v5 transformer
sees, so you can answer the only question that matters: does the
transformer actually beat gradient boosting?

Feature set per reading (t >= 1, previous valid reading exists):
    lag1            (previous valid rate, kWh/day)
    lag2            (rate 2 readings ago)
    ema             (exp moving avg of past rates, alpha=0.3)
    run_mean        (running mean of past rates)
    run_std         (running std of past rates)
    peer_avg        (shrunk peer-average rate)
    sin_doy, cos_doy(seasonality of current reading)
    dt              (gap in days)
    log_dt          (log gap)
    history_len     (number of prior valid readings)
    tariff_code, meter_type, is_urban, region_in, phase,
    amper, section_code  (static categoricals)

Target: consumption (rate × dt) for the current reading — same as v5.

Two regressors are trained:
    (1) consumption-space:  predict cons directly
    (2) rate-space:         predict rate, multiply by dt at eval
The rate-space model usually generalizes better for long-gap meters.

Both are evaluated with the same segmented breakdown as v5_segmented_eval
so the comparison is apples-to-apples.

Usage:
    python lightgbm_baseline.py \
        --train meters_electricity_train.npz \
        --val   meters_electricity_val.npz \
        --test  meters_electricity_test.npz \
        --cards static_cardinalities_ramz.json
"""

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import lightgbm as lgb


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

CAT_FEATURES = STATIC_NAMES  # LightGBM native categorical


# ════════════════════════════════════════════════════════════
# Feature extraction
# ════════════════════════════════════════════════════════════
def extract_features(npz_path, cards, default_rate=4.6):
    """
    Build (X, y_cons, y_rate, meta) for all readings t>=1.

    meta carries gap, meter_vol, history_len, tariff for segmentation.
    """
    data = np.load(npz_path, allow_pickle=True)
    values = data["values"]
    masks  = data["masks"]
    static = data["static"]
    N = len(values)
    card_keys = list(cards.keys())
    static_pos = {name: card_keys.index(name) for name in STATIC_NAMES}

    X_rows = []
    y_cons = []
    y_rate = []
    meta_gap = []
    meta_vol = []
    meta_hist = []
    meta_tariff = []

    for i in range(N):
        v = values[i]
        if v.ndim != 2 or v.shape[0] < 2:
            continue
        m = masks[i].astype(bool)[:, 0]
        T = v.shape[0]
        rate = v[:, VAL_IDX_RATE]
        dt   = v[:, VAL_IDX_DT]
        peer = v[:, VAL_IDX_PEERAVG]
        sind = v[:, VAL_IDX_SINDOY]
        cosd = v[:, VAL_IDX_COSDOY]

        static_vals = {name: int(static[i, static_pos[name]])
                       for name in STATIC_NAMES}

        valid_rates_all = rate[m]
        meter_vol = float(np.std(valid_rates_all)) if valid_rates_all.size >= 2 else 0.0

        # Running stats over valid past readings
        past_rates = []
        ema = default_rate
        lag1 = default_rate
        lag2 = default_rate
        n_hist = 0

        for t in range(T):
            if not m[t]:
                continue
            if n_hist >= 1:
                # Emit a training sample for reading t
                run_mean = float(np.mean(past_rates)) if past_rates else default_rate
                run_std  = float(np.std(past_rates)) if len(past_rates) >= 2 else 0.0
                row = [
                    lag1, lag2, ema, run_mean, run_std,
                    float(peer[t]), float(sind[t]), float(cosd[t]),
                    float(dt[t]), float(np.log1p(dt[t])), float(n_hist),
                ] + [static_vals[name] for name in STATIC_NAMES]
                X_rows.append(row)
                y_cons.append(float(rate[t] * dt[t]))
                y_rate.append(float(rate[t]))
                meta_gap.append(float(dt[t]))
                meta_vol.append(meter_vol)
                meta_hist.append(n_hist)
                meta_tariff.append(static_vals["tariff_code"])

            # Update history with reading t
            past_rates.append(float(rate[t]))
            lag2 = lag1
            lag1 = float(rate[t])
            ema = 0.3 * float(rate[t]) + 0.7 * ema
            n_hist += 1

    X = np.asarray(X_rows, dtype=np.float64)
    meta = {
        "gap": np.asarray(meta_gap),
        "vol": np.asarray(meta_vol),
        "hist": np.asarray(meta_hist),
        "tariff": np.asarray(meta_tariff),
    }
    print(f"   {npz_path}: {len(X):,} samples, {X.shape[1]} features")
    return X, np.asarray(y_cons), np.asarray(y_rate), meta


# ════════════════════════════════════════════════════════════
# Segmented MAE report
# ════════════════════════════════════════════════════════════
def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def report_segmented(pred, target, meta, label,
                     gap_easy=70, vol_easy=3.0, hist_easy=5):
    print(f"\n-- {label}: segmented MAE --")
    gap, vol, hist = meta["gap"], meta["vol"], meta["hist"]

    def show(name, sel):
        n = int(sel.sum())
        if n == 0:
            print(f"    {name:<28} n=0")
            return
        print(f"    {name:<28} n={n:>8,} ({100*n/len(target):4.1f}%)  "
              f"MAE={mae(pred[sel], target[sel]):>7.1f}")

    core = (gap < gap_easy) & (vol < vol_easy) & (hist >= hist_easy)
    show("WELL-CONDITIONED core", core)
    show("HARD (rest)", ~core)
    show("ALL", np.ones_like(core, dtype=bool))


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="meters_electricity_train.npz")
    ap.add_argument("--val",   default="meters_electricity_val.npz")
    ap.add_argument("--test",  default="meters_electricity_test.npz")
    ap.add_argument("--cards", default="static_cardinalities_ramz.json")
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--n_estimators", type=int, default=2000)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    args = ap.parse_args()

    with open(args.cards) as f:
        cards = json.load(f, object_pairs_hook=OrderedDict)

    print("Extracting features ...")
    Xtr, ytr_cons, ytr_rate, mtr = extract_features(args.train, cards)
    Xva, yva_cons, yva_rate, mva = extract_features(args.val, cards)
    Xte, yte_cons, yte_rate, mte = extract_features(args.test, cards)

    cat_idx = [FEATURE_NAMES.index(c) for c in CAT_FEATURES]

    common = dict(
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=50,
        reg_lambda=1.0,
        n_jobs=-1,
        verbosity=-1,
    )

    # ── Model 1: consumption space ──
    print("\nTraining LightGBM (consumption space, L1 objective) ...")
    m_cons = lgb.LGBMRegressor(objective="l1", **common)
    m_cons.fit(
        Xtr, ytr_cons,
        eval_set=[(Xva, yva_cons)],
        eval_metric="l1",
        categorical_feature=cat_idx,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    pred_cons_te = m_cons.predict(Xte).clip(min=0)
    print(f"   best_iteration: {m_cons.best_iteration_}")

    # ── Model 2: rate space ──
    print("\nTraining LightGBM (rate space, L1 objective) ...")
    m_rate = lgb.LGBMRegressor(objective="l1", **common)
    m_rate.fit(
        Xtr, ytr_rate,
        eval_set=[(Xva, yva_rate)],
        eval_metric="l1",
        categorical_feature=cat_idx,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    dt_te = Xte[:, FEATURE_NAMES.index("dt")]
    pred_rate_te = (m_rate.predict(Xte).clip(min=0)) * dt_te
    print(f"   best_iteration: {m_rate.best_iteration_}")

    # ── Overall results ──
    print("\n" + "=" * 64)
    print("LIGHTGBM TEST RESULTS (consumption MAE)")
    print("=" * 64)
    for name, pred in [
        ("LightGBM (consumption-space)", pred_cons_te),
        ("LightGBM (rate-space × dt)",   pred_rate_te),
    ]:
        m = mae(pred, yte_cons)
        r = float(np.sqrt(np.mean((pred - yte_cons) ** 2)))
        print(f"  {name:<32} MAE={m:>7.1f}  RMSE={r:>7.1f}")

    # Pick the better one for segmentation
    better = (pred_cons_te if mae(pred_cons_te, yte_cons)
              <= mae(pred_rate_te, yte_cons) else pred_rate_te)
    better_label = ("consumption-space"
                    if better is pred_cons_te else "rate-space")
    report_segmented(better, yte_cons, mte,
                     f"LightGBM ({better_label})")

    # ── Feature importance ──
    print("\n-- LightGBM feature importance (consumption model, gain) --")
    imp = m_cons.booster_.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]
    for j in order:
        print(f"    {FEATURE_NAMES[j]:<14} {imp[j]:>12,.0f}")

    print("\n" + "=" * 64)
    print("COMPARISON CONTEXT")
    print("=" * 64)
    print("  v5 transformer test MAE (your run):  ~134")
    print(f"  LightGBM best test MAE:               {min(mae(pred_cons_te, yte_cons), mae(pred_rate_te, yte_cons)):.1f}")
    print("\n  Interpretation:")
    print("    - If LightGBM >> 134: transformer adds real value.")
    print("    - If LightGBM ~ 134: the floor is in the data/features, "
          "not the model class.")
    print("    - If LightGBM < 134: gradient boosting is the better tool "
          "here.")

    print("\nDone.")


if __name__ == "__main__":
    main()
