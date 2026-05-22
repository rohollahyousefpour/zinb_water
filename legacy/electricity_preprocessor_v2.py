"""
Electricity-Meter Preprocessor v5 — Fixes & Feature Additions
==============================================================

Changes vs v4:

  1. **Correct reservoir sampling.** The v4 reservoir had off-by-one
     and double-counting bugs; the resulting p99.5 was biased toward
     early chunks. Replaced with a single canonical implementation.

  2. **Physical-range filtering.** Train rows with implausible rates
     (negative, or > PHYSICAL_RATE_MAX kWh/day) are dropped from peer
     aggregates and replaced with NaN in per-meter sequences. Tunable
     via PHYSICAL_RATE_MAX. Default 2000 kWh/day catches meter rollover
     and data-entry errors while preserving real industrial loads.

  3. **Seasonality features.** Each reading now exposes
     sin(2π·doy/365) and cos(2π·doy/365) as additional values
     channels. The model can now distinguish July vs January readings.
     Total values channels: 5 (rate, dt, peer_avg, sin_doy, cos_doy)
     vs 3 in v4.

  4. **Coarser L1 aggregation (year-quarter).** v4 grouped
     `(section, tariff, year-month)` with 38.5% singleton buckets. v5
     uses `(section, tariff, year-quarter)` for L1, dropping singleton
     rate while keeping seasonal signal. L2 and L3 remain monthly.
     Configurable via L1_PERIOD.

  5. **Rare-tariff bucketing.** Tariff codes appearing < MIN_TARIFF_COUNT
     times in train are remapped to a single "tariff_other" code. This
     shrinks the tariff embedding table from ~113 to ~15-25 and forces
     similar rare tariffs to share representation.

  6. **Train-only outlier guard for percentile reservoir.** The
     reservoir now ignores values above PHYSICAL_RATE_MAX × max_dt
     (i.e., the same physical guard) so the p99.5 reflects realistic
     consumption.

Output schema:
  values: (T, 5) — [rate, dt, peer_avg, sin_doy, cos_doy]
  All other arrays (times, abs_dates, masks, static, ramz) unchanged.

Backwards compatibility:
  This breaks the model's `n_value_features` from 3 → 5. The
  electricity_zinb_patches.py InputEmbedding(n_value_features=...)
  must be updated to 5, and forward() must slice the correct
  peer_rate index (still index 2).
"""

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
MAX_PERIODS    = 59
DATE_FORMAT    = "%Y-%m-%d"
CSV_CHUNK_SIZE = 20_000
PARALLEL_JOBS  = 8
SUB_CHUNK_ROWS = 5_000

# Hierarchical aggregate config
K_SHRINK          = 2
MIN_BUCKET_FOR_L1 = 2
L1_PERIOD         = "Q"   # 'M' (monthly) or 'Q' (quarterly). 'Q' reduces singletons.
L2_PERIOD         = "M"
L3_PERIOD         = "M"

# Physical guard (kWh/day). Reads above this are treated as meter rollover
# or data-entry errors and excluded from training signal.
PHYSICAL_RATE_MAX = 2000.0

# Rare-tariff bucketing
MIN_TARIFF_COUNT = 200    # tariffs with < this many train rows are bucketed
RARE_TARIFF_CODE = "__OTHER__"

# Train/val/test split
SPLIT_SEED       = 42
FRAC_TRAIN       = 0.80
FRAC_VAL         = 0.10

# Percentile reservoir
PCTILE_SAMPLE_SIZE = 2_000_000
PERCENTILES        = [50, 90, 95, 99, 99.5, 99.9, 100]

OUTPUT_DIR    = Path(".")
OUTPUT_PREFIX = "meters_electricity"

STATIC_NUM_COLS = []
STATIC_CAT_COLS = [
    "meter_type", "tariff_code", "is_urban",
    "region_in", "phase", "amper", "section_code",
]


# ════════════════════════════════════════════════════════════
# COLUMN HELPERS
# ════════════════════════════════════════════════════════════
def gcols(prefix):
    return [f"{prefix}_g{str(i).zfill(2)}" for i in range(MAX_PERIODS)]


CURR_COLS  = gcols("curr_read_greg")
CONS_COLS  = gcols("consumption")
DAYS_COUNT = gcols("days_count")


# ════════════════════════════════════════════════════════════
# STATIC ENCODING
# ════════════════════════════════════════════════════════════
CAT_MAPS = {}
RARE_TARIFF_SET = set()  # populated in Pass 0


def remap_rare_tariffs(df):
    """Replace rare tariff codes (computed on train) with a single bucket."""
    if not RARE_TARIFF_SET:
        return df
    df = df.copy()
    df["tariff_code"] = df["tariff_code"].where(
        ~df["tariff_code"].isin(RARE_TARIFF_SET), RARE_TARIFF_CODE
    )
    return df


def encode_static(df):
    static_feats = []
    for col in STATIC_CAT_COLS:
        if col not in CAT_MAPS:
            CAT_MAPS[col] = {}
        for v in df[col].dropna().unique():
            if v not in CAT_MAPS[col]:
                CAT_MAPS[col][v] = len(CAT_MAPS[col])
        static_feats.append(
            df[col].map(CAT_MAPS[col]).fillna(0).astype(np.int64).values
        )
    for col in STATIC_NUM_COLS:
        static_feats.append(df[col].astype(np.float32).fillna(0.0).values)
    return np.column_stack(static_feats)


# ════════════════════════════════════════════════════════════
# SEASONALITY FEATURES
# ════════════════════════════════════════════════════════════
def compute_seasonality(dates_array):
    """
    Given an array of np.datetime64[D] (possibly with NaT), return
    (sin_doy, cos_doy) as float32 arrays of the same shape. NaT → 0.0.
    """
    out_sin = np.zeros(dates_array.shape, dtype=np.float32)
    out_cos = np.zeros(dates_array.shape, dtype=np.float32)
    valid = ~pd.isna(dates_array)
    if not valid.any():
        return out_sin, out_cos
    valid_dates = pd.to_datetime(dates_array[valid])
    doy = valid_dates.dayofyear.values.astype(np.float32)
    angle = 2.0 * np.pi * doy / 365.25
    out_sin[valid] = np.sin(angle).astype(np.float32)
    out_cos[valid] = np.cos(angle).astype(np.float32)
    return out_sin, out_cos


# ════════════════════════════════════════════════════════════
# RESERVOIR SAMPLING — CORRECT
# ════════════════════════════════════════════════════════════
class Reservoir:
    """
    Canonical reservoir sampler. Each call to update() processes a batch
    of new values and maintains uniform sampling over the full stream.
    """
    def __init__(self, capacity, seed=0):
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=np.float32)
        self.n_seen = 0
        self.rng = np.random.default_rng(seed)

    def update(self, new_values):
        for v in new_values:
            if self.n_seen < self.capacity:
                self.buffer[self.n_seen] = v
            else:
                j = self.rng.integers(0, self.n_seen + 1)
                if j < self.capacity:
                    self.buffer[j] = v
            self.n_seen += 1

    def samples(self):
        """Return the valid portion of the buffer."""
        return self.buffer[:min(self.n_seen, self.capacity)]

    def percentiles(self, pcts):
        s = self.samples()
        if s.size == 0:
            return {f"p{p}": 0.0 for p in pcts}
        vals = np.percentile(s, pcts)
        return {f"p{p}": float(v) for p, v in zip(pcts, vals)}


# ════════════════════════════════════════════════════════════
# PASS 0 — SPLIT + TARIFF FREQUENCY
# ════════════════════════════════════════════════════════════
def build_split_and_tariff_freq(csv_path, seed=SPLIT_SEED,
                                frac_train=FRAC_TRAIN, frac_val=FRAC_VAL,
                                min_tariff_count=MIN_TARIFF_COUNT):
    """
    Pass 0: build ramz-based split and compute train-only tariff frequencies.
    Returns the split dict and the set of rare tariff codes.
    """
    print("⏳ Pass 0a: collecting unique ramz codes ...")
    all_ramz = []
    for chunk in pd.read_csv(csv_path, chunksize=CSV_CHUNK_SIZE,
                             usecols=["ramz"], engine="c",
                             low_memory=False):
        all_ramz.append(chunk["ramz"].to_numpy())
    all_ramz = np.unique(np.concatenate(all_ramz))
    n = len(all_ramz)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(frac_train * n)
    n_va = int(frac_val * n)
    train_ramz = set(all_ramz[perm[:n_tr]].tolist())
    val_ramz   = set(all_ramz[perm[n_tr:n_tr + n_va]].tolist())
    test_ramz  = set(all_ramz[perm[n_tr + n_va:]].tolist())

    print(f"   total meters: {n:,}  "
          f"→ train={len(train_ramz):,}  "
          f"val={len(val_ramz):,}  "
          f"test={len(test_ramz):,}")

    print("⏳ Pass 0b: counting tariff frequencies on train rows ...")
    tariff_counts = Counter()
    for chunk in pd.read_csv(csv_path, chunksize=CSV_CHUNK_SIZE,
                             usecols=["ramz", "tariff_code"], engine="c",
                             low_memory=False):
        train_mask = chunk["ramz"].isin(train_ramz)
        if not train_mask.any():
            continue
        tariff_counts.update(chunk.loc[train_mask, "tariff_code"].dropna()
                             .tolist())

    rare_tariffs = {t for t, c in tariff_counts.items() if c < min_tariff_count}
    common_tariffs = {t: c for t, c in tariff_counts.items()
                      if c >= min_tariff_count}
    print(f"   total distinct tariffs: {len(tariff_counts):,}")
    print(f"   common tariffs (≥{min_tariff_count}): {len(common_tariffs):,}")
    print(f"   rare tariffs (bucketed → '{RARE_TARIFF_CODE}'): "
          f"{len(rare_tariffs):,}")

    return ({"train_ramz": train_ramz,
             "val_ramz":   val_ramz,
             "test_ramz":  test_ramz},
            rare_tariffs)


# ════════════════════════════════════════════════════════════
# PASS 1 — AGGREGATES + TARGET PERCENTILES (TRAIN ONLY)
# ════════════════════════════════════════════════════════════
def compute_hierarchical_aggregates(csv_path, train_ramz_set,
                                    chunk_size=10_000,
                                    pctile_sample_size=PCTILE_SAMPLE_SIZE,
                                    physical_rate_max=PHYSICAL_RATE_MAX):
    """
    Returns: L1, L2, L3 dicts, global_mean rate, target_percentiles dict,
    diagnostic counts.

    L1 keys are (section_code, tariff_code, L1_period).
    L2 keys are (tariff_code, L2_period).
    L3 keys are (L3_period,).

    All aggregates are built from train rows only. Rates outside
    [0, physical_rate_max] are excluded.
    """
    print("⏳ Pass 1: hierarchical aggregates from TRAIN rows only")
    print(f"   L1 period: {L1_PERIOD}, L2 period: {L2_PERIOD}, "
          f"L3 period: {L3_PERIOD}")
    print(f"   K_SHRINK: {K_SHRINK}, "
          f"physical_rate_max: {physical_rate_max} kWh/day")

    L1, L2, L3 = {}, {}, {}
    global_sum, global_n = 0.0, 0

    reservoir = Reservoir(pctile_sample_size, seed=SPLIT_SEED)

    # Counters for diagnostics
    n_total = 0
    n_dropped_negative = 0
    n_dropped_extreme = 0

    needed = ["ramz", "section_code", "tariff_code"] + \
             CURR_COLS + CONS_COLS + DAYS_COUNT

    reader = pd.read_csv(csv_path, chunksize=chunk_size,
                         usecols=needed, engine="c", low_memory=False)

    for ci, chunk in enumerate(reader):
        in_train = chunk["ramz"].isin(train_ramz_set)
        if not in_train.any():
            continue
        chunk = chunk.loc[in_train].reset_index(drop=True)
        chunk = remap_rare_tariffs(chunk)

        dt = chunk[DAYS_COUNT].values
        cons = chunk[CONS_COLS].values

        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(dt > 0, cons / dt, np.nan).astype(np.float32)

        # Diagnostic: count outlier rates before masking
        valid_rate_mask = ~np.isnan(rate)
        n_total += int(valid_rate_mask.sum())
        n_dropped_negative += int(((rate < 0) & valid_rate_mask).sum())
        n_dropped_extreme += int(((rate > physical_rate_max) &
                                  valid_rate_mask).sum())

        # Apply physical guard: NaN-out implausible rates
        rate = np.where((rate < 0) | (rate > physical_rate_max), np.nan, rate)

        # Reservoir update: total consumption per reading (cons),
        # excluding rows where rate was masked
        cons_for_reservoir = np.where(
            (dt > 0) & ~np.isnan(rate) & ~np.isnan(cons) & (cons >= 0),
            cons, np.nan
        )
        valid_cons = cons_for_reservoir[~np.isnan(cons_for_reservoir)]
        reservoir.update(valid_cons)

        # ── Build rate aggregates ──
        rate_df = pd.DataFrame(rate, columns=CONS_COLS, index=chunk.index)
        rate_df["section_code"] = chunk["section_code"]
        rate_df["tariff_code"]  = chunk["tariff_code"]

        m_dates = chunk.melt(id_vars=["section_code", "tariff_code"],
                             value_vars=CURR_COLS, value_name="date")
        m_rates = rate_df.melt(id_vars=["section_code", "tariff_code"],
                               value_vars=CONS_COLS, value_name="rate")

        tmp = m_dates[["section_code", "tariff_code", "date"]].copy()
        tmp["rate"] = m_rates["rate"]
        tmp = tmp.dropna(subset=["date", "rate"])
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce",
                                     format=DATE_FORMAT)
        tmp = tmp.dropna(subset=["date"])

        # Period keys for each level
        tmp["l1_period"] = tmp["date"].dt.to_period(L1_PERIOD)
        tmp["l2_period"] = tmp["date"].dt.to_period(L2_PERIOD)
        tmp["l3_period"] = tmp["date"].dt.to_period(L3_PERIOD)

        global_sum += float(tmp["rate"].sum())
        global_n   += int(len(tmp))

        g1 = tmp.groupby(["section_code", "tariff_code", "l1_period"])["rate"]\
                .agg(["sum", "count"])
        for idx, row in g1.iterrows():
            d = L1.setdefault(idx, {"s": 0.0, "n": 0})
            d["s"] += float(row["sum"]); d["n"] += int(row["count"])

        g2 = tmp.groupby(["tariff_code", "l2_period"])["rate"]\
                .agg(["sum", "count"])
        for idx, row in g2.iterrows():
            d = L2.setdefault(idx, {"s": 0.0, "n": 0})
            d["s"] += float(row["sum"]); d["n"] += int(row["count"])

        g3 = tmp.groupby(["l3_period"])["rate"]\
                .agg(["sum", "count"])
        for idx, row in g3.iterrows():
            d = L3.setdefault(idx, {"s": 0.0, "n": 0})
            d["s"] += float(row["sum"]); d["n"] += int(row["count"])

        if ci % 10 == 0:
            print(f"   ... chunk {ci}  |L1|={len(L1):,}  "
                  f"|L2|={len(L2):,}  |L3|={len(L3):,}")

    global_mean = (global_sum / global_n) if global_n > 0 else 0.0
    target_percentiles = reservoir.percentiles(PERCENTILES)

    diagnostics = {
        "n_train_rate_total":     n_total,
        "n_dropped_negative":     n_dropped_negative,
        "n_dropped_extreme_rate": n_dropped_extreme,
        "fraction_dropped":       (n_dropped_negative + n_dropped_extreme)
                                  / max(n_total, 1),
        "reservoir_n_seen":       reservoir.n_seen,
        "reservoir_filled":       min(reservoir.n_seen, reservoir.capacity),
    }

    print(f"✅ Pass 1 complete.")
    print(f"   buckets: L1={len(L1):,}, L2={len(L2):,}, L3={len(L3):,}")
    print(f"   global_mean_rate: {global_mean:.3f}")
    print(f"   train rate samples: {n_total:,} valid")
    print(f"   dropped (negative): {n_dropped_negative:,} "
          f"({100*n_dropped_negative/max(n_total,1):.3f}%)")
    print(f"   dropped (> {physical_rate_max} kWh/day): {n_dropped_extreme:,} "
          f"({100*n_dropped_extreme/max(n_total,1):.3f}%)")
    print(f"   reservoir: {diagnostics['reservoir_filled']:,} samples "
          f"from {reservoir.n_seen:,} total")
    print("   Train target percentiles (total consumption per reading):")
    for k, v in target_percentiles.items():
        print(f"     {k:>6s} = {v:>10.1f} kWh")

    return L1, L2, L3, global_mean, target_percentiles, diagnostics


# ════════════════════════════════════════════════════════════
# PEER-AVG LOOKUP
# ════════════════════════════════════════════════════════════
def peer_avg_shrunk(y_i, sec, tar, l1_per, l2_per, l3_per,
                    L1, L2, L3, global_mean, k=K_SHRINK, is_train=False):
    """
    Empirical-Bayes shrunk peer average. For train rows, performs
    leave-one-out subtraction at each level so the meter's own value
    doesn't contribute to its peer estimate.
    """
    d1 = L1.get((sec, tar, l1_per))
    if d1 is not None:
        n1 = d1["n"] - 1 if is_train else d1["n"]
        if n1 >= max(MIN_BUCKET_FOR_L1 - 1, 1):
            s1 = d1["s"] - y_i if is_train else d1["s"]
            mu1 = s1 / n1
        else:
            n1, mu1 = 0, None
    else:
        n1, mu1 = 0, None

    d2 = L2.get((tar, l2_per))
    if d2 is not None:
        n2 = d2["n"] - 1 if is_train else d2["n"]
        if n2 >= 1:
            s2 = d2["s"] - y_i if is_train else d2["s"]
            mu2 = s2 / n2
        else:
            n2, mu2 = 0, None
    else:
        n2, mu2 = 0, None

    d3 = L3.get(l3_per)
    if d3 is not None:
        n3 = d3["n"] - 1 if is_train else d3["n"]
        if n3 >= 1:
            s3 = d3["s"] - y_i if is_train else d3["s"]
            mu3 = s3 / n3
        else:
            mu3 = global_mean
    else:
        mu3 = global_mean

    if mu2 is None:
        prior = mu3
    else:
        w2 = n2 / (n2 + k)
        prior = w2 * mu2 + (1 - w2) * mu3

    if mu1 is None:
        return float(prior)
    w1 = n1 / (n1 + k)
    return float(w1 * mu1 + (1 - w1) * prior)


# ════════════════════════════════════════════════════════════
# PASS 2 — BUILD PER-METER SEQUENCES
# ════════════════════════════════════════════════════════════
VAL_RAMZ_SET = set()


def process_subchunk(df, static_feats, L1, L2, L3, global_mean,
                     train_ramz_set, physical_rate_max):
    curr_raw = df[CURR_COLS].to_numpy(dtype=object)
    cons     = df[CONS_COLS].to_numpy(dtype=np.float32)
    dt_raw   = df[DAYS_COUNT].to_numpy(dtype=np.float32)
    ramz     = df["ramz"].to_numpy(dtype=np.int64)
    sec      = df["section_code"].to_numpy()
    tar      = df["tariff_code"].to_numpy()

    flat = pd.to_datetime(curr_raw.ravel(), errors="coerce",
                          format=DATE_FORMAT)
    flat_dates = flat.to_numpy()
    flat_l1    = flat.to_period(L1_PERIOD).to_numpy()
    flat_l2    = flat.to_period(L2_PERIOD).to_numpy()
    flat_l3    = flat.to_period(L3_PERIOD).to_numpy()
    curr_dates = flat_dates.reshape(curr_raw.shape)
    curr_l1    = flat_l1.reshape(curr_raw.shape)
    curr_l2    = flat_l2.reshape(curr_raw.shape)
    curr_l3    = flat_l3.reshape(curr_raw.shape)
    abs_ts     = (curr_dates - np.datetime64("1970-01-01T00:00:00")) \
                 / np.timedelta64(1, "s")

    # Seasonality features (sin/cos of day-of-year)
    sin_doy, cos_doy = compute_seasonality(curr_dates)

    valid = ~pd.isna(curr_dates)
    keep  = valid.any(axis=1)
    if not keep.any():
        return [], [], [], [], [], [], []

    curr_dates = curr_dates[keep]
    curr_l1 = curr_l1[keep]; curr_l2 = curr_l2[keep]; curr_l3 = curr_l3[keep]
    sin_doy = sin_doy[keep]; cos_doy = cos_doy[keep]
    cons = cons[keep]; dt_raw = dt_raw[keep]
    valid = valid[keep]; static_feats = static_feats[keep]
    ramz = ramz[keep]; sec = sec[keep]; tar = tar[keep]
    abs_ts = abs_ts[keep]

    N = curr_dates.shape[0]
    first_idx = np.argmax(valid, axis=1)
    t0 = curr_dates[np.arange(N), first_idx][:, None]
    times = (curr_dates - t0).astype("timedelta64[D]").astype(np.float32)
    times[~valid] = np.nan

    mask    = (~np.isnan(cons)).astype(np.float32)
    dt_safe = np.where(dt_raw <= 0, np.nan, dt_raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = cons / dt_safe

    # Apply physical guard to rate (and zero out mask for extreme reads)
    extreme = (rate > physical_rate_max) | (rate < 0)
    rate = np.where(extreme, 0.0, rate)
    mask = np.where(extreme, 0.0, mask)

    rate    = np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)
    dt_cln  = np.nan_to_num(dt_safe, nan=0.0, posinf=0.0, neginf=0.0)

    times_r, abs_r, values_r, masks_r, static_r, ramz_r, split_r = \
        [], [], [], [], [], [], []

    for i in range(N):
        idx_keep = ~np.isnan(times[i])
        v_rate   = rate[i][idx_keep]
        v_dt     = dt_cln[i][idx_keep]
        v_l1     = curr_l1[i][idx_keep]
        v_l2     = curr_l2[i][idx_keep]
        v_l3     = curr_l3[i][idx_keep]
        v_sin    = sin_doy[i][idx_keep]
        v_cos    = cos_doy[i][idx_keep]

        is_train = int(ramz[i]) in train_ramz_set

        p_avgs = np.fromiter(
            (peer_avg_shrunk(float(r), sec[i], tar[i],
                             l1p, l2p, l3p,
                             L1, L2, L3, global_mean,
                             k=K_SHRINK, is_train=is_train)
             for r, l1p, l2p, l3p in zip(v_rate, v_l1, v_l2, v_l3)),
            dtype=np.float32, count=len(v_rate)
        )

        # values shape: (T, 5) — rate, dt, peer_avg, sin_doy, cos_doy
        values_i = np.column_stack([v_rate, v_dt, p_avgs, v_sin, v_cos])\
                     .astype(np.float32)

        times_r.append(times[i][idx_keep])
        values_r.append(values_i)
        masks_r.append(mask[i][idx_keep][:, None])
        static_r.append(static_feats[i])
        ramz_r.append(ramz[i])
        abs_r.append(abs_ts[i][idx_keep].astype(np.float64))
        split_r.append("train" if is_train else
                       ("val" if int(ramz[i]) in VAL_RAMZ_SET else "test"))

    return times_r, abs_r, values_r, masks_r, static_r, ramz_r, split_r


def process_chunk_parallel(df_chunk, L1, L2, L3, global_mean,
                           train_ramz_set, physical_rate_max):
    df_chunk = remap_rare_tariffs(df_chunk)
    static_feats = encode_static(df_chunk)
    sub_chunks = [
        (df_chunk.iloc[i:i + SUB_CHUNK_ROWS].reset_index(drop=True),
         static_feats[i:i + SUB_CHUNK_ROWS])
        for i in range(0, len(df_chunk), SUB_CHUNK_ROWS)
    ]
    results = Parallel(n_jobs=PARALLEL_JOBS)(
        delayed(process_subchunk)(df, st, L1, L2, L3, global_mean,
                                  train_ramz_set, physical_rate_max)
        for df, st in sub_chunks
    )
    out = [[] for _ in range(7)]
    for tup in results:
        for j, lst in enumerate(tup):
            out[j].extend(lst)
    return tuple(out)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def preprocess(csv_path,
               output_dir=OUTPUT_DIR, output_prefix=OUTPUT_PREFIX,
               seed=SPLIT_SEED,
               frac_train=FRAC_TRAIN, frac_val=FRAC_VAL,
               physical_rate_max=PHYSICAL_RATE_MAX,
               min_tariff_count=MIN_TARIFF_COUNT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 0: split + tariff frequencies
    split, rare_tariffs = build_split_and_tariff_freq(
        csv_path, seed=seed,
        frac_train=frac_train, frac_val=frac_val,
        min_tariff_count=min_tariff_count,
    )
    train_ramz_set = split["train_ramz"]
    global VAL_RAMZ_SET, RARE_TARIFF_SET
    VAL_RAMZ_SET = split["val_ramz"]
    RARE_TARIFF_SET = rare_tariffs

    # Pass 1: aggregates + percentiles (with rare-tariff bucketing applied)
    L1, L2, L3, global_mean, target_pctiles, diag = \
        compute_hierarchical_aggregates(
            csv_path, train_ramz_set,
            physical_rate_max=physical_rate_max,
        )

    n1_sizes = np.array([d["n"] for d in L1.values()])
    print(f"   L1 buckets with n=1: {(n1_sizes == 1).sum():,} "
          f"({100*(n1_sizes == 1).mean():.1f}%)")
    print(f"   L1 buckets with n<5: {(n1_sizes < 5).sum():,} "
          f"({100*(n1_sizes < 5).mean():.1f}%)\n")

    # Pass 2: build per-meter sequences
    times_a, abs_a, values_a, masks_a, static_a, ramz_a, split_a = \
        [], [], [], [], [], [], []
    total = 0
    reader = pd.read_csv(csv_path, chunksize=CSV_CHUNK_SIZE, low_memory=False)
    for ci, df_chunk in enumerate(reader, start=1):
        t, a, v, m, s, r, sp = process_chunk_parallel(
            df_chunk, L1, L2, L3, global_mean,
            train_ramz_set, physical_rate_max,
        )
        times_a.extend(t); abs_a.extend(a); values_a.extend(v)
        masks_a.extend(m); static_a.extend(s); ramz_a.extend(r)
        split_a.extend(sp)
        total += len(t)
        print(f"✅ Chunk {ci}: meters processed = {total:,}")

    split_a  = np.array(split_a)
    static_a = np.array(static_a, dtype=np.int64)
    ramz_a   = np.array(ramz_a, dtype=np.int64)
    times_a  = np.array(times_a,  dtype=object)
    abs_a    = np.array(abs_a,    dtype=object)
    values_a = np.array(values_a, dtype=object)
    masks_a  = np.array(masks_a,  dtype=object)

    out_paths = {}
    for name in ("train", "val", "test"):
        sel = split_a == name
        out = output_dir / f"{output_prefix}_{name}.npz"
        np.savez_compressed(
            out,
            times    = times_a[sel],
            abs_dates= abs_a[sel],
            values   = values_a[sel],
            masks    = masks_a[sel],
            static   = static_a[sel],
            ramz     = ramz_a[sel],
        )
        out_paths[name] = out
        print(f"💾 {name:5s}: {int(sel.sum()):,} meters → {out}")

    # Persist config
    with open(output_dir / "static_cardinalities_ramz.json", "w") as f:
        json.dump({k: len(v) for k, v in CAT_MAPS.items()}, f, indent=2)

    with open(output_dir / "peer_avg_config.json", "w") as f:
        json.dump({
            "version": 5,
            "k_shrink": K_SHRINK,
            "min_bucket_for_l1": MIN_BUCKET_FOR_L1,
            "l1_period": L1_PERIOD,
            "l2_period": L2_PERIOD,
            "l3_period": L3_PERIOD,
            "n_l1_buckets": len(L1),
            "n_l2_buckets": len(L2),
            "n_l3_buckets": len(L3),
            "global_mean_rate": global_mean,
            "physical_rate_max": physical_rate_max,
            "min_tariff_count": min_tariff_count,
            "n_rare_tariffs_bucketed": len(rare_tariffs),
            "rare_tariff_label": RARE_TARIFF_CODE,
            "split_seed": seed,
            "frac_train": frac_train,
            "frac_val":   frac_val,
            "n_train_meters": len(train_ramz_set),
            "n_val_meters":   len(VAL_RAMZ_SET),
            "n_test_meters":  len(split["test_ramz"]),
            "train_target_percentiles": target_pctiles,
            "recommended_train_target_clip": target_pctiles["p99.5"],
            "n_value_features": 5,
            "value_feature_names": ["rate", "dt", "peer_avg",
                                    "sin_doy", "cos_doy"],
            "preprocessing_diagnostics": diag,
        }, f, indent=2)

    with open(output_dir / "peer_avg_aggregates.pkl", "wb") as f:
        pickle.dump({
            "L1": L1, "L2": L2, "L3": L3,
            "global_mean": global_mean,
            "k_shrink": K_SHRINK,
            "l1_period": L1_PERIOD,
            "l2_period": L2_PERIOD,
            "l3_period": L3_PERIOD,
            "physical_rate_max": physical_rate_max,
            "rare_tariffs": rare_tariffs,
            "train_target_percentiles": target_pctiles,
        }, f)

    print("\n✅ PREPROCESSING COMPLETE (v5)")
    print(f"   K_SHRINK               : {K_SHRINK}")
    print(f"   L1 period              : {L1_PERIOD}")
    print(f"   physical_rate_max      : {physical_rate_max} kWh/day")
    print(f"   min_tariff_count       : {min_tariff_count}")
    print(f"   rare tariffs bucketed  : {len(rare_tariffs):,}")
    print(f"   global_mean            : {global_mean:.3f}")
    print(f"   p99.5 clip             : {target_pctiles['p99.5']:.0f} kWh")
    print(f"   n_value_features       : 5 (rate, dt, peer_avg, "
          f"sin_doy, cos_doy)")
    print(f"   outputs                : {[str(p) for p in out_paths.values()]}")


if __name__ == "__main__":
    preprocess(r"E:\data\dastkari\data_train_wide.csv")
