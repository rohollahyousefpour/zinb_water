"""
Electricity-Meter Preprocessor v5b — Adds Per-Tariff GLM Coefficients
=====================================================================

Identical to v5 except adds Pass 3: fit a per-tariff linear regression
that predicts daily_rate from a small set of meter-level features
(lag1, ema-of-rate, peer_avg, sin_doy, cos_doy, constant). Coefficients
are saved to `peer_avg_aggregates.pkl` under key 'glm_by_tariff'.

The model loads these coefficients at init and uses them to compute
a much better `base_rate` than the current `wl*lag1 + we*ema` blend.
The transformer's correction head then has less work to do, leading
to faster convergence and lower asymptotic MAE.

Output NPZ schema is UNCHANGED from v5 (5-channel values). Only the
pickle file gains a new entry.

Output schema:
  values: (T, 5) — [rate, dt, peer_avg, sin_doy, cos_doy]   ← unchanged
  peer_avg_aggregates.pkl:
    'L1', 'L2', 'L3', 'global_mean', ...   ← unchanged
    'glm_by_tariff': {tariff_code: np.ndarray(6,)}  ← NEW
    'glm_feature_names': [...]              ← NEW
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# ════════════════════════════════════════════════════════════
# CONFIG (mirrors v5)
# ════════════════════════════════════════════════════════════
MAX_PERIODS    = 59
DATE_FORMAT    = "%Y-%m-%d"
CSV_CHUNK_SIZE = 20_000
PARALLEL_JOBS  = 8
SUB_CHUNK_ROWS = 5_000

K_SHRINK          = 2
MIN_BUCKET_FOR_L1 = 2
L1_PERIOD         = "Q"
L2_PERIOD         = "M"
L3_PERIOD         = "M"

PHYSICAL_RATE_MAX = 2000.0

MIN_TARIFF_COUNT = 200
RARE_TARIFF_CODE = "__OTHER__"

SPLIT_SEED = 42
FRAC_TRAIN = 0.80
FRAC_VAL   = 0.10

PCTILE_SAMPLE_SIZE = 2_000_000
PERCENTILES        = [50, 90, 95, 99, 99.5, 99.9, 100]

# GLM fitting config
GLM_RIDGE_LAMBDA  = 1.0     # L2 regularization for stability
GLM_MIN_SAMPLES   = 500     # tariffs below this fall back to global GLM
GLM_RATE_CLIP     = 500.0   # clip rate (target) to this max for fitting
GLM_FEATURE_NAMES = ["const", "lag1", "ema", "peer_avg",
                     "sin_doy", "cos_doy"]
GLM_N_FEATURES    = len(GLM_FEATURE_NAMES)

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
RARE_TARIFF_SET = set()


def remap_rare_tariffs(df):
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
# SEASONALITY
# ════════════════════════════════════════════════════════════
def compute_seasonality(dates_array):
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
# RESERVOIR
# ════════════════════════════════════════════════════════════
class Reservoir:
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
        return self.buffer[:min(self.n_seen, self.capacity)]

    def percentiles(self, pcts):
        s = self.samples()
        if s.size == 0:
            return {f"p{p}": 0.0 for p in pcts}
        vals = np.percentile(s, pcts)
        return {f"p{p}": float(v) for p, v in zip(pcts, vals)}


# ════════════════════════════════════════════════════════════
# PASS 0 — SPLIT + TARIFF FREQUENCIES (unchanged from v5)
# ════════════════════════════════════════════════════════════
def build_split_and_tariff_freq(csv_path, seed=SPLIT_SEED,
                                frac_train=FRAC_TRAIN, frac_val=FRAC_VAL,
                                min_tariff_count=MIN_TARIFF_COUNT):
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
# PASS 1 — AGGREGATES + PERCENTILES (unchanged from v5)
# ════════════════════════════════════════════════════════════
def compute_hierarchical_aggregates(csv_path, train_ramz_set,
                                    chunk_size=10_000,
                                    pctile_sample_size=PCTILE_SAMPLE_SIZE,
                                    physical_rate_max=PHYSICAL_RATE_MAX):
    print("⏳ Pass 1: hierarchical aggregates from TRAIN rows only")
    print(f"   L1 period: {L1_PERIOD}, L2 period: {L2_PERIOD}, "
          f"L3 period: {L3_PERIOD}")

    L1, L2, L3 = {}, {}, {}
    global_sum, global_n = 0.0, 0
    reservoir = Reservoir(pctile_sample_size, seed=SPLIT_SEED)
    n_total = 0; n_dropped_negative = 0; n_dropped_extreme = 0

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

        valid_rate_mask = ~np.isnan(rate)
        n_total += int(valid_rate_mask.sum())
        n_dropped_negative += int(((rate < 0) & valid_rate_mask).sum())
        n_dropped_extreme += int(((rate > physical_rate_max) &
                                  valid_rate_mask).sum())

        rate = np.where((rate < 0) | (rate > physical_rate_max), np.nan, rate)

        cons_for_reservoir = np.where(
            (dt > 0) & ~np.isnan(rate) & ~np.isnan(cons) & (cons >= 0),
            cons, np.nan
        )
        valid_cons = cons_for_reservoir[~np.isnan(cons_for_reservoir)]
        reservoir.update(valid_cons)

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
    }

    print(f"✅ Pass 1 complete. global_mean={global_mean:.3f}, "
          f"dropped_extreme={n_dropped_extreme:,}")
    return L1, L2, L3, global_mean, target_percentiles, diagnostics


# ════════════════════════════════════════════════════════════
# PEER-AVG LOOKUP (unchanged from v5)
# ════════════════════════════════════════════════════════════
def peer_avg_shrunk(y_i, sec, tar, l1_per, l2_per, l3_per,
                    L1, L2, L3, global_mean, k=K_SHRINK, is_train=False):
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
# PASS 1.5 — FIT PER-TARIFF GLM ON TRAIN ROWS
# ════════════════════════════════════════════════════════════
def fit_per_tariff_glm(csv_path, train_ramz_set, L1, L2, L3, global_mean,
                       chunk_size=10_000,
                       physical_rate_max=PHYSICAL_RATE_MAX,
                       ridge_lambda=GLM_RIDGE_LAMBDA,
                       min_samples=GLM_MIN_SAMPLES,
                       rate_clip=GLM_RATE_CLIP):
    """
    Fit per-tariff ridge regression: rate_t ~ const + lag1 + ema + peer_avg
                                              + sin_doy + cos_doy

    Returns:
      glm_by_tariff: dict tariff_code -> np.ndarray(6,) of coefficients
      global_glm:    np.ndarray(6,) — fallback for unseen / rare tariffs

    Implementation: accumulate X^T X and X^T y per tariff across all
    chunks (closed-form ridge), then solve once at the end.
    """
    print("⏳ Pass 1.5: fitting per-tariff GLM baselines ...")
    K = GLM_N_FEATURES

    # Accumulators
    XtX_by_tariff = defaultdict(lambda: np.zeros((K, K), dtype=np.float64))
    Xty_by_tariff = defaultdict(lambda: np.zeros(K, dtype=np.float64))
    n_by_tariff   = defaultdict(int)

    global_XtX = np.zeros((K, K), dtype=np.float64)
    global_Xty = np.zeros(K, dtype=np.float64)
    global_n = 0

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
        N = len(chunk)

        # Parse dates row-wise
        curr_raw = chunk[CURR_COLS].to_numpy(dtype=object)
        dt_raw = chunk[DAYS_COUNT].to_numpy(dtype=np.float32)
        cons   = chunk[CONS_COLS].to_numpy(dtype=np.float32)
        sec    = chunk["section_code"].to_numpy()
        tar    = chunk["tariff_code"].to_numpy()

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

        sin_doy, cos_doy = compute_seasonality(curr_dates)

        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(dt_raw > 0, cons / dt_raw, np.nan).astype(np.float32)
        rate = np.where((rate < 0) | (rate > physical_rate_max), np.nan, rate)

        # For each row, build training samples (t >= 1: need lag1, ema)
        # ema with alpha=0.3, reset per meter
        ema = np.zeros_like(rate)
        prev_rate = np.zeros(N, dtype=np.float32)
        prev_valid = np.zeros(N, dtype=bool)

        for t in range(curr_dates.shape[1]):
            r_t  = rate[:, t]
            v_t  = ~np.isnan(r_t)
            sin_t = sin_doy[:, t]
            cos_t = cos_doy[:, t]

            # EMA update (carries forward through invalid)
            if t == 0:
                ema[:, 0] = np.where(v_t, r_t, 0.0)
            else:
                ema_prev = ema[:, t - 1]
                ema[:, t] = np.where(v_t,
                                     0.3 * r_t + 0.7 * ema_prev,
                                     ema_prev)

            # Only use rows where t >= 1 (need lag) and current is valid
            if t == 0:
                prev_rate = np.where(v_t, r_t, prev_rate)
                prev_valid = v_t | prev_valid
                continue

            use = v_t & prev_valid
            if not use.any():
                # Update prev for next iteration
                prev_rate = np.where(v_t, r_t, prev_rate)
                prev_valid = v_t | prev_valid
                continue

            # Build peer_avg for these positions
            idx_use = np.where(use)[0]
            peer = np.zeros(len(idx_use), dtype=np.float32)
            for k, i in enumerate(idx_use):
                peer[k] = peer_avg_shrunk(
                    float(r_t[i]), sec[i], tar[i],
                    curr_l1[i, t], curr_l2[i, t], curr_l3[i, t],
                    L1, L2, L3, global_mean, k=K_SHRINK, is_train=True,
                )

            # Build feature matrix for the use rows
            n_use = len(idx_use)
            X = np.zeros((n_use, K), dtype=np.float64)
            X[:, 0] = 1.0                                        # const
            X[:, 1] = np.clip(prev_rate[idx_use], 0, rate_clip)  # lag1
            X[:, 2] = np.clip(ema[idx_use, t - 1], 0, rate_clip) # ema (prev)
            X[:, 3] = np.clip(peer, 0, rate_clip)                # peer_avg
            X[:, 4] = sin_t[idx_use]                             # sin_doy
            X[:, 5] = cos_t[idx_use]                             # cos_doy
            y = np.clip(r_t[idx_use], 0, rate_clip).astype(np.float64)

            # Accumulate per-tariff
            # Use pandas.unique to handle mixed int/string tariff codes
            # (post rare-bucketing the array has both original codes and
            # '__OTHER__'). np.unique would crash trying to sort them.
            tariffs = tar[idx_use]
            for t_code in pd.unique(tariffs):
                m = (tariffs == t_code)
                Xm = X[m]; ym = y[m]
                XtX_by_tariff[t_code] += Xm.T @ Xm
                Xty_by_tariff[t_code] += Xm.T @ ym
                n_by_tariff[t_code] += int(m.sum())

            # Global accumulator
            global_XtX += X.T @ X
            global_Xty += X.T @ y
            global_n   += n_use

            # Update prev for next iteration
            prev_rate = np.where(v_t, r_t, prev_rate)
            prev_valid = v_t | prev_valid

        if ci % 10 == 0:
            print(f"   ... chunk {ci}  |tariffs|={len(n_by_tariff):,}  "
                  f"global_n={global_n:,}")

    # Solve ridge regression for each tariff
    print(f"   Solving GLM systems (ridge λ={ridge_lambda}, "
          f"min_samples={min_samples}) ...")
    I = np.eye(K) * ridge_lambda
    I[0, 0] = 0.0  # don't regularize the intercept

    # Global GLM (fallback)
    global_glm = np.linalg.solve(global_XtX + I, global_Xty)

    glm_by_tariff = {}
    for t_code, n in n_by_tariff.items():
        if n < min_samples:
            glm_by_tariff[t_code] = global_glm.copy()
            continue
        try:
            coef = np.linalg.solve(
                XtX_by_tariff[t_code] + I,
                Xty_by_tariff[t_code],
            )
        except np.linalg.LinAlgError:
            coef = global_glm.copy()
        glm_by_tariff[t_code] = coef

    # Print coefficient summary
    print(f"   Global GLM coefficients:")
    for name, c in zip(GLM_FEATURE_NAMES, global_glm):
        print(f"     {name:>10s} = {c:>+8.3f}")
    print(f"   Per-tariff GLMs fitted: "
          f"{len(glm_by_tariff)} (n>={min_samples}: "
          f"{sum(1 for t,n in n_by_tariff.items() if n >= min_samples)})")
    for t_code in sorted(glm_by_tariff.keys(),
                         key=lambda x: -n_by_tariff[x])[:5]:
        n = n_by_tariff[t_code]
        coefs = glm_by_tariff[t_code]
        print(f"     tariff={str(t_code)[:8]:>8s}  n={n:>8,}  "
              f"const={coefs[0]:+.2f}  lag1={coefs[1]:+.2f}  "
              f"ema={coefs[2]:+.2f}  peer={coefs[3]:+.2f}  "
              f"sin={coefs[4]:+.2f}  cos={coefs[5]:+.2f}")

    return glm_by_tariff, global_glm, dict(n_by_tariff)


# ════════════════════════════════════════════════════════════
# PASS 2 — BUILD PER-METER SEQUENCES (unchanged from v5)
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
    extreme = (rate > physical_rate_max) | (rate < 0)
    rate = np.where(extreme, 0.0, rate)
    mask = np.where(extreme, 0.0, mask)
    rate    = np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)
    dt_cln  = np.nan_to_num(dt_safe, nan=0.0, posinf=0.0, neginf=0.0)

    times_r, abs_r, values_r, masks_r, static_r, ramz_r, split_r = \
        [], [], [], [], [], [], []

    for i in range(N):
        idx_keep = ~np.isnan(times[i])
        v_rate = rate[i][idx_keep]
        v_dt   = dt_cln[i][idx_keep]
        v_l1   = curr_l1[i][idx_keep]
        v_l2   = curr_l2[i][idx_keep]
        v_l3   = curr_l3[i][idx_keep]
        v_sin  = sin_doy[i][idx_keep]
        v_cos  = cos_doy[i][idx_keep]

        is_train = int(ramz[i]) in train_ramz_set

        p_avgs = np.fromiter(
            (peer_avg_shrunk(float(r), sec[i], tar[i], l1p, l2p, l3p,
                             L1, L2, L3, global_mean,
                             k=K_SHRINK, is_train=is_train)
             for r, l1p, l2p, l3p in zip(v_rate, v_l1, v_l2, v_l3)),
            dtype=np.float32, count=len(v_rate)
        )

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

    split, rare_tariffs = build_split_and_tariff_freq(
        csv_path, seed=seed,
        frac_train=frac_train, frac_val=frac_val,
        min_tariff_count=min_tariff_count,
    )
    train_ramz_set = split["train_ramz"]
    global VAL_RAMZ_SET, RARE_TARIFF_SET
    VAL_RAMZ_SET = split["val_ramz"]
    RARE_TARIFF_SET = rare_tariffs

    L1, L2, L3, global_mean, target_pctiles, diag = \
        compute_hierarchical_aggregates(
            csv_path, train_ramz_set,
            physical_rate_max=physical_rate_max,
        )

    # NEW: fit per-tariff GLM
    glm_by_tariff, global_glm, glm_n = fit_per_tariff_glm(
        csv_path, train_ramz_set, L1, L2, L3, global_mean,
        physical_rate_max=physical_rate_max,
    )

    n1_sizes = np.array([d["n"] for d in L1.values()])
    print(f"   L1 buckets with n=1: {(n1_sizes == 1).sum():,} "
          f"({100*(n1_sizes == 1).mean():.1f}%)\n")

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

    with open(output_dir / "static_cardinalities_ramz.json", "w") as f:
        json.dump({k: len(v) for k, v in CAT_MAPS.items()}, f, indent=2)

    # NEW v5b: save tariff_code → embedding_index mapping for GLM lookup
    tariff_to_idx = {str(k): int(v)
                     for k, v in CAT_MAPS.get("tariff_code", {}).items()}
    with open(output_dir / "tariff_code_to_index.json", "w") as f:
        json.dump(tariff_to_idx, f, indent=2)
    print(f"   Saved tariff→index map: "
          f"{len(tariff_to_idx)} entries → tariff_code_to_index.json")

    with open(output_dir / "peer_avg_config.json", "w") as f:
        json.dump({
            "version": "5b",
            "k_shrink": K_SHRINK,
            "l1_period": L1_PERIOD,
            "global_mean_rate": global_mean,
            "physical_rate_max": physical_rate_max,
            "min_tariff_count": min_tariff_count,
            "n_rare_tariffs_bucketed": len(rare_tariffs),
            "rare_tariff_label": RARE_TARIFF_CODE,
            "split_seed": seed,
            "n_train_meters": len(train_ramz_set),
            "n_val_meters":   len(VAL_RAMZ_SET),
            "n_test_meters":  len(split["test_ramz"]),
            "train_target_percentiles": target_pctiles,
            "n_value_features": 5,
            "value_feature_names": ["rate", "dt", "peer_avg",
                                    "sin_doy", "cos_doy"],
            "glm_ridge_lambda": GLM_RIDGE_LAMBDA,
            "glm_feature_names": GLM_FEATURE_NAMES,
            "glm_n_tariffs": len(glm_by_tariff),
        }, f, indent=2)

    with open(output_dir / "peer_avg_aggregates.pkl", "wb") as f:
        pickle.dump({
            "L1": L1, "L2": L2, "L3": L3,
            "global_mean": global_mean,
            "k_shrink": K_SHRINK,
            "l1_period": L1_PERIOD,
            "physical_rate_max": physical_rate_max,
            "rare_tariffs": rare_tariffs,
            "train_target_percentiles": target_pctiles,
            # NEW
            "glm_by_tariff": glm_by_tariff,
            "global_glm": global_glm,
            "glm_feature_names": GLM_FEATURE_NAMES,
            "glm_ridge_lambda": GLM_RIDGE_LAMBDA,
            "glm_n_samples_per_tariff": glm_n,
        }, f)

    print("\n✅ PREPROCESSING COMPLETE (v5b)")
    print(f"   global_mean       : {global_mean:.3f}")
    print(f"   p99.5             : {target_pctiles['p99.5']:.0f} kWh")
    print(f"   GLMs fitted       : {len(glm_by_tariff)}")
    print(f"   outputs           : {[str(p) for p in out_paths.values()]}")


if __name__ == "__main__":
    preprocess(r"E:\data\dastkari\data_train_wide.csv")