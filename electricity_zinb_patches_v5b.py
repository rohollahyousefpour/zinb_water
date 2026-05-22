"""
Electricity-ZINB Patches v5b — GLM-Pretrained Baseline + Wider Output Scale
==========================================================================
Drop-in replacements for v5 patches with two changes:

  1. GLM-pretrained baseline rate. The model loads per-tariff linear
     regression coefficients from peer_avg_aggregates.pkl and uses
     them to compute a much better `base_rate` than v5's
     `wl*lag1 + we*ema` blend.

     The coefficients are FROZEN at training time (registered as
     non-parameter buffers) so the GLM acts as a fixed, principled
     baseline. The transformer learns corrections on top.

  2. Output scale clamp widened from ±3 to ±4. This lets the model
     scale up more aggressively for industrial tariffs where the
     observed log_scale was being pinned at the clamp.

NPZ schema is unchanged from v5: values=(T, 5).

Usage:
    import electricity_zinb_patches_v5b as p
    ds = p.ImprovedElectricityMeterDataset(npz, cards)
    model = p.ImprovedZINBElectricityMeterEncoder(
        ..., aggregates_pkl="peer_avg_aggregates.pkl",
    )
"""

import math
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import Improved_embeding as orig


# Value-channel indices (v5 schema)
VAL_IDX_RATE    = 0
VAL_IDX_DT      = 1
VAL_IDX_PEERAVG = 2
VAL_IDX_SINDOY  = 3
VAL_IDX_COSDOY  = 4
N_VALUE_CHANNELS = 5

# GLM features (must match preprocessor v5b)
GLM_FEATURE_NAMES = ["const", "lag1", "ema", "peer_avg",
                     "sin_doy", "cos_doy"]
GLM_N_FEATURES = len(GLM_FEATURE_NAMES)

# Output scale clamp
OUTPUT_SCALE_CLAMP = 4.0   # widened from 3.0


# ════════════════════════════════════════════════════════════
# AbsoluteTimeEmbedding (unchanged)
# ════════════════════════════════════════════════════════════
class AbsoluteTimeEmbedding(nn.Module):
    def __init__(self, d_model: int, n_years: int = 12,
                 reference_year: int = 2013):
        super().__init__()
        self.n_years = n_years
        self.reference_year = reference_year

        self.month_emb  = nn.Embedding(12, d_model // 4)
        self.season_emb = nn.Embedding(4, d_model // 4)
        self.year_emb   = nn.Embedding(n_years, d_model // 4)

        self.day_of_year_proj = nn.Sequential(
            nn.Linear(4, d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, d_model // 4),
        )
        self.combine = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            nn.SiLU(), nn.Linear(d_model, d_model),
        )

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        day_of_year = times % 365.25
        month  = (day_of_year / 30.44).long().clamp(0, 11)
        season = (month // 3).clamp(0, 3)
        year   = (times / 365.25).long().clamp(0, self.n_years - 1)
        annual_angle      = 2 * math.pi * day_of_year / 365.25
        semi_annual_angle = 4 * math.pi * day_of_year / 365.25
        cyclical = torch.stack([
            torch.sin(annual_angle),       torch.cos(annual_angle),
            torch.sin(semi_annual_angle),  torch.cos(semi_annual_angle),
        ], dim=-1)
        month_e  = self.month_emb(month)
        season_e = self.season_emb(season)
        year_e   = self.year_emb(year)
        day_e    = self.day_of_year_proj(cyclical)
        return self.combine(torch.cat([month_e, season_e, year_e, day_e],
                                      dim=-1))


orig.AbsoluteTimeEmbedding = AbsoluteTimeEmbedding


# ════════════════════════════════════════════════════════════
# Dataset (unchanged from v5 — 5-channel values)
# ════════════════════════════════════════════════════════════
class ImprovedElectricityMeterDataset(Dataset):
    def __init__(self, npz_path: str,
                 static_cardinalities: Optional[OrderedDict] = None,
                 reference_date: str = "2013-01-01"):
        data = np.load(npz_path, allow_pickle=True)
        self.values = data["values"]
        self.times  = data["times"]
        self.mask   = data["masks"]
        self.static = data["static"]
        self.ramz   = data["ramz"]
        self.N      = len(self.values)
        self.reference_date = np.datetime64(reference_date)

        if self.N > 0:
            first_v = self.values[0]
            if first_v.ndim == 2 and first_v.shape[1] != N_VALUE_CHANNELS:
                raise ValueError(
                    f"Expected {N_VALUE_CHANNELS} value channels (v5 schema), "
                    f"got {first_v.shape[1]}."
                )

        if static_cardinalities is not None:
            self._validate_static(static_cardinalities)

        self.rate_mean, self.rate_std = self._compute_rate_stats()

    def __len__(self):
        return self.N

    def _validate_static(self, cardinalities):
        n_feats = len(cardinalities)
        assert self.static.shape[1] == n_feats
        for i, (name, card) in enumerate(cardinalities.items()):
            col = self.static[:, i]
            if col.min() < 0:
                raise ValueError(f"Feature '{name}' has negative index")
            if col.max() >= card:
                raise ValueError(
                    f"Feature '{name}': max {col.max()} >= card {card}")

    def _compute_rate_stats(self):
        all_rates = []
        for i in range(self.N):
            m = self.mask[i].astype(bool)[:, 0]
            if not m.any():
                continue
            v = self.values[i]
            all_rates.append(v[m, VAL_IDX_RATE])
        if not all_rates:
            return 0.0, 1.0
        all_rates = np.concatenate(all_rates)
        return float(np.median(all_rates)), float(max(np.std(all_rates), 1e-4))

    def __getitem__(self, idx):
        v = self.values[idx].copy()
        t = self.times[idx]
        m = self.mask[idx].astype(bool)[:, 0]
        if len(t) > 0 and isinstance(t[0], np.datetime64):
            abs_times = (t - self.reference_date) / np.timedelta64(1, 'D')
        elif len(t) > 0 and isinstance(t[0], np.timedelta64):
            abs_times = t / np.timedelta64(1, 'D')
        else:
            abs_times = t.astype(np.float64)
        v[m, VAL_IDX_RATE] = v[m, VAL_IDX_RATE] * v[m, VAL_IDX_DT]
        v[~m] = 0.0
        return {
            "values": torch.tensor(v, dtype=torch.float32),
            "times":  torch.tensor(abs_times.copy(), dtype=torch.float32),
            "mask":   torch.tensor(m, dtype=torch.bool),
            "static": torch.tensor(self.static[idx], dtype=torch.long),
            "ramz":   torch.tensor(self.ramz[idx], dtype=torch.long),
        }


ImprovedWaterMeterDataset = ImprovedElectricityMeterDataset


# ════════════════════════════════════════════════════════════
# Metrics (unchanged from v5)
# ════════════════════════════════════════════════════════════
class ElectricityMeterMetrics:
    def __init__(self, zero_threshold_rate: Optional[float] = None):
        self.zero_threshold_rate = zero_threshold_rate
        self.reset()

    def reset(self):
        self.predictions, self.targets = [], []
        self.gates, self.masks, self.dts = [], [], []

    @torch.no_grad()
    def update(self, out):
        expected = (1 - out["gate"]) * out["mu"]
        self.predictions.append(expected.cpu())
        self.targets.append(out["target"].cpu())
        self.gates.append(out["gate"].cpu())
        self.masks.append(out["mask"].cpu())
        self.dts.append(out["dt"].cpu())

    def compute(self):
        preds   = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets,     dim=0)
        gates   = torch.cat(self.gates,       dim=0)
        masks   = torch.cat(self.masks,       dim=0)
        dts     = torch.cat(self.dts,         dim=0)
        m_flat = masks.bool()
        p = preds[m_flat]; t = targets[m_flat]
        g = gates[m_flat]; d = dts[m_flat]
        if p.numel() == 0:
            return {"mae": 0.0, "rmse": 0.0, "n": 0}
        err = (p - t).abs()
        mae = err.mean().item()
        rmse = (err.pow(2).mean()).sqrt().item()
        if self.zero_threshold_rate is None:
            pred_zero = (g > 0.5)
        else:
            pred_zero = (p / d.clamp(min=1.0)) < self.zero_threshold_rate
        actual_zero = (t < 1e-3)
        tp = (pred_zero & actual_zero).sum().item()
        fp = (pred_zero & ~actual_zero).sum().item()
        fn = (~pred_zero & actual_zero).sum().item()
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-9))
        return {
            "mae": mae, "rmse": rmse, "n": int(p.numel()),
            "zero_precision": precision, "zero_recall": recall,
            "zero_f1": f1,
            "actual_zero_rate": float(actual_zero.float().mean()),
            "pred_zero_rate":   float(pred_zero.float().mean()),
        }


WaterMeterMetrics = ElectricityMeterMetrics


# ════════════════════════════════════════════════════════════
# GLM helper — load coefficients into a per-tariff lookup table
# ════════════════════════════════════════════════════════════
def load_glm_coefficients(aggregates_pkl: str,
                          static_cardinalities: OrderedDict,
                          tariff_remapping: Optional[dict] = None):
    """
    Load GLM coefficients from preprocessor pickle and arrange them
    into a tensor of shape (c_tariff, GLM_N_FEATURES).

    Args:
      aggregates_pkl: path to peer_avg_aggregates.pkl
      static_cardinalities: from static_cardinalities_ramz.json
      tariff_remapping: optional map raw_tariff_code -> embedding_index.
                       If None, attempts to read from CAT_MAPS-style data.

    Returns:
      coef_tensor: torch.Tensor of shape (c_tariff, K=6)
      global_glm:  torch.Tensor of shape (K,)
    """
    with open(aggregates_pkl, "rb") as f:
        agg = pickle.load(f)

    glm_by_tariff = agg["glm_by_tariff"]
    global_glm = agg["global_glm"]
    K = GLM_N_FEATURES

    c_tariff = static_cardinalities["tariff_code"]

    # Initialize all tariff coefficients to the global GLM (fallback)
    coef = np.tile(global_glm.astype(np.float32), (c_tariff, 1))

    # If we have a remapping from raw tariff codes to embedding indices,
    # use it. Otherwise, we have a problem because we don't know which
    # embedding row corresponds to which raw tariff_code in the pickle.
    #
    # Strategy: read the static_cardinalities_ramz.json mapping at runtime
    # via the json file's companion, OR require the user to pass it in.
    #
    # For now, we save the encoded mapping separately. Read from a
    # secondary file `tariff_code_to_index.json` if present.
    mapping_path = Path(aggregates_pkl).parent / "tariff_code_to_index.json"
    if mapping_path.exists():
        import json
        with open(mapping_path) as f:
            raw_to_idx = json.load(f)
        # raw_to_idx keys are strings; pickle keys may be int/str
        for raw_tariff, coef_vec in glm_by_tariff.items():
            key = str(raw_tariff)
            if key in raw_to_idx:
                idx = raw_to_idx[key]
                if 0 <= idx < c_tariff:
                    coef[idx] = coef_vec.astype(np.float32)
        print(f"[GLM] Loaded {len(glm_by_tariff)} per-tariff GLM coefficients "
              f"into {c_tariff} embedding rows.")
    else:
        print(f"[GLM] WARNING: {mapping_path} not found. Using global GLM "
              f"for all {c_tariff} tariff embedding rows. "
              f"To enable per-tariff GLMs, save tariff_code_to_index.json "
              f"from the preprocessor's CAT_MAPS['tariff_code'].")

    return (torch.tensor(coef, dtype=torch.float32),
            torch.tensor(global_glm.astype(np.float32), dtype=torch.float32))


# ════════════════════════════════════════════════════════════
# Model — GLM-pretrained baseline
# ════════════════════════════════════════════════════════════
N_HANDCRAFTED_FEATURES = 11  # same as v5


class ImprovedZINBElectricityMeterEncoder(nn.Module):
    """
    v5b encoder.

    Differences vs v5:
      * `base_rate` is computed from GLM coefficients (frozen, per-tariff)
        instead of the lag1/ema blend with learnable weights.
      * Output scale clamp widened from ±3 to ±OUTPUT_SCALE_CLAMP.

    The model still produces a correction on top of base_rate, but base_rate
    is now a stronger, tariff-specific linear regression baseline.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        static_cardinalities: OrderedDict,
        dropout: float = 0.05,
        use_time_aware_attention: bool = True,
        default_rate: float = 5.0,
        d_static_ctx: int = 64,
        aggregates_pkl: Optional[str] = "peer_avg_aggregates.pkl",
    ):
        super().__init__()
        self.d_model = d_model
        self.use_time_aware_attention = use_time_aware_attention
        self.default_rate = default_rate

        self.static_cardinalities = OrderedDict(static_cardinalities)
        self._idx = {name: i for i, name
                     in enumerate(self.static_cardinalities.keys())}

        # Static embeddings
        d_emb = d_model // 8
        self.static_head_embeddings = nn.ModuleList([
            nn.Embedding(card, d_emb)
            for card in self.static_cardinalities.values()
        ])
        static_raw_dim = d_emb * len(self.static_cardinalities)

        c_tariff = self.static_cardinalities["tariff_code"]
        c_type   = self.static_cardinalities["meter_type"]
        c_urban  = self.static_cardinalities["is_urban"]
        c_region = self.static_cardinalities["region_in"]
        c_phase  = self.static_cardinalities["phase"]
        c_amper  = self.static_cardinalities["amper"]

        # Default-rate fallback (used when GLM produces non-positive baseline)
        default_log = math.log(max(default_rate, 1e-3))
        self.default_rate_by_tariff = nn.Embedding(c_tariff, 1)
        self.default_rate_by_type   = nn.Embedding(c_type, 1)
        self.default_rate_by_urban  = nn.Embedding(c_urban, 1)
        self.default_rate_by_region = nn.Embedding(c_region, 1)
        self.default_rate_by_phase  = nn.Embedding(c_phase, 1)
        self.default_rate_by_amper  = nn.Embedding(c_amper, 1)
        for emb in [self.default_rate_by_tariff, self.default_rate_by_type,
                    self.default_rate_by_urban,  self.default_rate_by_region,
                    self.default_rate_by_phase,  self.default_rate_by_amper]:
            nn.init.constant_(emb.weight, default_log / 6.0)

        # Output scale (per tariff and per meter type) — clamp widened
        self.tariff_output_log_scale     = nn.Embedding(c_tariff, 1)
        self.meter_type_output_log_scale = nn.Embedding(c_type, 1)
        nn.init.zeros_(self.tariff_output_log_scale.weight)
        nn.init.zeros_(self.meter_type_output_log_scale.weight)

        # Gate / alpha biases
        self.segment_gate_bias_tariff = nn.Embedding(c_tariff, 1)
        self.segment_gate_bias_urban  = nn.Embedding(c_urban, 1)
        nn.init.zeros_(self.segment_gate_bias_tariff.weight)
        nn.init.zeros_(self.segment_gate_bias_urban.weight)

        self.segment_alpha_bias = nn.Embedding(c_tariff, 1)
        nn.init.zeros_(self.segment_alpha_bias.weight)

        # ── GLM coefficients (FROZEN buffer, not parameter) ──
        # Shape: (c_tariff, GLM_N_FEATURES=6)
        if aggregates_pkl is not None and Path(aggregates_pkl).exists():
            glm_coef, global_glm = load_glm_coefficients(
                aggregates_pkl, self.static_cardinalities)
            self.register_buffer("glm_coef", glm_coef)
            self.register_buffer("global_glm", global_glm)
            self._glm_loaded = True
            print(f"[v5b] GLM baseline ENABLED "
                  f"(coef shape {tuple(glm_coef.shape)})")
        else:
            # Fallback: zero GLM means baseline reverts to default rate
            self.register_buffer(
                "glm_coef",
                torch.zeros(c_tariff, GLM_N_FEATURES, dtype=torch.float32),
            )
            self.register_buffer(
                "global_glm",
                torch.zeros(GLM_N_FEATURES, dtype=torch.float32),
            )
            self._glm_loaded = False
            print(f"[v5b] GLM baseline DISABLED (pickle not found). "
                  f"Falling back to default rate baseline.")

        # Static context projection
        self.static_ctx_proj = nn.Sequential(
            nn.Linear(static_raw_dim, d_static_ctx),
            nn.LayerNorm(d_static_ctx), nn.SiLU(),
        )

        head_input_dim = d_model + d_static_ctx

        # Input embedding: 10 hand-crafted features (same as v5)
        self.emb = orig.InputEmbedding(
            d_model, static_cardinalities,
            n_value_features=N_HANDCRAFTED_FEATURES,
            dt_index=6, dropout=dropout,
        )

        if use_time_aware_attention:
            self.encoder = orig.TimeAwareTransformerEncoder(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                dim_feedforward=4 * d_model, dropout=dropout, causal=True,
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=4 * d_model, dropout=dropout,
                batch_first=True, norm_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Heads
        self.correction_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1), nn.Softplus(),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self._init_heads()

    def _init_heads(self):
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        nn.init.constant_(self.scale_head[-2].bias, -2.0)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.constant_(self.alpha_head[-1].bias, 0.0)

    def _s(self, static, name):
        return static[:, self._idx[name]]

    def _get_default_rate(self, static):
        log_rate = (
            self.default_rate_by_tariff(self._s(static, "tariff_code")).squeeze(-1)
            + self.default_rate_by_type(self._s(static, "meter_type")).squeeze(-1)
            + self.default_rate_by_urban(self._s(static, "is_urban")).squeeze(-1)
            + self.default_rate_by_region(self._s(static, "region_in")).squeeze(-1)
            + self.default_rate_by_phase(self._s(static, "phase")).squeeze(-1)
            + self.default_rate_by_amper(self._s(static, "amper")).squeeze(-1)
        )
        return F.softplus(log_rate)

    def _get_static_ctx(self, static, L):
        embs = [emb(static[:, i])
                for i, emb in enumerate(self.static_head_embeddings)]
        ctx = self.static_ctx_proj(torch.cat(embs, dim=-1))
        return ctx.unsqueeze(1).expand(-1, L, -1)

    # ════════════════════════════════════════════════════════════
    # GLM baseline: per-tariff linear regression
    # ════════════════════════════════════════════════════════════
    def _compute_glm_baseline(self, static, daily_rate, ema, peer_rate,
                              sin_doy, cos_doy, mask):
        """
        Compute base_rate using per-tariff GLM coefficients.

        Features: [const, lag1, ema_prev, peer, sin_doy, cos_doy]
        Output:   F.softplus(linear combination)  (always positive)
        """
        B, L = daily_rate.shape
        device = daily_rate.device

        # Build lag1 and ema_prev (shifted by 1)
        lag1 = torch.zeros(B, L, device=device)
        lag1[:, 1:] = daily_rate[:, :-1]

        ema_prev = torch.zeros(B, L, device=device)
        ema_prev[:, 1:] = ema[:, :-1]

        # At t=0, we don't have lag1 or ema_prev. Use default_rate as fallback.
        default = self._get_default_rate(static).unsqueeze(1)  # (B, 1)
        lag1[:, :1] = default
        ema_prev[:, :1] = default

        # Look up GLM coefficients per row
        tariff_ids = self._s(static, "tariff_code")  # (B,)
        coef = self.glm_coef[tariff_ids]              # (B, 6)
        # Unpack
        c_const = coef[:, 0:1]   # (B, 1)
        c_lag1  = coef[:, 1:2]
        c_ema   = coef[:, 2:3]
        c_peer  = coef[:, 3:4]
        c_sin   = coef[:, 4:5]
        c_cos   = coef[:, 5:6]

        # Build linear combination
        # Each c_X is (B, 1), each feature is (B, L), so result is (B, L)
        glm_logit = (c_const
                     + c_lag1 * lag1.clamp(0, 500)
                     + c_ema  * ema_prev.clamp(0, 500)
                     + c_peer * peer_rate.clamp(0, 500)
                     + c_sin  * sin_doy
                     + c_cos  * cos_doy)

        # Pass through softplus for positivity, clamp range
        base = F.softplus(glm_logit).clamp(min=0.01, max=500.0)

        # For positions where GLM was untrained (default fallback) or
        # produces near-zero, fall back to default_rate.
        # `default` is (B, 1); broadcast over L.
        base = torch.where(base < 0.05, default.expand_as(base), base)
        return base

    def forward(self, batch):
        values = batch["values"]
        mask   = batch["mask"]
        static = batch["static"]
        times  = batch.get("times", None)
        B, L, C = values.shape
        assert C == N_VALUE_CHANNELS, (
            f"Expected {N_VALUE_CHANNELS} value channels, got {C}."
        )

        target    = values[..., VAL_IDX_RATE].float()
        dt        = values[..., VAL_IDX_DT].float()
        peer_rate = values[..., VAL_IDX_PEERAVG].float()
        sin_doy   = values[..., VAL_IDX_SINDOY].float()
        cos_doy   = values[..., VAL_IDX_COSDOY].float()

        dt_safe  = dt.clamp(min=0.5)
        tgt_safe = target.clamp(min=0.0)
        dt_safe  = torch.where(mask, dt_safe, torch.ones_like(dt_safe))
        tgt_safe = torch.where(mask, tgt_safe, torch.zeros_like(tgt_safe))

        daily_rate = (tgt_safe / dt_safe).clamp(0.0, 1e4)
        peer_rate  = torch.where(mask, peer_rate, torch.zeros_like(peer_rate))
        sin_doy    = torch.where(mask, sin_doy, torch.zeros_like(sin_doy))
        cos_doy    = torch.where(mask, cos_doy, torch.zeros_like(cos_doy))

        # ── Build hand-crafted features (10) — same as v5 ──
        features, lag1, ema = self._build_features(
            daily_rate, dt_safe, mask, static,
            peer_rate, sin_doy, cos_doy)

        # ── GLM baseline rate (replaces v5's wl*lag1 + we*ema) ──
        base_rate = self._compute_glm_baseline(
            static, daily_rate, ema, peer_rate, sin_doy, cos_doy, mask)
        baseline_mu = (base_rate * dt_safe).clamp(min=0.0)

        # ── Transformer encoder ──
        attn_times = times if times is not None else dt_safe.cumsum(dim=1)
        x = self.emb(features, static, mask, absolute_times=attn_times,
                     raw_dt=dt_safe)

        if self.use_time_aware_attention:
            h = self.encoder(x, attn_times, mask)
        else:
            causal = torch.triu(torch.ones(L, L, device=x.device,
                                           dtype=torch.bool),
                                diagonal=1)
            h = self.encoder(x, mask=causal, src_key_padding_mask=~mask)

        static_ctx = self._get_static_ctx(static, L)
        h_cond = torch.cat([h, static_ctx], dim=-1)

        # Correction on top of GLM baseline
        raw_corr = self.correction_head(h_cond).squeeze(-1)
        scale    = self.scale_head(h_cond).squeeze(-1).clamp(max=3.0)
        corr_fac = scale * torch.tanh(raw_corr)
        pred_rate = (base_rate * (1.0 + corr_fac)).clamp(min=0.0)

        # Output scale (per-tariff and per-meter-type) — clamp ±4 (was ±3)
        tariff_ids     = self._s(static, "tariff_code")
        meter_type_ids = self._s(static, "meter_type")
        log_scale = (self.tariff_output_log_scale(tariff_ids).squeeze(-1)
                     + self.meter_type_output_log_scale(meter_type_ids).squeeze(-1)
                    ).clamp(-OUTPUT_SCALE_CLAMP, OUTPUT_SCALE_CLAMP)
        out_scale = log_scale.exp()
        mu = (pred_rate * dt_safe * out_scale.unsqueeze(1))\
             .clamp(min=0.0, max=5e4)

        # Alpha (NB dispersion)
        raw_alpha = self.alpha_head(h_cond).squeeze(-1)
        seg_ab = self.segment_alpha_bias(tariff_ids).squeeze(-1)
        raw_alpha = (raw_alpha + seg_ab.unsqueeze(1)).clamp(-5.0, 5.0)
        alpha = (F.softplus(raw_alpha) + 0.05).clamp(min=0.05, max=20.0)

        # Gate (zero-inflation)
        raw_gate = self.gate_head(h_cond).squeeze(-1)
        seg_gb = (self.segment_gate_bias_tariff(tariff_ids).squeeze(-1)
                  + self.segment_gate_bias_urban(self._s(static, "is_urban")).squeeze(-1))
        raw_gate = (raw_gate + seg_gb.unsqueeze(1)).clamp(-10.0, 10.0)
        gate = torch.sigmoid(raw_gate).clamp(min=1e-6, max=1 - 1e-6)

        return {
            "mu": mu, "alpha": alpha, "gate": gate, "gate_logits": raw_gate,
            "target": tgt_safe, "dt": dt_safe, "mask": mask,
            "times": attn_times, "hidden": h,
            "baseline_mu": baseline_mu, "baseline_rate": base_rate,
            "rate_correction": pred_rate - base_rate,
            "predicted_rate": pred_rate,
            "correction_factor": corr_fac, "output_scale": out_scale,
            "peer_rate": peer_rate,
            "_static_for_loss": static,
        }

    def _build_features(self, daily_rate, dt_safe, mask, static,
                        peer_rate, sin_doy, cos_doy):
        """Identical to v5 — produces 10-dim feature tensor."""
        B, L = daily_rate.shape
        device = daily_rate.device
        default = self._get_default_rate(static)
        default_expanded = default.unsqueeze(1)

        lag1 = torch.zeros(B, L, device=device)
        lag1[:, 1:] = daily_rate[:, :-1]
        lag1[:, 0]  = default

        lag2 = torch.zeros(B, L, device=device)
        if L >= 2:
            lag2[:, 2:] = daily_rate[:, :-2]
            lag2[:, :2] = default_expanded.expand(-1, 2)
        else:
            lag2[:, 0] = default

        alpha_ema = 0.3
        ema = torch.zeros(B, L, device=device)
        ema[:, 0] = default
        for t in range(1, L):
            v_prev = mask[:, t - 1].float()
            ema[:, t] = (v_prev * (alpha_ema * daily_rate[:, t - 1]
                                   + (1 - alpha_ema) * ema[:, t - 1])
                         + (1 - v_prev) * ema[:, t - 1])

        masked_rate = daily_rate * mask.float()
        cs    = masked_rate.cumsum(dim=1)
        cs_sq = (masked_rate ** 2).cumsum(dim=1)
        cnt   = mask.float().cumsum(dim=1).clamp(min=1)
        run_mean = cs / cnt
        run_var  = (cs_sq / cnt - run_mean ** 2).clamp(min=0)
        run_std  = run_var.sqrt()

        sh_std = torch.zeros_like(run_std)
        sh_std[:, 1:] = run_std[:, :-1]
        sh_mean = torch.zeros_like(run_mean)
        sh_mean[:, 1:] = run_mean[:, :-1]
        sh_mean[:, 0]  = default

        norm_scale = sh_mean.clamp(min=1.0)
        rate_diff = lag1 - lag2

        peer_clean = peer_rate * mask.float()
        features = torch.stack([
            lag1, lag1 / norm_scale, lag2 / norm_scale,
            ema / norm_scale, sh_std / norm_scale,
            rate_diff / norm_scale, dt_safe / 30.0,
            peer_clean, (lag1 - peer_clean) / norm_scale,
            sin_doy, cos_doy,
        ], dim=-1)
        return features, lag1, ema


ImprovedZINBWaterMeterEncoder = ImprovedZINBElectricityMeterEncoder
