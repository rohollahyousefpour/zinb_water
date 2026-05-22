"""
Electricity-ZINB Patches v5 — Updated for Preprocessor v5
==========================================================
Drop-in replacements wired to the v5 NPZ schema:

  values: (T, 5) — [rate, dt_days, peer_avg, sin_doy, cos_doy]

Changes vs the previous patches file:

  1. Dataset `__getitem__` now transforms ONLY column 0 (rate → cons)
     and passes columns 2–4 through unchanged (peer_avg is already
     a rate; sin_doy/cos_doy are scale-free).

  2. Model `_build_features` now produces a 10-feature tensor:
        [lag1, lag1/scale, lag2/scale, ema/scale,
         sh_std/scale, rate_diff/scale, dt_safe/30,
         peer_avg/scale, sin_doy, cos_doy]
     InputEmbedding receives n_value_features=10, dt_index=6.

  3. `forward()` extracts the new channels from values and feeds
     them to `_build_features`, plus masks them where applicable.

  4. AbsoluteTimeEmbedding still constructible with n_years.
     Dataset/metrics/aliases unchanged in spirit.

Usage:
    import electricity_zinb_patches_v5 as p
    ds = p.ImprovedElectricityMeterDataset(npz, cards)
    model = p.ImprovedZINBElectricityMeterEncoder(...)
"""

import math
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import Improved_embeding as orig


# ════════════════════════════════════════════════════════════
# Value-channel indices (must match preprocessor v5 schema)
# ════════════════════════════════════════════════════════════
VAL_IDX_RATE    = 0
VAL_IDX_DT      = 1
VAL_IDX_PEERAVG = 2
VAL_IDX_SINDOY  = 3
VAL_IDX_COSDOY  = 4
N_VALUE_CHANNELS = 5


# ════════════════════════════════════════════════════════════
# PATCH 1 — AbsoluteTimeEmbedding (unchanged from v4 patches)
# ════════════════════════════════════════════════════════════
class AbsoluteTimeEmbedding(nn.Module):
    """Calendar-time embeddings with configurable n_years."""

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
# PATCH 2 — Dataset (5-channel values; only col 0 → consumption)
# ════════════════════════════════════════════════════════════
class ImprovedElectricityMeterDataset(Dataset):
    """
    Electricity-meter dataset reading the v5 NPZ schema:

      values: (T, 5) = [rate, dt_days, peer_avg, sin_doy, cos_doy]
      static: (N, S) with S = len(static_cardinalities)

    __getitem__ multiplies column 0 (rate) by column 1 (dt) so the
    model receives total consumption at index 0. Columns 2–4 are
    passed through unchanged.
    """

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

        # Sanity check the schema
        if self.N > 0:
            first_v = self.values[0]
            if first_v.ndim == 2 and first_v.shape[1] != N_VALUE_CHANNELS:
                raise ValueError(
                    f"Expected {N_VALUE_CHANNELS} value channels (v5 schema), "
                    f"got {first_v.shape[1]}. Re-run preprocessor v5."
                )

        if static_cardinalities is not None:
            self._validate_static(static_cardinalities)

        self.rate_mean, self.rate_std = self._compute_rate_stats()

    def __len__(self):
        return self.N

    def _validate_static(self, cardinalities):
        n_feats = len(cardinalities)
        assert self.static.shape[1] == n_feats, (
            f"Static array has {self.static.shape[1]} columns but "
            f"cardinalities define {n_feats}. "
            f"Expected columns: {list(cardinalities.keys())}"
        )
        for i, (name, card) in enumerate(cardinalities.items()):
            col = self.static[:, i]
            if col.min() < 0:
                raise ValueError(
                    f"Feature '{name}' (col {i}) has negative index"
                )
            if col.max() >= card:
                raise ValueError(
                    f"Feature '{name}' (col {i}): max index {col.max()} "
                    f"≥ cardinality {card}"
                )

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

        # Recover consumption from (rate, dt) — only on column 0
        v[m, VAL_IDX_RATE] = v[m, VAL_IDX_RATE] * v[m, VAL_IDX_DT]
        v[~m] = 0.0

        return {
            "values": torch.tensor(v, dtype=torch.float32),
            "times":  torch.tensor(abs_times.copy(), dtype=torch.float32),
            "mask":   torch.tensor(m, dtype=torch.bool),
            "static": torch.tensor(self.static[idx], dtype=torch.long),
            "ramz":   torch.tensor(self.ramz[idx], dtype=torch.long),
        }


# Backwards-compat alias
ImprovedWaterMeterDataset = ImprovedElectricityMeterDataset


# ════════════════════════════════════════════════════════════
# PATCH 3 — Metrics (unchanged from v4 patches)
# ════════════════════════════════════════════════════════════
class ElectricityMeterMetrics:
    """
    Test-time metrics for electricity ZINB model.

    Zero-classification: gate > 0.5 by default.
    """

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

        # Variable-L safe: flatten valid positions only
        m_flat = masks.bool()
        p = preds[m_flat]
        t = targets[m_flat]
        g = gates[m_flat]
        d = dts[m_flat]

        if p.numel() == 0:
            return {"mae": 0.0, "rmse": 0.0, "n": 0}

        err = (p - t).abs()
        mae = err.mean().item()
        rmse = (err.pow(2).mean()).sqrt().item()

        # Zero classification (gate-based by default)
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
# PATCH 4 — Model (10 hand-crafted features incl. peer_avg + seasonality)
# ════════════════════════════════════════════════════════════
N_HANDCRAFTED_FEATURES = 10  # lag1, lag1/s, lag2/s, ema/s, sh_std/s,
                              # rate_diff/s, dt/30, peer/s, sin_doy, cos_doy


class ImprovedZINBElectricityMeterEncoder(nn.Module):
    """
    v5 encoder. Inputs:
      values: (B, L, 5) = [target=rate*dt, dt, peer_avg, sin_doy, cos_doy]
      mask:   (B, L) bool
      static: (B, S) long, with S = len(static_cardinalities)
      times:  (B, L) float, absolute days from reference date (optional)

    Differences vs prior version:
      * Builds 10 hand-crafted features (was 7 or 8).
      * peer_avg gets the same normalisation as lag1 (divides by sh_mean).
      * sin_doy/cos_doy are passed through unmodified.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.use_time_aware_attention = use_time_aware_attention
        self.default_rate = default_rate

        self.static_cardinalities = OrderedDict(static_cardinalities)
        self._idx = {name: i for i, name
                     in enumerate(self.static_cardinalities.keys())}

        # ── Static embeddings (one per feature) ──
        d_emb = d_model // 8
        self.static_head_embeddings = nn.ModuleList([
            nn.Embedding(card, d_emb)
            for card in self.static_cardinalities.values()
        ])
        static_raw_dim = d_emb * len(self.static_cardinalities)

        # ── Per-segment biases / blends ──
        c_tariff = self.static_cardinalities["tariff_code"]
        c_type   = self.static_cardinalities["meter_type"]
        c_urban  = self.static_cardinalities["is_urban"]
        c_region = self.static_cardinalities["region_in"]
        c_phase  = self.static_cardinalities["phase"]
        c_amper  = self.static_cardinalities["amper"]

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

        self.baseline_blend_tariff = nn.Embedding(c_tariff, 2)
        self.baseline_blend_urban  = nn.Embedding(c_urban, 2)
        nn.init.zeros_(self.baseline_blend_tariff.weight)
        nn.init.zeros_(self.baseline_blend_urban.weight)

        self.tariff_output_log_scale     = nn.Embedding(c_tariff, 1)
        self.meter_type_output_log_scale = nn.Embedding(c_type, 1)
        nn.init.zeros_(self.tariff_output_log_scale.weight)
        nn.init.zeros_(self.meter_type_output_log_scale.weight)

        self.segment_gate_bias_tariff = nn.Embedding(c_tariff, 1)
        self.segment_gate_bias_urban  = nn.Embedding(c_urban, 1)
        nn.init.zeros_(self.segment_gate_bias_tariff.weight)
        nn.init.zeros_(self.segment_gate_bias_urban.weight)

        self.segment_alpha_bias = nn.Embedding(c_tariff, 1)
        nn.init.zeros_(self.segment_alpha_bias.weight)

        # ── Static context projection (broadcast over time) ──
        self.static_ctx_proj = nn.Sequential(
            nn.Linear(static_raw_dim, d_static_ctx),
            nn.LayerNorm(d_static_ctx), nn.SiLU(),
        )

        head_input_dim = d_model + d_static_ctx

        # ── Input embedding: 10 hand-crafted features ──
        self.emb = orig.InputEmbedding(
            d_model, static_cardinalities,
            n_value_features=N_HANDCRAFTED_FEATURES,
            dt_index=6, dropout=dropout,
        )

        # ── Encoder ──
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

        # ── Heads ──
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

    # ── Convenience accessors ──
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

    def _get_baseline_blend(self, static):
        raw = (self.baseline_blend_tariff(self._s(static, "tariff_code"))
               + self.baseline_blend_urban(self._s(static, "is_urban")))
        w = F.softmax(raw, dim=-1)
        return w[:, 0], w[:, 1]

    def _get_static_ctx(self, static, L):
        embs = [emb(static[:, i])
                for i, emb in enumerate(self.static_head_embeddings)]
        ctx = self.static_ctx_proj(torch.cat(embs, dim=-1))
        return ctx.unsqueeze(1).expand(-1, L, -1)

    def _build_features(self, daily_rate, dt_safe, mask, static,
                        peer_rate, sin_doy, cos_doy):
        """
        Returns features tensor of shape (B, L, 10).

        Feature columns:
          0: lag1                       (kWh/day, abs)
          1: lag1 / sh_mean            (normalised)
          2: lag2 / sh_mean
          3: ema / sh_mean
          4: sh_std / sh_mean          (CV-like)
          5: (lag1 - lag2) / sh_mean   (rate diff)
          6: dt_safe / 30              (gap in months)
          7: peer_rate / sh_mean       (peer-avg, normalised)
          8: sin_doy                   (seasonality)
          9: cos_doy
        """
        B, L = daily_rate.shape
        device = daily_rate.device
        default = self._get_default_rate(static)
        default_expanded = default.unsqueeze(1)

        # lag1, lag2
        lag1 = torch.zeros(B, L, device=device)
        lag1[:, 1:] = daily_rate[:, :-1]
        lag1[:, 0]  = default

        lag2 = torch.zeros(B, L, device=device)
        if L >= 2:
            lag2[:, 2:] = daily_rate[:, :-2]
            lag2[:, :2] = default_expanded.expand(-1, 2)
        else:
            lag2[:, 0] = default

        # Exponential moving average of past rates
        alpha_ema = 0.3
        ema = torch.zeros(B, L, device=device)
        ema[:, 0] = default
        for t in range(1, L):
            v_prev = mask[:, t - 1].float()
            ema[:, t] = (v_prev * (alpha_ema * daily_rate[:, t - 1]
                                   + (1 - alpha_ema) * ema[:, t - 1])
                         + (1 - v_prev) * ema[:, t - 1])

        # Running mean and std of past rates
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

        # peer_rate is already a rate (kWh/day). Normalise like lag1.
        # Zero out where mask is False so we don't leak garbage.
        peer_clean = peer_rate * mask.float()

        features = torch.stack([
            lag1,                          # 0
            lag1 / norm_scale,             # 1
            lag2 / norm_scale,             # 2
            ema / norm_scale,              # 3
            sh_std / norm_scale,           # 4
            rate_diff / norm_scale,        # 5
            dt_safe / 30.0,                # 6  ← dt_index for InputEmbedding
            peer_clean / norm_scale,       # 7
            sin_doy,                       # 8
            cos_doy,                       # 9
        ], dim=-1)
        return features, lag1, ema

    def _build_baseline_rate(self, lag1_rate, ema_rate, mask, static):
        wl, we = self._get_baseline_blend(static)
        wl = wl.unsqueeze(1); we = we.unsqueeze(1)
        base = wl * lag1_rate + we * ema_rate
        default = self._get_default_rate(static).unsqueeze(1)
        return torch.where(base < 1e-6, default.expand_as(base), base)

    def forward(self, batch):
        values = batch["values"]
        mask   = batch["mask"]
        static = batch["static"]
        times  = batch.get("times", None)
        B, L, C = values.shape
        assert C == N_VALUE_CHANNELS, (
            f"Expected {N_VALUE_CHANNELS} value channels, got {C}. "
            f"Re-run preprocessor v5."
        )

        # Extract channels
        target    = values[..., VAL_IDX_RATE].float()    # cons = rate*dt
        dt        = values[..., VAL_IDX_DT].float()
        peer_rate = values[..., VAL_IDX_PEERAVG].float()
        sin_doy   = values[..., VAL_IDX_SINDOY].float()
        cos_doy   = values[..., VAL_IDX_COSDOY].float()

        dt_safe  = dt.clamp(min=0.5)
        tgt_safe = target.clamp(min=0.0)
        dt_safe  = torch.where(mask, dt_safe, torch.ones_like(dt_safe))
        tgt_safe = torch.where(mask, tgt_safe, torch.zeros_like(tgt_safe))

        daily_rate = (tgt_safe / dt_safe).clamp(0.0, 1e4)

        # Mask peer_rate / seasonality where mask is False
        peer_rate = torch.where(mask, peer_rate,
                                torch.zeros_like(peer_rate))
        sin_doy = torch.where(mask, sin_doy, torch.zeros_like(sin_doy))
        cos_doy = torch.where(mask, cos_doy, torch.zeros_like(cos_doy))

        features, lag1, ema = self._build_features(
            daily_rate, dt_safe, mask, static,
            peer_rate, sin_doy, cos_doy,
        )
        base_rate = self._build_baseline_rate(lag1, ema, mask, static)
        baseline_mu = (base_rate * dt_safe).clamp(min=0.0)

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

        raw_corr = self.correction_head(h_cond).squeeze(-1)
        scale    = self.scale_head(h_cond).squeeze(-1).clamp(max=3.0)
        corr_fac = scale * torch.tanh(raw_corr)
        pred_rate = (base_rate * (1.0 + corr_fac)).clamp(min=0.0)

        tariff_ids     = self._s(static, "tariff_code")
        meter_type_ids = self._s(static, "meter_type")
        log_scale = (self.tariff_output_log_scale(tariff_ids).squeeze(-1)
                     + self.meter_type_output_log_scale(meter_type_ids).squeeze(-1)
                    ).clamp(-3.0, 3.0)
        out_scale = log_scale.exp()
        mu = (pred_rate * dt_safe * out_scale.unsqueeze(1))\
             .clamp(min=0.0, max=5e4)

        raw_alpha = self.alpha_head(h_cond).squeeze(-1)
        seg_ab = self.segment_alpha_bias(tariff_ids).squeeze(-1)
        raw_alpha = (raw_alpha + seg_ab.unsqueeze(1)).clamp(-5.0, 5.0)
        alpha = (F.softplus(raw_alpha) + 0.05).clamp(min=0.05, max=20.0)

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


ImprovedZINBWaterMeterEncoder = ImprovedZINBElectricityMeterEncoder
