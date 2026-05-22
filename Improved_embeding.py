"""
Time-Aware Transformer for ZINB Water Consumption Forecasting
─────────────────────────────────────────────────────────────

Static features schema:
{
    "meter_type": 2,
    "tariff_code": 72,
    "is_urban": 2,
    "region_in": 2,
    "phase": 2,
    "amper": 6
}
"""

import math
import os
import time
import json
from collections import OrderedDict, defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


# ════════════════════════════════════════════════════════════
# 1. COLLATE FUNCTION
# ════════════════════════════════════════════════════════════

def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Collate variable-length sequences with padding."""
    max_len = max(item["values"].shape[0] for item in batch)
    B = len(batch)
    d_values = batch[0]["values"].shape[-1]
    n_static = batch[0]["static"].shape[0]

    values = torch.zeros(B, max_len, d_values)
    times = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    static = torch.zeros(B, n_static, dtype=torch.long)
    ramz = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["values"].shape[0]
        values[i, :seq_len] = item["values"]
        times[i, :seq_len] = item["times"]
        mask[i, :seq_len] = item["mask"]
        static[i] = item["static"]
        ramz[i]= item['ramz']

    return {
        "values": values,
        "times": times,
        "mask": mask,
        "static": static,
        "ramz": ramz
    }


# ════════════════════════════════════════════════════════════
# 2. EMBEDDING COMPONENTS
# ════════════════════════════════════════════════════════════


class AbsoluteTimeEmbedding(nn.Module):
    """Calendar-time embeddings capturing seasonality and yearly trends."""

    def __init__(self, d_model, n_years=10):
        super().__init__()
        self.n_years = n_years

        self.month_emb = nn.Embedding(12, d_model // 4)
        self.season_emb = nn.Embedding(4, d_model // 4)
        self.year_emb = nn.Embedding(n_years, d_model // 4)

        self.day_of_year_proj = nn.Sequential(
            nn.Linear(4, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model // 4),
        )
        self.combine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, times):
        """times: [B, L] — days since reference date."""
        day_of_year = times % 365.25
        month = (day_of_year / 30.44).long().clamp(0, 11)
        season = (month // 3).clamp(0, 3)
        year = (times / 365.25).long().clamp(0, self.n_years - 1)

        annual_angle = 2 * math.pi * day_of_year / 365.25
        semi_annual_angle = 4 * math.pi * day_of_year / 365.25
        cyclical = torch.stack([
            torch.sin(annual_angle), torch.cos(annual_angle),
            torch.sin(semi_annual_angle), torch.cos(semi_annual_angle),
        ], dim=-1)

        month_e = self.month_emb(month)
        season_e = self.season_emb(season)
        year_e = self.year_emb(year)
        day_e = self.day_of_year_proj(cyclical)

        combined = torch.cat([month_e, season_e, year_e, day_e], dim=-1)
        return self.combine(combined)


class ImprovedTimeEmbedding(nn.Module):
    """Multi-scale time-gap embedding with fixed + learned frequencies."""

    def __init__(self, d_model, n_frequencies=16, dropout=0.05):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(math.log(1e-3), math.log(1e1), n_frequencies)),
        )
        self.n_learned = n_frequencies // 2
        self.learned_freqs = nn.Parameter(torch.randn(self.n_learned) * 0.1)

        in_dim = 5 + 2 * n_frequencies + 2 * self.n_learned
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, dt):
        dt = dt.unsqueeze(-1)
        dt_safe = dt.clamp(min=1e-8)

        basic = torch.cat([
            dt / 60.0,
            torch.log1p(dt_safe),
            torch.sqrt(dt_safe),
            torch.tanh(dt / 30.0),
            (dt > 60).float(),
        ], dim=-1)

        sin_f = torch.sin(dt * self.freqs)
        cos_f = torch.cos(dt * self.freqs)

        learned_f = self.learned_freqs.exp().clamp(max=100.0)
        sin_l = torch.sin(dt * learned_f)
        cos_l = torch.cos(dt * learned_f)

        feats = torch.cat([basic, sin_f, cos_f, sin_l, cos_l], dim=-1)
        return self.mlp(feats)


class RelativePositionBias(nn.Module):
    """Learnable attention bias based on absolute time difference."""

    def __init__(self, n_heads, max_seq_warn=200):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_warn = max_seq_warn
        self._warned = False

        self.time_bias_proj = nn.Sequential(
            nn.Linear(1, n_heads * 2),
            nn.SiLU(),
            nn.Linear(n_heads * 2, n_heads),
        )

    def forward(self, time_diffs):
        """time_diffs: [B, L, L]  →  [B, n_heads, L, L]"""
        L = time_diffs.shape[1]
        if L > self.max_seq_warn and not self._warned:
            print(f"[WARNING] RelativePositionBias: L={L} creates "
                  f"O(B×{L}²×{self.n_heads}) intermediate tensor")
            self._warned = True

        time_diffs_norm = time_diffs.unsqueeze(-1) / 60.0
        bias = self.time_bias_proj(time_diffs_norm)
        return bias.permute(0, 3, 1, 2)


class ImprovedRateEmbedding(nn.Module):
    """Per-feature expansion (raw, log1p, sqrt, tanh) then MLP."""

    def __init__(self, d_model, n_input_features=7, dropout=0.05):
        super().__init__()
        self.n_features = n_input_features
        per_feat = 4  # raw, log1p, sqrt, tanh
        total_dim = n_input_features * per_feat

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, values):
        """values: [B, L, n_features]"""
        feats = []
        for i in range(self.n_features):
            v = values[..., i:i + 1]
            v_safe = v.abs().clamp(min=1e-8)
            feats.extend([
                v / 10.0,
                torch.sign(v) * torch.log1p(v_safe),
                torch.sqrt(v_safe),
                torch.tanh(v / 20.0),
            ])
        return self.mlp(torch.cat(feats, dim=-1))

class StaticEmbedding(nn.Module):
    """
    Embeds categorical static features and projects to d_model.

    Updated for 6-feature static schema:
    {meter_type: 2, tariff_code: 72, is_urban: 2, region_in: 2, phase: 2, amper: 6}
    """

    def __init__(self, cardinalities, d_model):
        super().__init__()
        self.feature_names = list(cardinalities.keys())
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, d_model // 2)
            for card in cardinalities.values()
        ])
        n_static = len(cardinalities)
        self.proj = nn.Sequential(
            nn.Linear(n_static * (d_model // 2), d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, static):
        """static: [B, S] int64  where S = number of static features"""
        embs = [emb(static[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.proj(torch.cat(embs, dim=-1))



class InputEmbedding(nn.Module):
    # REMOVED: unused `max_seq_len` parameter
    def __init__(self, d_model, static_cardinalities,
                 n_value_features=3, dt_index=1, dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.dt_index = dt_index

        self.rate_emb = ImprovedRateEmbedding(
            d_model, n_input_features=n_value_features, dropout=dropout)
        self.dt_emb = ImprovedTimeEmbedding(d_model, dropout=dropout)
        self.static_emb = StaticEmbedding(static_cardinalities, d_model)
        self.abs_time_emb = AbsoluteTimeEmbedding(d_model)

        n_components = 4

        self.combine = nn.Sequential(
            nn.Linear(d_model * n_components, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model * n_components, n_components),
            nn.Softmax(dim=-1),
        )

    def forward(self, values, static, mask, absolute_times=None, raw_dt=None):
        B, L, _ = values.shape

        if raw_dt is not None:
            dt = raw_dt
        else:
            dt = values[..., self.dt_index]

        rate_e = self.rate_emb(values)
        dt_e = self.dt_emb(dt)
        static_e = self.static_emb(static)
        static_e = static_e.unsqueeze(1).expand(-1, L, -1)

        if absolute_times is not None:
            abs_e = self.abs_time_emb(absolute_times)
        else:
            abs_e = self.abs_time_emb(dt.cumsum(dim=1))

        components = [rate_e, dt_e, static_e, abs_e]
        combined = torch.cat(components, dim=-1)

        stacked = torch.stack(components, dim=2)
        gate_w = self.gate(combined)
        gated = (stacked * gate_w.unsqueeze(-1)).sum(dim=2)

        output = self.combine(combined) + gated
        output = output * mask.unsqueeze(-1).float()
        return output


# ════════════════════════════════════════════════════════════
# 3. TRANSFORMER COMPONENTS
# ════════════════════════════════════════════════════════════

class TimeAwareTransformerLayer(nn.Module):
    """Pre-norm Transformer layer with time-aware bias and causal masking."""

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1,
                 causal=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.time_bias = RelativePositionBias(n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_diffs, mask=None):
        B, L, _ = x.shape
        x_norm = self.norm1(x)

        Q = self.q_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = attn + self.time_bias(time_diffs)

        if mask is not None:
            pad_mask = (~mask).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad_mask, float('-inf'))

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = attn - attn.max(dim=-1, keepdim=True).values.clamp(min=-1e9)
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)

        x = x + self.dropout(out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TimeAwareTransformerEncoder(nn.Module):
    """Stack of time-aware Transformer layers."""

    def __init__(self, d_model, n_heads, n_layers, dim_feedforward,
                 dropout=0.1, causal=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TimeAwareTransformerLayer(d_model, n_heads, dim_feedforward,
                                     dropout, causal=causal)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, times, mask=None):
        masked_times = times * mask.float()
        time_diffs = (masked_times.unsqueeze(2) - masked_times.unsqueeze(1)).abs()

        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        time_diffs = time_diffs * pair_mask.float()
        time_diffs = time_diffs.clamp(max=365.0)

        for layer in self.layers:
            x = layer(x, time_diffs, mask)
        return self.final_norm(x)


# ════════════════════════════════════════════════════════════
# 4. DATASET
# ════════════════════════════════════════════════════════════

class ImprovedWaterMeterDataset(Dataset):
    """Enhanced dataset that preserves absolute timestamps."""

    def __init__(self, npz_path: str,
                 static_cardinalities: Optional[OrderedDict] = None,
                 reference_date="2017-01-01"):
        data = np.load(npz_path, allow_pickle=True)

        self.values = data["values"]
        self.times = data["times"]
        self.mask = data["masks"]
        self.static = data["static"]
        self.ramz = data["ramz"]
        self.N = len(self.values)
        self.reference_date = np.datetime64(reference_date)

        if static_cardinalities is not None:
            self._validate_static(static_cardinalities)

        self.rate_mean, self.rate_std = self._compute_rate_stats()
        # REMOVED: self.dt_max = self._compute_dt_max() — never read

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
                raise ValueError(f"Feature '{name}' (col {i}) has negative index")
            if col.max() >= card:
                raise ValueError(
                    f"Feature '{name}' (col {i}): max index {col.max()} ≥ cardinality {card}"
                )

    def _compute_rate_stats(self):
        all_rates = []
        for i in range(self.N):
            m = self.mask[i].astype(bool)[:,0]
            if not m.any():
                continue
            v = self.values[i]
            rates = v[m, 0]
            all_rates.append(rates)
        if not all_rates:
            return 0.0, 1.0
        all_rates = np.concatenate(all_rates)
        return float(np.median(all_rates)), float(max(np.std(all_rates), 1e-4))

    # REMOVED: _compute_dt_max — result was never used

    def __getitem__(self, idx):
        v = self.values[idx].copy()
        t = self.times[idx]
        m = self.mask[idx].astype(bool)[:,0]


        if len(t) > 0 and isinstance(t[0], np.datetime64):
            abs_times = (t - self.reference_date) / np.timedelta64(1, 'D')
        elif len(t) > 0 and isinstance(t[0], np.timedelta64):
            abs_times = t / np.timedelta64(1, 'D')
        else:
            abs_times = t.astype(np.float64)

        v[m, 0] = v[m, 0] * v[m, 1]
        v[~m] = 0.0

        return {
            "values": torch.tensor(v, dtype=torch.float32),
            "times": torch.tensor(abs_times.copy(), dtype=torch.float32),
            "mask": torch.tensor(m, dtype=torch.bool),
            "static": torch.tensor(self.static[idx], dtype=torch.long),
            "ramz":torch.tensor(self.ramz[idx], dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════
# 5. LOSS FUNCTIONS
# ════════════════════════════════════════════════════════════

class ZINBLoss(nn.Module):
    def __init__(self, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, mu, alpha, gate, target, mask, sample_weights=None):
        mu = mu.float().clamp(min=self.eps, max=1e6)
        alpha = alpha.float().clamp(min=0.01, max=100.0)
        gate = gate.float().clamp(min=self.eps, max=1.0 - self.eps)
        target = target.float().clamp(min=0.0)

        r = (1.0 / alpha).clamp(min=1e-4, max=1e4)

        log_mu = torch.log(mu)
        log_r = torch.log(r)
        log_mu_r = torch.log(mu + r)

        lgamma_tr = torch.lgamma((target + r).clamp(min=self.eps, max=1e6))
        lgamma_r = torch.lgamma(r.clamp(min=self.eps, max=1e6))
        lgamma_t1 = torch.lgamma((target + 1.0).clamp(min=1.0, max=1e6))

        nb_ll = (lgamma_tr - lgamma_r - lgamma_t1
                 + target * log_mu + r * log_r
                 - (target + r) * log_mu_r)
        nb_ll = nb_ll.clamp(-300.0, 50.0)

        nb_zero_ll = (r * log_r - r * log_mu_r).clamp(-300.0, 50.0)

        log_gate = torch.log(gate)
        log_1mg = torch.log(1.0 - gate)

        ll_pos = nb_ll + log_1mg
        ll_zero = torch.logaddexp(log_gate, nb_zero_ll + log_1mg)

        ll = torch.where(target < 0.5, ll_zero, ll_pos)
        ll = torch.nan_to_num(ll, nan=-100.0, posinf=-100.0, neginf=-100.0)
        ll = ll.clamp(-300.0, 50.0)

        nll = -ll
        if sample_weights is not None:
            nll = nll * sample_weights
        nll = nll * mask.float()

        if self.reduction == 'mean':
            return nll.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return nll.sum()
        return nll


# REMOVED: SmoothnessLoss — never instantiated
# REMOVED: SeasonalConsistencyLoss — never instantiated
# REMOVED: ZeroInflationRegularizer — never instantiated
# REMOVED: MuScaleRegularizer — never instantiated


# ════════════════════════════════════════════════════════════
# 6. METRICS
# ════════════════════════════════════════════════════════════

class WaterMeterMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.gates = []
        self.masks = []
        self.dts = []

    @torch.no_grad()
    def update(self, out):
        expected = (1 - out["gate"]) * out["mu"]
        self.predictions.append(expected.cpu())
        self.targets.append(out["target"].cpu())
        self.gates.append(out["gate"].cpu())
        self.masks.append(out["mask"].cpu())
        self.dts.append(out["dt"].cpu())

    def compute(self):
        preds = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)
        gates = torch.cat(self.gates, dim=0)
        masks = torch.cat(self.masks, dim=0)

        vp = preds[masks]
        vt = targets[masks]
        vg = gates[masks]

        metrics = {}
        metrics["mae"] = (vp - vt).abs().mean().item()
        metrics["rmse"] = ((vp - vt) ** 2).mean().sqrt().item()
        metrics["mape"] = ((vp - vt).abs() / vt.clamp(min=1)).mean().item() * 100

        lp = torch.log1p(vp)
        lt = torch.log1p(vt)
        metrics["log_mae"] = (lp - lt).abs().mean().item()

        is_zero_t = vt < 0.5
        is_zero_p = vp < 1.5
        metrics["zero_recall"] = is_zero_p[is_zero_t].float().mean().item() if is_zero_t.any() else 0.0
        metrics["zero_precision"] = is_zero_t[is_zero_p].float().mean().item() if is_zero_p.any() else 0.0
        metrics["avg_gate"] = vg.mean().item()
        metrics["actual_zero_rate"] = is_zero_t.float().mean().item()

        if vp.std() > 1e-6 and vt.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([vp, vt]))[0, 1]
            metrics["correlation"] = corr.item() if not torch.isnan(corr) else 0.0
        else:
            metrics["correlation"] = 0.0

        for q in [0.5, 0.9, 0.95]:
            err = (vp - vt).abs()
            metrics[f"p{int(q * 100)}_error"] = torch.quantile(err, q).item()

        return metrics


# ════════════════════════════════════════════════════════════
# 7. TRAINING UTILITIES
# ════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, mode='min',
                 restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None
        self.improved = False

    def __call__(self, score, model, epoch):
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        self.improved = improved

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best and self.best_state is not None:
                    model.load_state_dict(self.best_state)
                return True
            return False


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 warmup_lr=1e-7, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            alpha = epoch / max(self.warmup_epochs, 1)
            lrs = [self.warmup_lr + alpha * (b - self.warmup_lr)
                   for b in self.base_lrs]
        else:
            denom = max(self.total_epochs - self.warmup_epochs, 1)
            progress = (epoch - self.warmup_epochs) / denom
            alpha = 0.5 * (1 + math.cos(math.pi * progress))
            lrs = [self.min_lr + alpha * (b - self.min_lr)
                   for b in self.base_lrs]

        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class GradientAccumulator:
    def __init__(self, accumulation_steps=1):
        self.accumulation_steps = accumulation_steps

    def should_step(self, batch_idx):
        return (batch_idx + 1) % self.accumulation_steps == 0

    def is_last_batch_in_epoch(self, batch_idx, total_batches):
        return batch_idx == total_batches - 1

    def scale_loss(self, loss):
        return loss / self.accumulation_steps


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


class TrainingLogger:
    def __init__(self, log_dir, experiment_name):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.history = defaultdict(list)
        self.start_time = time.time()

    def log_config(self, config):
        path = os.path.join(self.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def log(self, metrics, epoch, phase='train'):
        for k, v in metrics.items():
            self.history[f"{phase}/{k}"].append((epoch, v))

    def log_scalar(self, name, value, epoch):
        self.history[name].append((epoch, value))

    def save_history(self):
        path = os.path.join(self.log_dir, "history.json")
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)

    def get_elapsed_time(self):
        return time.time() - self.start_time

    def print_summary(self, epoch, train_m, val_m, lr):
        elapsed = self.get_elapsed_time()
        h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
        print(f"\n{'═' * 80}")
        print(f"Epoch {epoch:03d} │ Time: {h:02d}h{m:02d}m │ LR: {lr:.2e}")

        common = sorted(set(train_m) & set(val_m))
        if common:
            print(f"{'─' * 80}")
            print(f"  {'Metric':<20} {'Train':>12} {'Val':>12}")
            print(f"  {'─' * 46}")
            for key in common:
                print(f"  {key:<20} {train_m[key]:>12.4f} {val_m[key]:>12.4f}")

        train_only = sorted(set(train_m) - set(val_m))
        if train_only:
            print(f"{'─' * 80}")
            print(f"  Train only:")
            for key in train_only:
                print(f"    {key:<20} {train_m[key]:>12.4f}")

        val_only = sorted(set(val_m) - set(train_m))
        if val_only:
            print(f"{'─' * 80}")
            print(f"  Val only:")
            for key in val_only:
                print(f"    {key:<20} {val_m[key]:>12.4f}")

        print(f"{'═' * 80}\n")


# ════════════════════════════════════════════════════════════
# 8. DATA AUGMENTATION
# ════════════════════════════════════════════════════════════

class TimeSeriesAugmenter:
    def __init__(self, noise_std=0.05,
                 scale_range=(0.9, 1.1),
                 mask_prob=0.1):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.mask_prob = mask_prob

    def __call__(self, batch):
        values = batch["values"].clone()
        mask = batch["mask"].clone()
        B, L, _ = values.shape
        device = values.device

        if self.noise_std > 0:
            magnitude = values[..., 0].abs().clamp(min=1.0)
            noise = (torch.randn_like(values[..., 0])
                     * self.noise_std * torch.sqrt(magnitude))
            noise = noise * mask.float()
            values[..., 0] = (values[..., 0] + noise).clamp(min=0)

        if self.scale_range[0] < self.scale_range[1]:
            scale = torch.empty(B, 1, device=device).uniform_(*self.scale_range)
            values[..., 0] = values[..., 0] * scale

        if self.mask_prob > 0:
            keep = torch.rand(B, L, device=device) > self.mask_prob

            valid_positions = mask.float().cumsum(dim=1)
            max_valid = valid_positions.max(dim=1, keepdim=True).values
            is_recent = valid_positions > (max_valid - 2)
            keep = keep | is_recent

            new_mask = mask & keep
            valid_counts = new_mask.sum(dim=1)
            revert = valid_counts < 2
            new_mask[revert] = mask[revert]
            mask = new_mask
            values[~mask] = 0.0

        batch["values"] = values
        batch["mask"] = mask
        return batch


# ════════════════════════════════════════════════════════════
# 9. MAIN MODEL
# ════════════════════════════════════════════════════════════

class ImprovedZINBWaterMeterEncoder(nn.Module):
    """
    ZINB water-meter encoder with 6 static features:
    {meter_type: 2, tariff_code: 72, is_urban: 2, region_in: 2, phase: 2, amper: 6}
    """

    def __init__(self, d_model=256, n_heads=8, n_layers=6,
                 static_cardinalities=None, dropout=0.1,
                 use_time_aware_attention=True,
                 default_rate=30.0):
        super().__init__()
        self.use_time_aware_attention = use_time_aware_attention

        # ── Static feature index constants ──
        self._idx_meter_type = 0
        self._idx_tariff_code = 1
        self._idx_is_urban = 2
        self._idx_region_in = 3
        self._idx_phase = 4
        self._idx_amper = 5

        card_list = list(static_cardinalities.values())
        # REMOVED: card_names — assigned but never read

        meter_type_card = card_list[self._idx_meter_type]
        tariff_card = card_list[self._idx_tariff_code]
        is_urban_card = card_list[self._idx_is_urban]
        region_in_card = card_list[self._idx_region_in]
        phase_card = card_list[self._idx_phase]
        amper_card = card_list[self._idx_amper]

        # ── Per-segment default rates ──
        self.default_rate_by_tariff = nn.Embedding(tariff_card, 1)
        self.default_rate_by_type = nn.Embedding(meter_type_card, 1)
        self.default_rate_by_urban = nn.Embedding(is_urban_card, 1)
        self.default_rate_by_region = nn.Embedding(region_in_card, 1)
        self.default_rate_by_phase = nn.Embedding(phase_card, 1)
        self.default_rate_by_amper = nn.Embedding(amper_card, 1)

        nn.init.constant_(self.default_rate_by_tariff.weight, math.log(default_rate))
        nn.init.constant_(self.default_rate_by_type.weight, 0.0)
        nn.init.constant_(self.default_rate_by_urban.weight, 0.0)
        nn.init.constant_(self.default_rate_by_region.weight, 0.0)
        nn.init.constant_(self.default_rate_by_phase.weight, 0.0)
        nn.init.constant_(self.default_rate_by_amper.weight, 0.0)

        # ── Per-segment baseline blend weights ──
        self.baseline_blend_tariff = nn.Embedding(tariff_card, 2)
        self.baseline_blend_urban = nn.Embedding(is_urban_card, 2)
        with torch.no_grad():
            self.baseline_blend_tariff.weight[:, 0] = 0.6
            self.baseline_blend_tariff.weight[:, 1] = 0.4
            nn.init.constant_(self.baseline_blend_urban.weight, 0.0)

        self.tariff_output_log_scale = nn.Embedding(tariff_card, 1)
        self.meter_type_output_log_scale = nn.Embedding(meter_type_card, 1)
        nn.init.zeros_(self.tariff_output_log_scale.weight)
        nn.init.zeros_(self.meter_type_output_log_scale.weight)

        # Per-segment gate bias: different segments have different zero rates
        self.segment_gate_bias_tariff = nn.Embedding(tariff_card, 1)
        self.segment_gate_bias_urban = nn.Embedding(is_urban_card, 1)
        nn.init.zeros_(self.segment_gate_bias_tariff.weight)
        nn.init.zeros_(self.segment_gate_bias_urban.weight)

        # Per-segment dispersion bias
        self.segment_alpha_bias = nn.Embedding(tariff_card, 1)
        nn.init.zeros_(self.segment_alpha_bias.weight)

        # ── Static context for conditioning output heads ──
        n_static = len(static_cardinalities)
        d_static_ctx = d_model // 4

        self.static_head_embeddings = nn.ModuleList([
            nn.Embedding(card, d_static_ctx // n_static)
            for card in static_cardinalities.values()
        ])
        static_raw_dim = n_static * (d_static_ctx // n_static)
        self.static_ctx_proj = nn.Sequential(
            nn.Linear(static_raw_dim, d_static_ctx),
            nn.LayerNorm(d_static_ctx),
            nn.SiLU(),
        )

        # ── Conditioned heads ──
        head_input_dim = d_model + d_static_ctx

        self.emb = InputEmbedding(
            d_model, static_cardinalities,
            n_value_features=7,
            dt_index=6,
            dropout=dropout,
        )

        if use_time_aware_attention:
            self.encoder = TimeAwareTransformerEncoder(
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

        self.correction_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.scale_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus(),
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.gate_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self._init_heads()

    def _get_default_rate(self, static):
        log_rate = (
            self.default_rate_by_tariff(static[:, self._idx_tariff_code]).squeeze(-1)
            + self.default_rate_by_type(static[:, self._idx_meter_type]).squeeze(-1)
            + self.default_rate_by_urban(static[:, self._idx_is_urban]).squeeze(-1)
            + self.default_rate_by_region(static[:, self._idx_region_in]).squeeze(-1)
            + self.default_rate_by_phase(static[:, self._idx_phase]).squeeze(-1)
            + self.default_rate_by_amper(static[:, self._idx_amper]).squeeze(-1)
        )
        return F.softplus(log_rate)

    def _get_baseline_blend(self, static):
        raw_weights = (
            self.baseline_blend_tariff(static[:, self._idx_tariff_code])
            + self.baseline_blend_urban(static[:, self._idx_is_urban])
        )
        weights = F.softmax(raw_weights, dim=-1)
        return weights[:, 0], weights[:, 1]

    def _get_static_ctx(self, static, L):
        embs = [emb(static[:, i]) for i, emb in enumerate(self.static_head_embeddings)]
        ctx = self.static_ctx_proj(torch.cat(embs, dim=-1))
        return ctx.unsqueeze(1).expand(-1, L, -1)

    def _build_features(self, daily_rate, dt_safe, mask, static):
        B, L = daily_rate.shape
        device = daily_rate.device

        default = self._get_default_rate(static)
        default_expanded = default.unsqueeze(1)

        lag1 = torch.zeros(B, L, device=device)
        lag1[:, 1:] = daily_rate[:, :-1]
        lag1[:, 0] = default

        lag2 = torch.zeros(B, L, device=device)
        lag2[:, 2:] = daily_rate[:, :-2]
        lag2[:, :2] = default_expanded.expand(-1, 2)

        alpha = 0.3
        ema = torch.zeros(B, L, device=device)
        ema[:, 0] = default
        for t in range(1, L):
            valid_prev = mask[:, t - 1].float()
            ema[:, t] = (
                valid_prev * (alpha * daily_rate[:, t - 1] + (1 - alpha) * ema[:, t - 1])
                + (1 - valid_prev) * ema[:, t - 1]
            )

        masked_rate = daily_rate * mask.float()
        cumsum = masked_rate.cumsum(dim=1)
        cumsum_sq = (masked_rate ** 2).cumsum(dim=1)
        cumcount = mask.float().cumsum(dim=1).clamp(min=1)
        running_mean = cumsum / cumcount
        running_var = (cumsum_sq / cumcount - running_mean ** 2).clamp(min=0)
        running_std = running_var.sqrt()
        shifted_std = torch.zeros_like(running_std)
        shifted_std[:, 1:] = running_std[:, :-1]

        rate_diff = lag1 - lag2

        shifted_mean = torch.zeros_like(running_mean)
        shifted_mean[:, 1:] = running_mean[:, :-1]
        shifted_mean[:, 0] = default

        norm_scale = shifted_mean.clamp(min=1.0)

        features = torch.stack([
            lag1,
            lag1 / norm_scale,
            lag2 / norm_scale,
            ema / norm_scale,
            shifted_std / norm_scale,
            rate_diff / norm_scale,
            dt_safe / 30.0,
        ], dim=-1)

        return features, lag1, ema

    def _build_baseline_rate(self, lag1_rate, ema_rate, mask, static):
        lag1_w, ema_w = self._get_baseline_blend(static)
        lag1_w = lag1_w.unsqueeze(1)
        ema_w = ema_w.unsqueeze(1)

        baseline_rate = lag1_w * lag1_rate + ema_w * ema_rate

        default = self._get_default_rate(static).unsqueeze(1)
        is_cold = baseline_rate < 1e-6
        baseline_rate = torch.where(
            is_cold, default.expand_as(baseline_rate), baseline_rate,
        )
        return baseline_rate

    def _init_heads(self):
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        # CHANGED: start with small corrections for multiplicative mode
        nn.init.constant_(self.scale_head[-2].bias, -2.0)  # softplus(-2)≈0.13
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.constant_(self.alpha_head[-1].bias, 0.0)

    # REMOVED: `return_hidden` parameter — `hidden` is always in the output
    def forward(self, batch):
        values = batch["values"]
        mask = batch["mask"]
        static = batch["static"]
        times = batch.get("times", None)

        B, L, _ = values.shape

        dt = values[..., 1].float()
        target = values[..., 0].float()

        dt_safe = dt.clamp(min=0.5)
        target_safe = target.clamp(min=0.0)
        dt_safe = torch.where(mask, dt_safe, torch.ones_like(dt_safe))
        target_safe = torch.where(mask, target_safe, torch.zeros_like(target_safe))

        daily_rate = (target_safe / dt_safe).clamp(0.0, 1e4)

        features, lag1_rate, ema_rate = self._build_features(
            daily_rate, dt_safe, mask, static
        )
        baseline_rate = self._build_baseline_rate(
            lag1_rate, ema_rate, mask, static
        )
        baseline_mu = (baseline_rate * dt_safe).clamp(min=0.0)

        attention_times = times if times is not None else dt_safe.cumsum(dim=1)

        x = self.emb(features, static, mask,
                     absolute_times=attention_times, raw_dt=dt_safe)

        if self.use_time_aware_attention:
            h = self.encoder(x, attention_times, mask)
        else:
            causal = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )
            h = self.encoder(x, mask=causal, src_key_padding_mask=~mask)

        # ── Static-conditioned heads ──
        static_ctx = self._get_static_ctx(static, L)
        h_cond = torch.cat([h, static_ctx], dim=-1)

        # ═══════════════════════════════════════════════════════════
        # CHANGED: Multiplicative correction (scale-invariant)
        # Instead of: predicted = baseline + correction
        # Now:        predicted = baseline * (1 + factor)
        # A ±20% correction means the same thing for all segments
        # ═══════════════════════════════════════════════════════════
        raw_correction = self.correction_head(h_cond).squeeze(-1)
        scale = self.scale_head(h_cond).squeeze(-1).clamp(max=3.0)
        correction_factor = scale * torch.tanh(raw_correction)

        predicted_rate = (baseline_rate * (1.0 + correction_factor)).clamp(min=0.0)

        # ═══════════════════════════════════════════════════════════
        # NEW: Per-segment output scaling
        # Tariff + meter_type define the consumption MAGNITUDE
        # Learned log-scale initialized at 0 (no change), regularized
        # ═══════════════════════════════════════════════════════════
        tariff_ids = static[:, self._idx_tariff_code]
        meter_type_ids = static[:, self._idx_meter_type]

        output_log_scale = (
            self.tariff_output_log_scale(tariff_ids).squeeze(-1)
            + self.meter_type_output_log_scale(meter_type_ids).squeeze(-1)
        ).clamp(-3.0, 3.0)                        # scale ∈ [0.05, 20]
        output_scale = output_log_scale.exp()      # [B]

        mu = (
            predicted_rate * dt_safe * output_scale.unsqueeze(1)
        ).clamp(min=0.0, max=5e4)

        # ═══════════════════════════════════════════════════════════
        # CHANGED: Per-segment alpha with tariff bias
        # High-consumption segments need different dispersion
        # ═══════════════════════════════════════════════════════════
        raw_alpha = self.alpha_head(h_cond).squeeze(-1)
        seg_alpha_b = self.segment_alpha_bias(tariff_ids).squeeze(-1)
        raw_alpha = (raw_alpha + seg_alpha_b.unsqueeze(1)).clamp(-5.0, 5.0)
        alpha = (F.softplus(raw_alpha) + 0.05).clamp(min=0.05, max=20.0)

        # ═══════════════════════════════════════════════════════════
        # CHANGED: Per-segment gate with tariff + urban bias
        # Urban/rural and tariff determine zero-consumption likelihood
        # ═══════════════════════════════════════════════════════════
        raw_gate = self.gate_head(h_cond).squeeze(-1)
        seg_gate_b = (
            self.segment_gate_bias_tariff(tariff_ids).squeeze(-1)
            + self.segment_gate_bias_urban(
                static[:, self._idx_is_urban]
            ).squeeze(-1)
        )
        raw_gate = (raw_gate + seg_gate_b.unsqueeze(1)).clamp(-10.0, 10.0)
        gate = torch.sigmoid(raw_gate).clamp(min=1e-6, max=1 - 1e-6)

        # For backward-compatible diagnostics
        rate_correction = predicted_rate - baseline_rate

        return {
            "mu": mu,
            "alpha": alpha,
            "gate": gate,
            "gate_logits": raw_gate,
            "target": target_safe,
            "dt": dt_safe,
            "mask": mask,
            "times": attention_times,
            "hidden": h,
            "baseline_mu": baseline_mu,
            "baseline_rate": baseline_rate,
            "rate_correction": rate_correction,
            "predicted_rate": predicted_rate,
            "correction_factor": correction_factor,
            "output_scale": output_scale,
            "_static_for_loss": static,
        }


# ════════════════════════════════════════════════════════════
# 10. COMBINED LOSS
# ════════════════════════════════════════════════════════════

class ImprovedCombinedLoss(nn.Module):
    def __init__(
        self,
        raw_huber_weight=1.0,
        log_huber_weight=3.0,       # ← was 50.0
        zinb_weight=0.1,
        gate_bce_weight=3.0,        # ← was 100.0
        calibration_weight=2.0,     # ← NEW
        warmup_zinb_epoch=10,
        zinb_detach_epochs=20,
    ):
        super().__init__()
        self.raw_huber_weight = raw_huber_weight
        self.log_huber_weight = log_huber_weight
        self.zinb_weight = zinb_weight
        self.gate_bce_weight = gate_bce_weight
        self.calibration_weight = calibration_weight
        self.warmup_zinb_epoch = warmup_zinb_epoch
        self.zinb_detach_epochs = zinb_detach_epochs

        self.zinb_loss = ZINBLoss()

        self.raw_huber = nn.HuberLoss(reduction='none', delta=100.0)  # ← was 20.0
        self.log_huber = nn.HuberLoss(reduction='none', delta=1.5)    # ← was 1.0

    def forward(self, out, epoch=0, return_components=False):
        mu = out["mu"]
        gate = out["gate"]
        target = out["target"]
        mask = out["mask"]

        # ═══════════════════════════════════════════════════════
        # KEY CHANGE: predict with expected value (1-gate)*mu
        # This aligns training objective with evaluation metric
        # ═══════════════════════════════════════════════════════
        expected = (1 - gate) * mu

        components = {}

        if mask.any():
            pred_v = expected[mask]       # ← was mu[mask]
            tgt_v = target[mask]

            # ── Raw-space Huber ──
            raw_huber_vals = self.raw_huber(pred_v, tgt_v)

            # Per-TARIFF importance (not per-meter_type)
            # Tariff is the strongest segment signal
            static_flat = out.get("_static_for_loss", None)
            if static_flat is not None and static_flat.shape[0] == mu.shape[0]:
                tariff = static_flat[:, 1]
                tariff_expanded = tariff.unsqueeze(1).expand_as(mu)
                tc_flat = tariff_expanded[mask]

                importance = torch.ones_like(tgt_v)
                for tc in tariff.unique():
                    seg = tc_flat == tc
                    if seg.sum() > 3:
                        # Normalize by segment std → equal gradient per segment
                        seg_std = tgt_v[seg].std().clamp(min=10.0)
                        importance[seg] = 1.0 / seg_std
                importance = importance / importance.mean().clamp(min=1e-6)
                importance = importance.clamp(0.1, 5.0)
            else:
                importance = torch.ones_like(tgt_v)

            raw_loss = (raw_huber_vals * importance).mean()
            components["raw_huber"] = raw_loss

            # ── Log-space Huber ──
            log_pred = torch.log1p(pred_v)
            log_tgt = torch.log1p(tgt_v)
            log_loss = (self.log_huber(log_pred, log_tgt) * importance).mean()
            components["log_huber"] = log_loss

            # ════════════════════════════════════════════════
            # NEW: Calibration loss — fixes systematic bias
            # Penalizes when batch mean prediction ≠ mean target
            # ════════════════════════════════════════════════
            pred_mean = pred_v.mean()
            tgt_mean = tgt_v.mean().clamp(min=1.0)
            calibration = ((pred_mean - tgt_mean) / tgt_mean).pow(2)

            # Also per-segment calibration for top segments
            if static_flat is not None and static_flat.shape[0] == mu.shape[0]:
                seg_cal = torch.tensor(0.0, device=mu.device)
                seg_count = 0
                for tc in tariff.unique():
                    seg = tc_flat == tc
                    if seg.sum() > 10:
                        seg_pred_mean = pred_v[seg].mean()
                        seg_tgt_mean = tgt_v[seg].mean().clamp(min=1.0)
                        seg_cal = seg_cal + (
                            (seg_pred_mean - seg_tgt_mean) / seg_tgt_mean
                        ).pow(2)
                        seg_count += 1
                if seg_count > 0:
                    calibration = calibration + seg_cal / seg_count

            components["calibration"] = calibration

        else:
            raw_loss = mu.sum() * 0.0
            log_loss = mu.sum() * 0.0
            calibration = mu.sum() * 0.0
            components["raw_huber"] = raw_loss
            components["log_huber"] = log_loss
            components["calibration"] = calibration

        # ── Gate BCE (unchanged) ──
        if mask.any():
            is_zero = (target < 0.5).float()
            gate_loss = F.binary_cross_entropy_with_logits(
                out["gate_logits"][mask], is_zero[mask], reduction='mean',
            )
        else:
            gate_loss = mu.sum() * 0.0
        components["gate_bce"] = gate_loss

        # ── ZINB ──
        if epoch >= self.warmup_zinb_epoch:
            detach_end = self.warmup_zinb_epoch + self.zinb_detach_epochs
            zinb_mu = mu.detach() if epoch < detach_end else mu
            zinb = self.zinb_loss(zinb_mu, out["alpha"], out["gate"], target, mask)
            zinb_w = self.zinb_weight
        else:
            zinb = mu.sum() * 0.0
            zinb_w = 0.0
        components["zinb"] = zinb

        # ── Alpha warmstart ──
        if epoch < self.warmup_zinb_epoch and mask.any():
            pred_v_mu = mu[mask]
            tgt_v_ws = target[mask]
            log_var_target = torch.log(tgt_v_ws.var().clamp(min=1.0))
            log_var_pred = torch.log(
                (out["alpha"][mask] * pred_v_mu ** 2).mean().clamp(min=1.0)
            )
            alpha_warmstart = 0.01 * (log_var_pred - log_var_target) ** 2
        else:
            alpha_warmstart = mu.sum() * 0.0
        components["alpha_warmstart"] = alpha_warmstart

        if "baseline_mu" in out and mask.any():
            components["baseline_mae"] = (
                (out["baseline_mu"] - target).abs()[mask].mean().detach()
            )

        # ── Total ──
        total = (
            self.raw_huber_weight * raw_loss
            + self.log_huber_weight * log_loss
            + self.gate_bce_weight * gate_loss
            + self.calibration_weight * calibration
            + zinb_w * zinb
            + alpha_warmstart
        )

        if torch.isnan(total) or torch.isinf(total):
            total = mu.sum() * 0.0 + 10.0

        if return_components:
            return total, components
        return total


# ════════════════════════════════════════════════════════════
# 11. TRAINING LOOP
# ════════════════════════════════════════════════════════════

def train_one_epoch(
    model, loader, optimizer, loss_fn, device, epoch,
    grad_accumulator, ema=None, augmenter=None,
    max_grad_norm=1.0, log_interval=50,
    scaler=None,
):
    model.train()
    accum = defaultdict(float)
    batch_count = 0
    nan_count = 0
    total_batches = len(loader)

    MAX_STORED_BATCHES = 50
    all_preds = []
    all_targets = []
    all_gates = []
    all_masks = []
    stored_count = 0

    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        if augmenter is not None:
            batch = augmenter(batch)

        did_backward = False
        did_unscale = False

        try:
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(batch)
                loss, components = loss_fn(
                    out, epoch=epoch, return_components=True
                )

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count <= 5:
                    print(f"[WARNING] NaN/Inf loss at batch {batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaled_loss = grad_accumulator.scale_loss(loss)

            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            did_backward = True

            do_step = (
                grad_accumulator.should_step(batch_idx)
                or grad_accumulator.is_last_batch_in_epoch(
                    batch_idx, total_batches
                )
            )

            if do_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                    did_unscale = True

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )

                grad_ok = not (
                    torch.isnan(grad_norm) or torch.isinf(grad_norm)
                )

                if grad_ok:
                    if use_amp:
                        scaler.step(optimizer)
                    else:
                        optimizer.step()
                    if ema is not None:
                        ema.update()
                else:
                    nan_count += 1

                if use_amp:
                    scaler.update()

                optimizer.zero_grad(set_to_none=True)

            # Accumulate metrics
            accum["loss"] += loss.item()

            for k, v in components.items():
                if isinstance(v, torch.Tensor):
                    accum[k] += v.item()
                else:
                    accum[k] += float(v)

            with torch.no_grad():
                m = out["mask"]
                if m.any():
                    accum["mae"] += (
                        (out["mu"] - out["target"]).abs()[m].mean().item()
                    )
                    if stored_count < MAX_STORED_BATCHES and batch_idx % 5 == 0:
                        all_preds.append(out["mu"].detach().cpu())
                        all_targets.append(out["target"].detach().cpu())
                        all_gates.append(out["gate"].detach().cpu())
                        all_masks.append(out["mask"].detach().cpu())
                        stored_count += 1

            batch_count += 1

            # Periodic logging
            if batch_idx % log_interval == 0 and batch_count > 0:
                avg_loss = accum["loss"] / batch_count
                avg_mae = accum["mae"] / batch_count

                with torch.no_grad():
                    m = out["mask"]
                    if m.any():
                        mu_raw = out["mu"][m]
                        tgt_v = out["target"][m]

                        # CONSOLIDATED: was computed twice as expected_v and mu_v
                        mu_v = ((1 - out["gate"]) * out["mu"])[m]
                        gate_v = out["gate"][m]
                        ratio = mu_v.mean() / tgt_v.mean().clamp(min=1)

                        parts = (
                            f"  B {batch_idx:04d}/{total_batches} │ "
                            f"L:{avg_loss:.3f} MAE:{avg_mae:.0f} │ "
                            f"μ:{mu_raw.mean():.0f}±{mu_raw.std():.0f} "
                            f"E:{mu_v.mean():.0f} "
                            f"t:{tgt_v.mean():.0f}±{tgt_v.std():.0f} "
                            f"r:{ratio:.2f} │ "
                            f"g:{gate_v.mean():.3f} "
                            f"z:{(tgt_v < 0.5).float().mean():.3f}"
                        )

                        if "baseline_mu" in out:
                            bl_v = out["baseline_mu"][m]
                            bl_mae = (bl_v - tgt_v).abs().mean()
                            parts += f" │ bl:{bl_mae:.0f}"

                        if "rate_correction" in out:
                            rc = out["rate_correction"][m]
                            parts += f" rc:{rc.mean():.1f}±{rc.std():.1f}"

                        print(parts)

        except RuntimeError as e:
            print(f"[ERROR] Batch {batch_idx}: {e}")
            nan_count += 1

            if use_amp:
                if did_unscale:
                    scaler.update()

            optimizer.zero_grad(set_to_none=True)
            continue

    # ── End-of-epoch diagnostics ──
    if batch_count > 0:
        model.eval()
        with torch.no_grad():
            try:
                diag_batch = next(iter(loader))
                diag_batch = {
                    k: v.to(device) for k, v in diag_batch.items()
                }
                diag_out = model(diag_batch)
                m = diag_out["mask"]

                if m.any():
                    expected_mu = ((1 - diag_out["gate"]) * diag_out["mu"])
                    mu_v = expected_mu[m]
                    t_v = diag_out["target"][m]
                    gate_v = diag_out["gate"][m]

                    print(f"\n  ── Epoch {epoch} Diagnostics ──")
                    print(
                        f"  Target:    "
                        f"mean={t_v.mean():.0f}  "
                        f"std={t_v.std():.0f}  "
                        f"median={t_v.median():.0f}  "
                        f"[{t_v.min():.0f}, {t_v.max():.0f}]"
                    )

                    if "baseline_mu" in diag_out:
                        bl_v = diag_out["baseline_mu"][m]
                        print(
                            f"  Baseline:  "
                            f"mean={bl_v.mean():.0f}  "
                            f"MAE={(bl_v - t_v).abs().mean():.1f}"
                        )

                    print(
                        f"  Model μ:   "
                        f"mean={mu_v.mean():.0f}  "
                        f"MAE={(mu_v - t_v).abs().mean():.1f}"
                    )

                    if "rate_correction" in diag_out:
                        rc = diag_out["rate_correction"][m]
                        br = diag_out["baseline_rate"][m]
                        pr = diag_out["predicted_rate"][m]
                        print(
                            f"  Base rate: "
                            f"mean={br.mean():.1f}  "
                            f"std={br.std():.1f}"
                        )
                        print(
                            f"  Rate corr: "
                            f"mean={rc.mean():.1f}  "
                            f"std={rc.std():.1f}  "
                            f"abs_mean={rc.abs().mean():.1f}"
                        )
                        print(
                            f"  Pred rate: "
                            f"mean={pr.mean():.1f}  "
                            f"std={pr.std():.1f}"
                        )

                    print(f"  Gate:      mean={gate_v.mean():.3f}")
                    print(
                        f"  Zero rate: "
                        f"actual={(t_v < 0.5).float().mean():.3f}  "
                        f"predicted={(mu_v < 1.5).float().mean():.3f}"
                    )

                    buckets = [
                        ("low",  0.00, 0.25, False),
                        ("mid",  0.25, 0.75, False),
                        ("high", 0.75, 1.00, True),
                    ]
                    for q_name, q_lo, q_hi, inclusive_upper in buckets:
                        lo = torch.quantile(t_v, q_lo)
                        hi = torch.quantile(t_v, min(q_hi, 1.0))

                        if inclusive_upper:
                            sel = t_v >= lo
                        else:
                            sel = (t_v >= lo) & (t_v < hi)

                        if sel.any():
                            qmae = (mu_v[sel] - t_v[sel]).abs().mean()
                            print(
                                f"  MAE ({q_name:>4} "
                                f"[{lo:.0f}-{hi:.0f}]): "
                                f"{qmae:.1f}"
                            )

                    # Per-segment MAE breakdown
                    if "_static_for_loss" in diag_out:
                        static_diag = diag_out["_static_for_loss"]
                        meter_types = static_diag[:, 0]
                        tariff_codes = static_diag[:, 1]

                        print(f"\n  ── Per-segment breakdown ──")
                        for mt in meter_types.unique():
                            mt_mask_2d = (meter_types == mt).unsqueeze(1).expand_as(diag_out["mu"])
                            seg = m & mt_mask_2d
                            if seg.any():
                                seg_mae = (diag_out["mu"][seg] - diag_out["target"][seg]).abs().mean()
                                seg_mean = diag_out["target"][seg].mean()
                                seg_count = seg.sum()
                                print(f"    meter_type={mt.item()}: "
                                      f"MAE={seg_mae:.1f}  "
                                      f"mean_target={seg_mean:.0f}  "
                                      f"n={seg_count}")

                        tariff_expanded = tariff_codes.unsqueeze(1).expand_as(diag_out["mu"])
                        tariff_flat = tariff_expanded[m]
                        target_flat = diag_out["target"][m]
                        pred_flat = diag_out["mu"][m]
                        unique_tariffs, counts = tariff_flat.unique(return_counts=True)
                        top_k = min(5, len(unique_tariffs))
                        top_idx = counts.argsort(descending=True)[:top_k]
                        print(f"    Top {top_k} tariff codes:")
                        for idx in top_idx:
                            tc = unique_tariffs[idx].item()
                            tc_mask = tariff_flat == tc
                            tc_mae = (pred_flat[tc_mask] - target_flat[tc_mask]).abs().mean()
                            tc_mean = target_flat[tc_mask].mean()
                            print(f"      tariff={tc:2d}: "
                                  f"MAE={tc_mae:.1f}  "
                                  f"mean={tc_mean:.0f}  "
                                  f"n={tc_mask.sum()}")

                    # Rate-space metrics
                    if "predicted_rate" in diag_out and "baseline_rate" in diag_out:
                        target_rate = (t_v / diag_out["dt"][m]).clamp(min=0)
                        pred_rate = diag_out["predicted_rate"][m]
                        base_rate = diag_out["baseline_rate"][m]
                        rate_corr = diag_out["rate_correction"][m]

                        rate_mae = (pred_rate - target_rate).abs().mean()
                        base_rate_mae = (base_rate - target_rate).abs().mean()
                        improvement = (1 - rate_mae / base_rate_mae.clamp(min=0.1)) * 100

                        print(f"\n  ── Rate-space metrics ──")
                        print(f"  Rate MAE:     baseline={base_rate_mae:.2f}  "
                              f"model={rate_mae:.2f}  "
                              f"improvement={improvement:.1f}%")
                        print(f"  Correction:   mean={rate_corr.mean():.2f}  "
                              f"std={rate_corr.std():.2f}  "
                              f"|mean|={rate_corr.abs().mean():.2f}")
                        print(f"  Correction/base: "
                              f"{(rate_corr.abs() / base_rate.clamp(min=0.1)).mean():.1%}")

                    print()

            except Exception as e:
                print(f"  [WARNING] Diagnostics failed: {e}\n")

    # Epoch-level aggregate metrics
    avg = {k: v / max(batch_count, 1) for k, v in accum.items()}
    avg["nan_batches"] = nan_count

    if all_preds:
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        gates = torch.cat(all_gates, dim=0)
        masks = torch.cat(all_masks, dim=0)

        vp = preds[masks]
        vt = targets[masks]
        vg = gates[masks]

        avg["rmse"] = ((vp - vt) ** 2).mean().sqrt().item()
        avg["mape"] = (
            (vp - vt).abs() / vt.clamp(min=1)
        ).mean().item() * 100
        avg["log_mae"] = (
            (torch.log1p(vp) - torch.log1p(vt)).abs().mean().item()
        )
        avg["avg_gate"] = vg.mean().item()
        avg["actual_zero_rate"] = (vt < 0.5).float().mean().item()

        if vp.std() > 1e-6 and vt.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([vp, vt]))[0, 1]
            avg["correlation"] = (
                corr.item() if not torch.isnan(corr) else 0.0
            )
        else:
            avg["correlation"] = 0.0

    return avg


# ════════════════════════════════════════════════════════════
# 12. VALIDATION LOOP
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch,
             use_ema=False, ema=None):
    model.eval()
    if use_ema and ema is not None:
        ema.apply_shadow()

    metrics = WaterMeterMetrics()
    total_loss = 0.0
    component_accum = defaultdict(float)
    batch_count = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch)
        loss, components = loss_fn(
            out, epoch=epoch, return_components=True
        )

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()

            for comp_name, comp_val in components.items():
                if isinstance(comp_val, torch.Tensor):
                    component_accum[comp_name] += comp_val.item()
                else:
                    component_accum[comp_name] += float(comp_val)

            metrics.update(out)
            batch_count += 1

    if use_ema and ema is not None:
        ema.restore()

    computed = metrics.compute()
    computed["loss"] = total_loss / max(batch_count, 1)

    for comp_name, comp_val in component_accum.items():
        key = f"loss_{comp_name}" if comp_name in computed else comp_name
        computed[key] = comp_val / max(batch_count, 1)

    return computed


# ════════════════════════════════════════════════════════════
# 13. SANITY CHECK
# ════════════════════════════════════════════════════════════

def sanity_check(model, train_loader, device):
    """Run before training to verify model can learn."""
    original_state = {
        k: v.cpu().clone() for k, v in model.state_dict().items()
    }

    model.to(device)
    model.eval()

    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        out = model(batch)
        m = out["mask"]

        if not m.any():
            print("[SANITY] No valid positions in batch — check data!")
            model.load_state_dict(original_state)
            return

        target_mean = out["target"][m].mean()
        target_std = out["target"][m].std()
        model_mae = (out["mu"] - out["target"]).abs()[m].mean()

        print(f"  Target: mean={target_mean:.0f}, std={target_std:.0f}")
        print(f"  Initial Model MAE: {model_mae:.1f}")

        if "baseline_mu" in out:
            baseline_mae = (
                (out["baseline_mu"] - out["target"]).abs()[m].mean()
            )
            print(f"  Baseline MAE: {baseline_mae:.1f}")
            if baseline_mae > 2 * target_std:
                print("  ⚠ Baseline MAE > 2×target_std — check lag logic")
        else:
            print("  (No baseline in model output)")

    # Overfit one batch
    model.train()
    temp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    temp_loss_fn = ImprovedCombinedLoss(
        raw_huber_weight=1.0,
        log_huber_weight=0.1,
        zinb_weight=0.0,
        gate_bce_weight=0.0,
        warmup_zinb_epoch=9999,
    )

    print("\n  Overfitting one batch (MAE should decrease):")
    prev_mae = float("inf")
    decreasing = True

    for step in range(100):
        temp_optim.zero_grad()
        out = model(batch)
        loss, _ = temp_loss_fn(out, return_components=True)

        if torch.isnan(loss):
            print(f"  ✗ NaN loss at step {step} — check model")
            decreasing = False
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        temp_optim.step()

        if step % 20 == 0:
            with torch.no_grad():
                mae = (
                    (out["mu"] - out["target"]).abs()[out["mask"]].mean()
                )
            print(f"    Step {step:3d}: loss={loss.item():.4f}, MAE={mae:.1f}")

            if step > 0 and mae > prev_mae * 1.1:
                print("  ⚠ MAE not consistently decreasing")
            prev_mae = mae.item()

    if decreasing:
        print("  ✓ Model can learn — proceed to full training.\n")
    else:
        print("  ✗ Model may have issues — review architecture.\n")

    model.load_state_dict(original_state)
    model.to(device)
    print("  ✓ Model state restored to original initialization.\n")


# ════════════════════════════════════════════════════════════
# 14. MAIN TRAINING ENTRYPOINT
# ════════════════════════════════════════════════════════════

def main_training_loop():
    config = {
        "data_path": r"D:\program\python\abfa_tracker\ODE\meters_latent_ode_ready.npz",
        "cats_path": r"D:\program\python\abfa_tracker\ODE\static_cardinalities.json",
        "checkpoint_dir": "checkpoints_v4",

        "d_model": 192,
        "n_heads": 6,
        "n_layers": 5,
        "dropout": 0.05,
        "use_time_aware_attention": True,

        "batch_size": 256,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "max_epochs": 500,
        "warmup_epochs": 5,
        "patience": 60,
        "max_grad_norm": 1.0,

        # ── CHANGED loss weights ──
        "raw_huber_weight": 1.0,  # keep
        "log_huber_weight": 3.0,  # ← was 50.0
        "gate_bce_weight": 3.0,  # ← was 100.0
        "calibration_weight": 2.0,  # ← NEW
        "zinb_weight": 0.1,
        "warmup_zinb_epoch": 20,

        "use_augmentation": False,
        "use_ema": True,
        "ema_decay": 0.995,

        "num_workers": 4,
        "seed": 42,
        "log_interval": 500,
        "run_sanity_check": True,

        "static_features": {
            "meter_type": 2,
            "tariff_code": 72,
            "is_urban": 2,
            "region_in": 2,
            "phase": 2,
            "amper": 6,
        },
    }

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_amp = device.type == "cuda"
    if use_amp:
        try:
            scaler = torch.amp.GradScaler('cuda')
        except TypeError:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    with open(config["cats_path"]) as f:
        static_cardinalities = json.load(f, object_pairs_hook=OrderedDict)

    expected = config["static_features"]
    print(f"Static features from file: {dict(static_cardinalities)}")
    print(f"Expected static features:  {expected}")
    for feat_name in expected:
        if feat_name not in static_cardinalities:
            print(f"  ⚠ WARNING: Expected feature '{feat_name}' not found in cardinalities file!")
        elif static_cardinalities[feat_name] != expected[feat_name]:
            print(f"  ⚠ WARNING: Feature '{feat_name}' cardinality mismatch: "
                  f"file={static_cardinalities[feat_name]} vs expected={expected[feat_name]}")

    full_dataset = ImprovedWaterMeterDataset(
        config["data_path"],
        static_cardinalities=static_cardinalities,
    )
    print(f"Dataset size: {len(full_dataset)}")
    print(f"Rate stats: mean={full_dataset.rate_mean:.1f}, "
          f"std={full_dataset.rate_std:.1f}")

    split_path = os.path.join(config["checkpoint_dir"], "data_split.pt")
    if os.path.exists(split_path):
        split = torch.load(split_path, weights_only=True)
    else:
        n = len(full_dataset)
        indices = np.random.permutation(n)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        split = {
            "train_indices": indices[:train_end].tolist(),
            "val_indices": indices[train_end:val_end].tolist(),
            "test_indices": indices[val_end:].tolist(),
        }
        torch.save(split, split_path)

    train_dataset = Subset(full_dataset, split["train_indices"])
    val_dataset = Subset(full_dataset, split["val_indices"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=config["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=config["num_workers"],
        pin_memory=True,
    )

    model = ImprovedZINBWaterMeterEncoder(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        static_cardinalities=static_cardinalities,
        dropout=config["dropout"],
        use_time_aware_attention=config["use_time_aware_attention"],
        default_rate=full_dataset.rate_mean,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if config["run_sanity_check"]:
        print("\n" + "─" * 60)
        print("SANITY CHECK")
        print("─" * 60)
        sanity_check(model, train_loader, device)

    embed_params = list(model.emb.parameters())
    encoder_params = list(model.encoder.parameters())

    # All static conditioning params — existing + NEW segment output params
    static_params = (
        list(model.static_head_embeddings.parameters())
        + list(model.static_ctx_proj.parameters())
        + list(model.default_rate_by_tariff.parameters())
        + list(model.default_rate_by_type.parameters())
        + list(model.default_rate_by_urban.parameters())
        + list(model.default_rate_by_region.parameters())
        + list(model.default_rate_by_phase.parameters())
        + list(model.default_rate_by_amper.parameters())
        + list(model.baseline_blend_tariff.parameters())
        + list(model.baseline_blend_urban.parameters())
    )

    # NEW: segment output params — separate group with mild weight_decay
    # to regularize rare tariff codes toward scale=1 (log_scale=0)
    segment_output_params = (
        list(model.tariff_output_log_scale.parameters())
        + list(model.meter_type_output_log_scale.parameters())
        + list(model.segment_gate_bias_tariff.parameters())
        + list(model.segment_gate_bias_urban.parameters())
        + list(model.segment_alpha_bias.parameters())
    )

    head_params = (
        list(model.correction_head.parameters())
        + list(model.scale_head.parameters())
        + list(model.alpha_head.parameters())
        + list(model.gate_head.parameters())
    )

    optimizer = torch.optim.AdamW([
        {"params": embed_params, "lr": config["learning_rate"]},
        {"params": encoder_params, "lr": config["learning_rate"]},
        {"params": static_params, "lr": config["learning_rate"] * 3,
         "weight_decay": 0.0},
        {"params": segment_output_params,                           # NEW group
         "lr": config["learning_rate"] * 5,                         # learn fast
         "weight_decay": 0.01},                                     # regularize rare tariffs
        {"params": head_params, "lr": config["learning_rate"] * 3},
    ], weight_decay=config["weight_decay"], betas=(0.9, 0.98), eps=1e-8)

    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=config["warmup_epochs"],
        total_epochs=config["max_epochs"],
        warmup_lr=1e-7, min_lr=1e-6,
    )

    loss_fn = ImprovedCombinedLoss(
        raw_huber_weight=config["raw_huber_weight"],
        log_huber_weight=config["log_huber_weight"],
        zinb_weight=config["zinb_weight"],
        gate_bce_weight=config["gate_bce_weight"],
        calibration_weight=config["calibration_weight"],     # NEW
        warmup_zinb_epoch=config["warmup_zinb_epoch"],
        zinb_detach_epochs=20,
    )

    early_stopping = EarlyStopping(
        patience=config["patience"], mode='min', restore_best=True
    )
    grad_accumulator = GradientAccumulator(
        config["gradient_accumulation_steps"]
    )
    ema = (
        ExponentialMovingAverage(model, config["ema_decay"])
        if config["use_ema"] else None
    )

    augmenter = (
        TimeSeriesAugmenter(
            noise_std=0.05,
            scale_range=(0.9, 1.1),
            mask_prob=0.1,
        ) if config["use_augmentation"] else None
    )

    logger = TrainingLogger(config["checkpoint_dir"], "run_001")
    logger.log_config(config)

    print("\n" + "═" * 80)
    print("STARTING TRAINING")
    print("═" * 80 + "\n")

    for epoch in range(config["max_epochs"]):
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            grad_accumulator=grad_accumulator,
            ema=ema,
            augmenter=augmenter,
            max_grad_norm=config["max_grad_norm"],
            log_interval=50 if epoch < 5 else config["log_interval"],
            scaler=scaler,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            use_ema=config["use_ema"],
            ema=ema,
        )

        train_metrics["lr"] = current_lr
        logger.log(train_metrics, epoch, "train")
        logger.log(val_metrics, epoch, "val")
        logger.log_scalar("lr", current_lr, epoch)
        logger.print_summary(epoch, train_metrics, val_metrics, current_lr)

        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": config,
            }, os.path.join(
                config["checkpoint_dir"],
                f"checkpoint_epoch_{epoch:03d}.pt",
            ))

        val_score = val_metrics["mae"]
        should_stop = early_stopping(val_score, model, epoch)

        if early_stopping.improved:
            torch.save(
                model.state_dict(),
                os.path.join(config["checkpoint_dir"], "best_model.pt"),
            )
            if ema is not None:
                ema.apply_shadow()
                torch.save(
                    model.state_dict(),
                    os.path.join(config["checkpoint_dir"], "best_model_ema.pt"),
                )
                ema.restore()
            print(f"  ✓ New best! MAE: {val_score:.4f}")

        if should_stop:
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            print(
                f"[INFO] Best: {early_stopping.best_score:.4f} "
                f"at epoch {early_stopping.best_epoch}"
            )
            break

    logger.save_history()
    elapsed = logger.get_elapsed_time()
    print("\n" + "═" * 80)
    print("TRAINING COMPLETE")
    print(f"  Best Validation MAE : {early_stopping.best_score:.4f}")
    print(f"  Best Epoch          : {early_stopping.best_epoch}")
    print(f"  Total Time          : {elapsed / 3600:.2f} hours")
    print("═" * 80)

    return model, logger


if __name__ == "__main__":
    model, logger = main_training_loop()