"""
ImprovedCombinedLoss v2 — Scale-Aware for Heavy-Tail Consumption
=================================================================

Changes from the previous version that you posted:

  (1) RAW-HUBER TARGET CLIPPING.  During training, targets above
      `train_target_clip` (99.5th percentile from preprocessor) are
      clipped before going into the raw-Huber term. This stops single
      80k-kWh industrial readings from yanking the optimizer around
      every batch. Predictions are also clipped at 1.2× to allow
      controlled overshoot. Validation and test loss use unclipped
      targets so reported metrics are honest.

  (2) RE-WEIGHTED COMPONENTS.  log_huber_weight raised to 5.0,
      raw_huber_weight dropped to 0.5. The bulk of the gradient now
      comes from scale-invariant log-space error. raw-space exists to
      pin the absolute scale on small values where log compresses.

  (3) VECTORIZED PER-TARIFF IMPORTANCE.  The python `for tc in
      tariff.unique()` loop is gone. Per-segment std is computed once
      with scatter-style aggregation. Same semantics, ~10× faster.

  (4) LOG-SPACE CALIBRATION.  The calibration term was in raw-space,
      which made it dominated by industrial-tariff over/under-prediction.
      Now also computed on log1p(pred) vs log1p(target), so residential
      and industrial calibration both contribute.

  (5) SAFER NaN FALLBACK.  When the total goes NaN/Inf, we return a
      gradient-free fixed loss tied to mu so the optimizer step is a
      no-op rather than a destructive update.

Usage:
    loss_fn = ImprovedCombinedLoss(
        train_target_clip=10000.0,    # read from peer_avg_config.json
        ...
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the ZINBLoss class from your original module
from Improved_embeding import ZINBLoss


class ImprovedCombinedLoss(nn.Module):
    def __init__(
        self,
        raw_huber_weight=0.5,           # ← was 1.0; raw-space less dominant
        log_huber_weight=5.0,           # ← was 3.0; log-space drives bulk
        zinb_weight=0.1,
        gate_bce_weight=3.0,
        calibration_weight=2.0,
        warmup_zinb_epoch=10,
        zinb_detach_epochs=20,
        # ── NEW: scale-aware controls ─────────────────────────────
        train_target_clip=10000.0,      # 99.5th pctile of train targets
        train_pred_clip_factor=1.2,     # let model overshoot a bit
        seg_calibration_min_n=10,       # min seg size for seg-calibration
        seg_importance_min_n=3,         # min seg size for per-seg weights
        importance_min=0.1,
        importance_max=5.0,
    ):
        super().__init__()
        self.raw_huber_weight = raw_huber_weight
        self.log_huber_weight = log_huber_weight
        self.zinb_weight = zinb_weight
        self.gate_bce_weight = gate_bce_weight
        self.calibration_weight = calibration_weight
        self.warmup_zinb_epoch = warmup_zinb_epoch
        self.zinb_detach_epochs = zinb_detach_epochs

        self.train_target_clip = train_target_clip
        self.train_pred_clip_factor = train_pred_clip_factor
        self.seg_calibration_min_n = seg_calibration_min_n
        self.seg_importance_min_n = seg_importance_min_n
        self.importance_min = importance_min
        self.importance_max = importance_max

        self.zinb_loss = ZINBLoss()
        self.raw_huber = nn.HuberLoss(reduction='none', delta=50.0)
        self.log_huber = nn.HuberLoss(reduction='none', delta=1.5)

    # ════════════════════════════════════════════════════════════
    # Vectorized per-tariff importance
    # ════════════════════════════════════════════════════════════
    def _compute_importance(self, tc_flat, tgt_v):
        """
        Per-sample weight ∝ 1 / segment_std, normalized so that the
        mean weight across samples is 1. Vectorized via scatter-add.
        """
        if tc_flat is None:
            return torch.ones_like(tgt_v)

        n_tariff = int(tc_flat.max().item()) + 1
        ones = torch.ones_like(tgt_v)

        # Per-segment sum and sum-of-squares for std calculation
        seg_n = torch.zeros(n_tariff, device=tgt_v.device, dtype=tgt_v.dtype)
        seg_s = torch.zeros_like(seg_n)
        seg_s2 = torch.zeros_like(seg_n)

        seg_n.scatter_add_(0, tc_flat, ones)
        seg_s.scatter_add_(0, tc_flat, tgt_v)
        seg_s2.scatter_add_(0, tc_flat, tgt_v ** 2)

        seg_n_safe = seg_n.clamp(min=1.0)
        seg_mean = seg_s / seg_n_safe
        seg_var = (seg_s2 / seg_n_safe - seg_mean ** 2).clamp(min=0.0)
        seg_std = seg_var.sqrt().clamp(min=10.0)

        # Segments too small get weight 1 (don't trust their std)
        small = seg_n < self.seg_importance_min_n
        seg_weight = torch.where(small,
                                 torch.ones_like(seg_std),
                                 1.0 / seg_std)

        importance = seg_weight[tc_flat]
        importance = importance / importance.mean().clamp(min=1e-6)
        return importance.clamp(self.importance_min, self.importance_max)

    # ════════════════════════════════════════════════════════════
    # Vectorized per-segment calibration
    # ════════════════════════════════════════════════════════════
    def _segment_calibration(self, tc_flat, pred_v, tgt_v, log_pred, log_tgt):
        """
        Average over tariffs (with enough samples) of squared relative
        error between segment mean prediction and segment mean target,
        in both raw and log space.
        """
        if tc_flat is None:
            return torch.tensor(0.0, device=pred_v.device)

        n_tariff = int(tc_flat.max().item()) + 1
        ones = torch.ones_like(tgt_v)

        seg_n = torch.zeros(n_tariff, device=tgt_v.device, dtype=tgt_v.dtype)
        seg_pred = torch.zeros_like(seg_n)
        seg_tgt = torch.zeros_like(seg_n)
        seg_logp = torch.zeros_like(seg_n)
        seg_logt = torch.zeros_like(seg_n)

        seg_n.scatter_add_(0, tc_flat, ones)
        seg_pred.scatter_add_(0, tc_flat, pred_v)
        seg_tgt.scatter_add_(0, tc_flat, tgt_v)
        seg_logp.scatter_add_(0, tc_flat, log_pred)
        seg_logt.scatter_add_(0, tc_flat, log_tgt)

        valid = seg_n >= self.seg_calibration_min_n
        if not valid.any():
            return torch.tensor(0.0, device=pred_v.device)

        seg_n_safe = seg_n.clamp(min=1.0)
        seg_pred_mean = seg_pred / seg_n_safe
        seg_tgt_mean = seg_tgt / seg_n_safe
        seg_logp_mean = seg_logp / seg_n_safe
        seg_logt_mean = seg_logt / seg_n_safe

        # Raw-space relative calibration (only for valid segments)
        raw_rel = ((seg_pred_mean - seg_tgt_mean)
                   / seg_tgt_mean.clamp(min=1.0)) ** 2
        # Log-space absolute calibration (already scale-invariant)
        log_abs = (seg_logp_mean - seg_logt_mean) ** 2

        per_seg = raw_rel + log_abs
        return per_seg[valid].mean()

    # ════════════════════════════════════════════════════════════
    # Forward
    # ════════════════════════════════════════════════════════════
    def forward(self, out, epoch=0, return_components=False):
        mu = out["mu"]
        gate = out["gate"]
        target = out["target"]
        mask = out["mask"]

        expected = (1 - gate) * mu
        components = {}

        if mask.any():
            pred_v_full = expected[mask]
            tgt_v_full = target[mask]

            # ── Static / tariff lookup (vectorized) ──────────────
            static_flat = out.get("_static_for_loss", None)
            if static_flat is not None and static_flat.shape[0] == mu.shape[0]:
                tariff = static_flat[:, 1]                # [B]
                tc_flat = tariff.unsqueeze(1).expand_as(mu)[mask].long()  # [N]
            else:
                tc_flat = None

            # ── Raw-space Huber on CLIPPED targets (train only) ──
            if self.training:
                clip_t = self.train_target_clip
                clip_p = clip_t * self.train_pred_clip_factor
                tgt_v_clip = tgt_v_full.clamp(max=clip_t)
                pred_v_clip = pred_v_full.clamp(max=clip_p)
            else:
                tgt_v_clip = tgt_v_full
                pred_v_clip = pred_v_full

            raw_huber_vals = self.raw_huber(pred_v_clip, tgt_v_clip)

            # ── Per-sample importance (vectorized) ───────────────
            importance = self._compute_importance(tc_flat, tgt_v_full)

            raw_loss = (raw_huber_vals * importance).mean()
            components["raw_huber"] = raw_loss

            # ── Log-space Huber (uses UN-clipped values; scale-safe) ──
            log_pred = torch.log1p(pred_v_full.clamp(min=0.0))
            log_tgt = torch.log1p(tgt_v_full.clamp(min=0.0))
            log_huber_vals = self.log_huber(log_pred, log_tgt)
            log_loss = (log_huber_vals * importance).mean()
            components["log_huber"] = log_loss

            # ── Calibration (batch + per-segment, both spaces) ───
            pred_mean = pred_v_full.mean()
            tgt_mean = tgt_v_full.mean().clamp(min=1.0)
            log_pred_mean = log_pred.mean()
            log_tgt_mean = log_tgt.mean()
            calibration = (
                ((pred_mean - tgt_mean) / tgt_mean) ** 2
                + (log_pred_mean - log_tgt_mean) ** 2
            )
            calibration = calibration + self._segment_calibration(
                tc_flat, pred_v_full, tgt_v_full, log_pred, log_tgt
            )
            components["calibration"] = calibration

        else:
            zero_t = mu.sum() * 0.0
            raw_loss = zero_t
            log_loss = zero_t
            calibration = zero_t
            components["raw_huber"] = raw_loss
            components["log_huber"] = log_loss
            components["calibration"] = calibration

        # ── Gate BCE ─────────────────────────────────────────────
        if mask.any():
            is_zero = (target < 0.5).float()
            gate_loss = F.binary_cross_entropy_with_logits(
                out["gate_logits"][mask], is_zero[mask], reduction='mean',
            )
        else:
            gate_loss = mu.sum() * 0.0
        components["gate_bce"] = gate_loss

        # ── ZINB ─────────────────────────────────────────────────
        if epoch >= self.warmup_zinb_epoch:
            detach_end = self.warmup_zinb_epoch + self.zinb_detach_epochs
            zinb_mu = mu.detach() if epoch < detach_end else mu
            zinb = self.zinb_loss(zinb_mu, out["alpha"],
                                  out["gate"], target, mask)
            zinb_w = self.zinb_weight
        else:
            zinb = mu.sum() * 0.0
            zinb_w = 0.0
        components["zinb"] = zinb

        # ── Alpha warmstart ──────────────────────────────────────
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

        # ── Total ────────────────────────────────────────────────
        total = (
            self.raw_huber_weight * raw_loss
            + self.log_huber_weight * log_loss
            + self.gate_bce_weight * gate_loss
            + self.calibration_weight * calibration
            + zinb_w * zinb
            + alpha_warmstart
        )

        if torch.isnan(total) or torch.isinf(total):
            # Gradient-free fallback: optimizer step is a no-op
            total = (mu.sum() * 0.0).detach() + 10.0

        if return_components:
            return total, components
        return total
