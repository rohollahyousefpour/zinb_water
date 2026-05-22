"""
Extra patches v2 — variable-L safe metrics + proper ZINB zero classification
=============================================================================
Replaces the v1 extra_patches.py.

Changes vs v1:
  (1) FixedElectricityMeterMetrics now also tracks alpha and mu separately,
      so it can compute the true ZINB P(Y=0) = π + (1-π)(1+αμ)^(-1/α).
      Classification rule: "predicted zero" iff P(Y=0) > 0.5.
      This fixes the zero_f1 = 0 you saw at epoch 0, which was a
      threshold artifact, not a model failure.
  (2) Print message suppressed in DataLoader worker processes (you
      were seeing 4 copies per epoch — only the main process prints now).
"""

import multiprocessing as _mp
from typing import Optional

import torch

import Improved_embeding as orig
import electricity_zinb_patches as p


# ════════════════════════════════════════════════════════════
# Fixed metrics — variable-L safe AND proper zero classifier
# ════════════════════════════════════════════════════════════
class FixedElectricityMeterMetrics:
    """
    Variable-length-safe (stores 1-D flattened tensors), and uses the
    full ZINB P(Y=0) for zero classification rather than a hard
    threshold on gate alone.
    """

    def __init__(self, zero_prob_threshold: float = 0.5):
        self.zero_prob_threshold = zero_prob_threshold
        self.reset()

    def reset(self):
        self.targets   = []
        self.mus       = []
        self.alphas    = []
        self.gates     = []
        self.expecteds = []
        self.dts       = []

    @torch.no_grad()
    def update(self, out):
        m = out["mask"]
        if not m.any():
            return
        mu    = out["mu"][m]
        alpha = out["alpha"][m]
        gate  = out["gate"][m]
        tgt   = out["target"][m]
        dt    = out["dt"][m]
        expected = (1 - gate) * mu

        self.mus.append(mu.cpu())
        self.alphas.append(alpha.cpu())
        self.gates.append(gate.cpu())
        self.targets.append(tgt.cpu())
        self.expecteds.append(expected.cpu())
        self.dts.append(dt.cpu())

    @staticmethod
    def _zinb_prob_zero(mu, alpha, gate, eps=1e-8):
        """P(Y = 0) = π + (1-π) * (1 + α μ)^(-1/α). Numerically stable."""
        mu    = mu.clamp(min=eps)
        alpha = alpha.clamp(min=0.01)
        log_nb0 = -(1.0 / alpha) * torch.log1p(alpha * mu)
        return gate + (1.0 - gate) * log_nb0.exp()

    def compute(self):
        if not self.expecteds:
            return {}

        vp     = torch.cat(self.expecteds, dim=0)
        vt     = torch.cat(self.targets,   dim=0)
        vmu    = torch.cat(self.mus,       dim=0)
        valpha = torch.cat(self.alphas,    dim=0)
        vgate  = torch.cat(self.gates,     dim=0)

        m = {}
        m["mae"]     = (vp - vt).abs().mean().item()
        m["rmse"]    = ((vp - vt) ** 2).mean().sqrt().item()
        m["mape"]    = ((vp - vt).abs() / vt.clamp(min=1)).mean().item() * 100
        m["log_mae"] = (torch.log1p(vp) - torch.log1p(vt)).abs().mean().item()

        # ── ZINB-based zero classification ──
        p_zero = self._zinb_prob_zero(vmu, valpha, vgate)
        is_zero_t = vt < 0.5
        is_zero_p = p_zero > self.zero_prob_threshold

        m["zero_recall"]    = (is_zero_p[is_zero_t].float().mean().item()
                               if is_zero_t.any() else 0.0)
        m["zero_precision"] = (is_zero_t[is_zero_p].float().mean().item()
                               if is_zero_p.any() else 0.0)
        pr, rc = m["zero_precision"], m["zero_recall"]
        m["zero_f1"] = (2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)

        m["avg_gate"]         = vgate.mean().item()
        m["avg_p_zero"]       = p_zero.mean().item()
        m["actual_zero_rate"] = is_zero_t.float().mean().item()
        m["pred_zero_rate"]   = is_zero_p.float().mean().item()

        if vp.std() > 1e-6 and vt.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([vp, vt]))[0, 1]
            m["correlation"] = corr.item() if not torch.isnan(corr) else 0.0
        else:
            m["correlation"] = 0.0

        for q in (0.5, 0.9, 0.95, 0.99):
            err = (vp - vt).abs()
            m[f"p{int(q * 100)}_error"] = torch.quantile(err, q).item()

        return m


# ════════════════════════════════════════════════════════════
# Monkey-patch and print only from main process
# ════════════════════════════════════════════════════════════
orig.WaterMeterMetrics    = FixedElectricityMeterMetrics
p.ElectricityMeterMetrics = FixedElectricityMeterMetrics
p.WaterMeterMetrics       = FixedElectricityMeterMetrics

if _mp.current_process().name == "MainProcess":
    print("[extra_patches v2] WaterMeterMetrics replaced "
          "(variable-L safe + ZINB-based zero classification).")
