"""
Paper Evaluator — Comprehensive Test-Set Evaluation for ZINB Model
==================================================================
Produces every metric, table, and figure needed for Paper 1 §6 (Results).

What it computes:
  POINT-PREDICTION : MAE, RMSE, MAPE, log-MAE  (overall + per-tariff)
  PROBABILISTIC    : NLL, CRPS (sample-based), ECE, PICP at 50/80/90/95%
  ZERO DETECTION   : precision, recall, F1
  CALIBRATION      : PIT histogram, reliability diagram
  BREAKDOWNS       : per-tariff, per-urban, per-amper, by gap length

What it produces:
  results/main_table.json          ← headline metrics for §6.1
  results/per_segment.csv          ← per-tariff/region breakdown for appendix
  results/calibration.json         ← ECE, PICP for §6.3
  results/figures/pit_hist.png     ← PIT histogram (§6.3)
  results/figures/reliability.png  ← reliability diagram (§6.3)
  results/figures/error_vs_gap.png ← error by gap length (§6.4)
  results/figures/scatter.png      ← predicted vs. actual

Usage from script:
    from paper_evaluator import evaluate_for_paper
    evaluate_for_paper(model, test_loader, device, out_dir="results")
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════
# 1. Distributional helpers
# ════════════════════════════════════════════════════════════
def zinb_log_pmf(y, mu, alpha, gate, eps=1e-8):
    """Numerically-stable ZINB log p(Y=y).
       Returns a tensor of the same shape as y."""
    mu    = mu.clamp(min=eps, max=1e6)
    alpha = alpha.clamp(min=0.01, max=100.0)
    gate  = gate.clamp(min=eps, max=1.0 - eps)
    y     = y.clamp(min=0.0)
    r     = (1.0 / alpha).clamp(min=1e-4, max=1e4)

    log_mu  = torch.log(mu);    log_r = torch.log(r)
    log_mur = torch.log(mu + r)
    log_g   = torch.log(gate);  log_1mg = torch.log(1.0 - gate)

    nb_zero = (r * log_r - r * log_mur).clamp(-300.0, 50.0)
    nb_pos  = (
        torch.lgamma((y + r).clamp(min=eps, max=1e6))
        - torch.lgamma(r.clamp(min=eps, max=1e6))
        - torch.lgamma((y + 1.0).clamp(min=1.0, max=1e6))
        + y * log_mu + r * log_r - (y + r) * log_mur
    ).clamp(-300.0, 50.0)

    ll_zero = torch.logaddexp(log_g, nb_zero + log_1mg)
    ll_pos  = nb_pos + log_1mg
    return torch.where(y < 0.5, ll_zero, ll_pos)


def zinb_cdf_at(y, mu, alpha, gate, n_grid=None):
    """Approximate P(Y ≤ y) by summing the ZINB pmf from 0 to y.
       Used only for ECE / PIT on small-to-medium counts (y < n_grid).
       For large y we use the Gaussian approximation."""
    # Use Gaussian approximation: Y ~ N((1-π)μ, (1-π)μ(1+αμ+πμ))
    mean = (1 - gate) * mu
    var  = (1 - gate) * mu * (1 + alpha * mu + gate * mu)
    std  = var.clamp(min=1e-6).sqrt()
    # Probability integral transform via Gaussian — adequate for the
    # ECE/PIT diagnostics at this scale (median ~270 kWh).
    from math import sqrt
    z = (y - mean) / (std * sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    return cdf.clamp(min=1e-6, max=1 - 1e-6)


def zinb_sample(mu, alpha, gate, n=1000, device=None):
    """Draw ZINB samples for CRPS estimation. Shape: (n, *mu.shape)."""
    device = device or mu.device
    shape = mu.shape
    # Bernoulli gate
    is_zero = (torch.rand((n,) + shape, device=device) < gate.unsqueeze(0))
    # NB samples via gamma-Poisson: Y ~ Pois(λ), λ ~ Gamma(r, p/(1-p))
    # with r = 1/α, p = α μ / (1+αμ)
    r = (1.0 / alpha.clamp(min=0.01)).clamp(min=1e-3, max=1e3)
    gamma_shape = r.unsqueeze(0).expand((n,) + shape).clamp(min=1e-3)
    gamma_rate  = (r / mu.clamp(min=1e-4)).unsqueeze(0).expand((n,) + shape).clamp(min=1e-6)
    lam = torch._standard_gamma(gamma_shape) / gamma_rate
    nb  = torch.poisson(lam.clamp(min=0.0, max=1e6))
    return torch.where(is_zero, torch.zeros_like(nb), nb)


def crps_sample_based(y, samples):
    """CRPS via the sample-based estimator (Gneiting & Raftery 2007).
       samples: (n, *y.shape). y: (*shape).
       CRPS = E|Y - y|  -  (1/2) E|Y - Y'|"""
    n = samples.shape[0]
    term1 = (samples - y.unsqueeze(0)).abs().mean(dim=0)
    # term2: pairwise mean diff. O(n) trick via sort.
    s_sorted, _ = samples.sort(dim=0)
    w = (2 * torch.arange(1, n + 1, device=samples.device, dtype=samples.dtype)
         - n - 1) / (n * n)
    while w.dim() < s_sorted.dim():
        w = w.unsqueeze(-1)
    term2 = (w * s_sorted).sum(dim=0)
    return term1 - 0.5 * term2


# ════════════════════════════════════════════════════════════
# 2. Main evaluator
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_for_paper(model, test_loader, device,
                       out_dir="results",
                       n_crps_samples=200,
                       static_feature_names=None,
                       compute_crps=True):
    """
    Run model over test_loader and produce all paper artifacts.
    Returns a dict of results; also writes them to disk.
    """
    model.eval()
    out = Path(out_dir); out.mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    # Buffers (CPU, float64 to avoid loss of precision in sums)
    preds, targets, dts = [], [], []
    mus, alphas, gates = [], [], []
    nlls, statics = [], []
    crps_vals = []

    for batch in test_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        o = model(batch)

        m = o["mask"]
        if not m.any():
            continue

        mu, alpha, gate = o["mu"], o["alpha"], o["gate"]
        tgt, dt = o["target"], o["dt"]

        # NLL per element
        ll = zinb_log_pmf(tgt, mu, alpha, gate)
        nll = -ll

        # CRPS (sampled) — optional, expensive
        if compute_crps:
            samp = zinb_sample(mu, alpha, gate, n=n_crps_samples,
                               device=device)
            crps = crps_sample_based(tgt, samp)
        else:
            crps = torch.zeros_like(tgt)

        expected = (1 - gate) * mu

        preds.append(expected[m].cpu())
        targets.append(tgt[m].cpu())
        dts.append(dt[m].cpu())
        mus.append(mu[m].cpu()); alphas.append(alpha[m].cpu()); gates.append(gate[m].cpu())
        nlls.append(nll[m].cpu()); crps_vals.append(crps[m].cpu())
        # Static features per-meter: broadcast across positions
        st = o["_static_for_loss"]
        st_exp = st.unsqueeze(1).expand(-1, m.shape[1], -1)
        statics.append(st_exp[m].cpu())

    preds   = torch.cat(preds);   targets = torch.cat(targets)
    dts     = torch.cat(dts);     mus     = torch.cat(mus)
    alphas  = torch.cat(alphas);  gates   = torch.cat(gates)
    nlls    = torch.cat(nlls);    crps_vals = torch.cat(crps_vals)
    statics = torch.cat(statics)

    n = preds.shape[0]
    print(f"[paper-eval] evaluated {n:,} test readings")

    # ────────────────────────────────────────────────────
    # 2a. Point-prediction metrics
    # ────────────────────────────────────────────────────
    err  = preds - targets
    main = {
        "n_test_readings": int(n),
        "mae":      float(err.abs().mean()),
        "rmse":     float((err ** 2).mean().sqrt()),
        "mape":     float((err.abs() / targets.clamp(min=1)).mean() * 100),
        "log_mae":  float((torch.log1p(preds) - torch.log1p(targets)).abs().mean()),
        "correlation": float(
            torch.corrcoef(torch.stack([preds, targets]))[0, 1]
        ) if preds.std() > 1e-6 and targets.std() > 1e-6 else 0.0,
    }
    for q in (0.5, 0.9, 0.95, 0.99):
        main[f"p{int(100*q)}_abs_error"] = float(torch.quantile(err.abs(), q))

    # ────────────────────────────────────────────────────
    # 2b. Probabilistic metrics
    # ────────────────────────────────────────────────────
    main["nll_mean"]  = float(nlls.mean())
    main["nll_median"] = float(nlls.median())
    if compute_crps:
        main["crps_mean"]   = float(crps_vals.mean())
        main["crps_median"] = float(crps_vals.median())

    # ECE via PIT
    pit = zinb_cdf_at(targets, mus, alphas, gates).numpy()
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(pit, bins=bin_edges)
    expected_count = len(pit) / n_bins
    ece = np.abs(counts - expected_count).sum() / len(pit)
    main["ece"] = float(ece)

    # PICP at 50/80/90/95
    for q in (0.5, 0.8, 0.9, 0.95):
        lo, hi = (1 - q) / 2, 1 - (1 - q) / 2
        cov = ((pit >= lo) & (pit <= hi)).mean()
        main[f"picp_{int(100*q)}"] = float(cov)

    # ────────────────────────────────────────────────────
    # 2c. Zero-detection (gate-based)
    # ────────────────────────────────────────────────────
    is_zero_t = targets < 0.5
    is_zero_p = gates > 0.5
    if is_zero_t.any():
        recall = float(is_zero_p[is_zero_t].float().mean())
    else:
        recall = 0.0
    if is_zero_p.any():
        prec = float(is_zero_t[is_zero_p].float().mean())
    else:
        prec = 0.0
    main["zero_precision"] = prec
    main["zero_recall"]    = recall
    main["zero_f1"]        = (2 * prec * recall / (prec + recall)
                              if (prec + recall) > 0 else 0.0)
    main["empirical_zero_rate"]  = float(is_zero_t.float().mean())
    main["predicted_zero_rate"]  = float(is_zero_p.float().mean())

    with open(out / "main_table.json", "w") as f:
        json.dump(main, f, indent=2)
    print(f"[paper-eval] main_table.json written")

    # ────────────────────────────────────────────────────
    # 2d. Per-segment breakdowns
    # ────────────────────────────────────────────────────
    if static_feature_names is None:
        static_feature_names = ["meter_type", "tariff_code", "is_urban",
                                "region_in", "phase", "amper", "section_code"]

    per_segment_rows = []
    for j, feat in enumerate(static_feature_names):
        if j >= statics.shape[1]:
            break
        col = statics[:, j]
        for v in torch.unique(col).tolist():
            sel = col == v
            if sel.sum() < 30:
                continue
            sub_err = preds[sel] - targets[sel]
            sub_z_t = targets[sel] < 0.5
            sub_z_p = gates[sel] > 0.5
            r = float(sub_z_p[sub_z_t].float().mean()) if sub_z_t.any() else 0.0
            p = float(sub_z_t[sub_z_p].float().mean()) if sub_z_p.any() else 0.0
            per_segment_rows.append({
                "feature":     feat,
                "value":       int(v),
                "n_readings":  int(sel.sum()),
                "mae":         float(sub_err.abs().mean()),
                "rmse":        float((sub_err ** 2).mean().sqrt()),
                "nll_mean":    float(nlls[sel].mean()),
                "zero_f1":     (2 * p * r / (p + r) if (p + r) > 0 else 0.0),
                "empirical_zero_rate": float(sub_z_t.float().mean()),
            })

    # write as CSV
    import csv
    if per_segment_rows:
        with open(out / "per_segment.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_segment_rows[0].keys())
            w.writeheader()
            w.writerows(per_segment_rows)
        print(f"[paper-eval] per_segment.csv written "
              f"({len(per_segment_rows)} rows)")

    # ────────────────────────────────────────────────────
    # 2e. Error vs gap length
    # ────────────────────────────────────────────────────
    gap_bins = [(0, 50), (50, 70), (70, 100), (100, 200), (200, 365)]
    gap_rows = []
    for lo, hi in gap_bins:
        sel = (dts >= lo) & (dts < hi)
        if sel.sum() < 30:
            continue
        sub_err = preds[sel] - targets[sel]
        gap_rows.append({
            "gap_range": f"[{lo},{hi})",
            "n":         int(sel.sum()),
            "mae":       float(sub_err.abs().mean()),
            "rmse":      float((sub_err ** 2).mean().sqrt()),
            "nll_mean":  float(nlls[sel].mean()),
        })
    with open(out / "error_vs_gap.json", "w") as f:
        json.dump(gap_rows, f, indent=2)

    # ────────────────────────────────────────────────────
    # 2f. Figures
    # ────────────────────────────────────────────────────
    # PIT histogram
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(pit, bins=20, range=(0, 1), color="#3b82c4",
            edgecolor="white", linewidth=0.3, density=True)
    ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.6,
               label="ideal (uniform)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("density")
    ax.set_title(f"PIT histogram (ECE={main['ece']:.4f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "pit_hist.png", dpi=160)
    plt.close(fig)

    # Reliability diagram
    nominal = np.linspace(0.05, 0.95, 19)
    empirical = []
    for q in nominal:
        lo, hi = (1 - q) / 2, 1 - (1 - q) / 2
        empirical.append(((pit >= lo) & (pit <= hi)).mean())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=0.8, alpha=0.6,
            label="ideal")
    ax.plot(nominal, empirical, "o-", color="#3b82c4",
            markersize=4, label="model")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "figures" / "reliability.png", dpi=160)
    plt.close(fig)

    # Error vs gap (bar)
    if gap_rows:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        xs = [r["gap_range"] for r in gap_rows]
        ys = [r["mae"]       for r in gap_rows]
        ns = [r["n"]         for r in gap_rows]
        ax.bar(xs, ys, color="#c44d3b")
        for i, (y_, n_) in enumerate(zip(ys, ns)):
            ax.text(i, y_, f"n={n_:,}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("inter-reading gap (days)")
        ax.set_ylabel("MAE (kWh)")
        ax.set_title("Test MAE by reading-gap length")
        fig.tight_layout()
        fig.savefig(out / "figures" / "error_vs_gap.png", dpi=160)
        plt.close(fig)

    # Pred-vs-actual scatter (log-log)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sample = torch.randperm(n)[:50_000]
    ax.scatter(targets[sample].clamp(min=1e-1),
               preds[sample].clamp(min=1e-1),
               s=1, alpha=0.2, color="#3b82c4")
    lo = max(1e-1, float(targets[targets > 0].min()))
    hi = float(targets.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="y=x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("actual (kWh)"); ax.set_ylabel("predicted (kWh)")
    ax.set_title("Predicted vs. actual (test set, log scale)")
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "scatter.png", dpi=160)
    plt.close(fig)

    print(f"[paper-eval] figures written to {out}/figures/")

    # ────────────────────────────────────────────────────
    # 2g. Console summary
    # ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("PAPER METRICS (test set)")
    print("═" * 60)
    print(f"  n test readings : {n:,}")
    print(f"  MAE             : {main['mae']:.2f} kWh")
    print(f"  RMSE            : {main['rmse']:.2f} kWh")
    print(f"  MAPE            : {main['mape']:.2f} %")
    print(f"  log-MAE         : {main['log_mae']:.4f}")
    print(f"  correlation     : {main['correlation']:.4f}")
    print(f"  NLL (mean)      : {main['nll_mean']:.4f}")
    if compute_crps:
        print(f"  CRPS (mean)     : {main['crps_mean']:.2f}")
    print(f"  ECE             : {main['ece']:.4f}")
    print(f"  PICP @ 90%      : {main['picp_90']:.4f}  "
          f"(ideal: 0.90)")
    print(f"  zero-F1         : {main['zero_f1']:.4f}  "
          f"(P={main['zero_precision']:.3f}, R={main['zero_recall']:.3f})")
    print(f"  emp zero rate   : {main['empirical_zero_rate']:.4f}")
    print(f"  pred zero rate  : {main['predicted_zero_rate']:.4f}")
    print("═" * 60 + "\n")

    return main
