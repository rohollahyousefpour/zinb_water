"""
Post-hoc calibration: temperature scaling + zero-threshold tuning
==================================================================
Applies two cheap, well-known calibration fixes to the trained model
WITHOUT retraining:

  Fix 1 — Temperature scaling on the ZINB dispersion parameter α.
          Learn one positive scalar T on the validation set by
          minimising NLL.  Apply α_test ← α_test * T on test.
          This addresses the under-dispersion visible in your PIT
          histogram (large spike at PIT ≈ 1.0) and should drop ECE
          from ~0.51 to ~0.10.

  Fix 2 — Zero-classification threshold τ.
          Currently we use   P(Y=0) > 0.5   as the zero predictor.
          Sweep τ ∈ {0.05, 0.10, …, 0.90} on val and pick the τ that
          maximises F1.  Apply it on test.  Expected zero-F1 lift:
          0.30 → ~0.50.

Inputs:
    checkpoints_paper/best_ema.pt   (or best.pt as fallback)
    split_val.npz, split_test.npz
    static_cardinalities_ramz.json

Outputs (under  results_paper/calibrated/  ):
    main_table_calibrated.json
    fit_summary.json   ← {T*, τ*, NLL/ECE before & after}
    figures/pit_hist.png
    figures/reliability.png
    figures/scatter.png
    figures/error_vs_gap.png

Usage:
    python calibrate_and_evaluate.py
"""

import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import Improved_embeding as orig
import electricity_zinb_patches as p
import extra_patches  # noqa: F401  (monkey-patches metrics)


# ═════════════════════════════════════════════════════════════════
# CONFIG — edit paths if your layout differs
# ═════════════════════════════════════════════════════════════════
CONFIG = {
    "data_dir":             ".",
    "val_npz":              "split_val.npz",
    "test_npz":             "split_test.npz",
    "cardinalities_json":   "static_cardinalities_ramz.json",
    "checkpoint_dir":       "checkpoints_paper",
    "checkpoint_name":      "best_ema.pt",     # falls back to best.pt
    "out_dir":              "results_paper/calibrated",

    # Architecture — must match training run
    "d_model":  192,
    "n_heads":  6,
    "n_layers": 5,
    "dropout":  0.05,
    "n_years":  12,
    "use_time_aware_attention": True,

    # Runtime
    "batch_size":  256,
    "num_workers": 0,        # main process only — avoid worker-spawn noise
    "seed":        42,
}


# ═════════════════════════════════════════════════════════════════
# 1. Predictions buffer (collects μ, α, π, y, dt, statics)
# ═════════════════════════════════════════════════════════════════
@torch.no_grad()
def collect_predictions(model, loader, device, tag):
    """Run model over a loader; return CPU tensors of flat predictions."""
    model.eval()
    mus, alphas, gates = [], [], []
    targets, dts, statics = [], [], []
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        o = model(batch)
        m = o["mask"]
        if not m.any():
            continue

        mus    .append(o["mu"][m].float().cpu())
        alphas .append(o["alpha"][m].float().cpu())
        gates  .append(o["gate"][m].float().cpu())
        targets.append(o["target"][m].float().cpu())
        dts    .append(o["dt"][m].float().cpu())

        st_exp = o["_static_for_loss"].unsqueeze(1).expand(-1, m.shape[1], -1)
        statics.append(st_exp[m].cpu())

        n_batches += 1
        if n_batches % 50 == 0:
            print(f"  [{tag}] {n_batches} batches done")

    out = {
        "mu":      torch.cat(mus),
        "alpha":   torch.cat(alphas),
        "gate":    torch.cat(gates),
        "target":  torch.cat(targets),
        "dt":      torch.cat(dts),
        "static":  torch.cat(statics),
    }
    print(f"  [{tag}] collected {out['mu'].numel():,} predictions")
    return out


# ═════════════════════════════════════════════════════════════════
# 2. ZINB log-PMF (numerically stable; same as paper_evaluator)
# ═════════════════════════════════════════════════════════════════
def zinb_log_pmf(y, mu, alpha, gate, eps=1e-8, alpha_floor=1e-4):
    """
    Numerically-stable ZINB log p(Y=y).

    `alpha_floor` is the lower bound for α before computing r=1/α. We keep it
    permissive (1e-4) here so temperature scaling can explore both directions;
    the training-time loss uses a stricter floor for stability.
    """
    mu    = mu.clamp(min=eps, max=1e6)
    alpha = alpha.clamp(min=alpha_floor, max=1e3)
    gate  = gate.clamp(min=eps, max=1.0 - eps)
    y     = y.clamp(min=0.0)
    r     = (1.0 / alpha).clamp(min=1e-4, max=1e5)

    log_mu  = torch.log(mu)
    log_r   = torch.log(r)
    log_mur = torch.log(mu + r)
    log_g   = torch.log(gate)
    log_1mg = torch.log(1.0 - gate)

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


def zinb_prob_zero(mu, alpha, gate, eps=1e-8):
    """P(Y=0) = π + (1-π)(1 + αμ)^(-1/α)."""
    mu    = mu.clamp(min=eps)
    alpha = alpha.clamp(min=0.01)
    log_nb0 = -(1.0 / alpha) * torch.log1p(alpha * mu)
    return gate + (1.0 - gate) * log_nb0.exp()


def zinb_gaussian_pit(y, mu, alpha, gate):
    """PIT under Gaussian approximation of ZINB. Used for ECE/PIT diagnostics."""
    mean = (1 - gate) * mu
    var  = (1 - gate) * mu * (1 + alpha * mu + gate * mu)
    std  = var.clamp(min=1e-6).sqrt()
    z = (y - mean) / (std * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    return cdf.clamp(min=1e-6, max=1 - 1e-6)


# ═════════════════════════════════════════════════════════════════
# 3. Fix 1 — Temperature scaling on α
# ═════════════════════════════════════════════════════════════════
def fit_temperature_scaling(val):
    """
    Learn a single positive scalar T such that α' = α * T minimises val NLL.

    Two-stage fit, since the NLL landscape can be very flat for T<1 if the
    model was clamped at training time:
      1. Coarse log-space grid over T ∈ [0.05, 200] to bracket the minimum.
      2. Local LBFGS refinement starting from the best grid point.

    Returns:
        T*       : float
        nll_before, nll_after : floats
    """
    print("\n══════════ Fix 1: Temperature scaling on α ══════════")
    y, mu, a, g = val["target"], val["mu"], val["alpha"], val["gate"]

    nll_before = float(-zinb_log_pmf(y, mu, a, g).mean())
    print(f"  val NLL before:  {nll_before:.6f}")

    # ─── Stage 1: coarse log-space grid ───
    grid = torch.cat([
        torch.logspace(math.log10(0.05), math.log10(0.95), 8),
        torch.tensor([1.0]),
        torch.logspace(math.log10(1.05), math.log10(200.0), 24),
    ])
    grid_nlls = []
    for T in grid:
        nll = float(-zinb_log_pmf(y, mu, a * T, g).mean())
        grid_nlls.append(nll)
    j = int(np.argmin(grid_nlls))
    T_grid = float(grid[j])
    print(f"  best grid T:     {T_grid:.4f}  (NLL={grid_nlls[j]:.6f})")

    # ─── Stage 2: local LBFGS refinement ───
    # Parameterise T = exp(τ) for unbounded search; init at log(T_grid).
    tau = torch.tensor([math.log(max(T_grid, 1e-3))], requires_grad=True)
    opt = torch.optim.LBFGS([tau], lr=0.1, max_iter=80,
                            tolerance_grad=1e-8, tolerance_change=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = tau.exp()
        nll = -zinb_log_pmf(y, mu, a * T, g).mean()
        nll.backward()
        return nll

    opt.step(closure)
    T_star = float(tau.exp().item())
    nll_after = float(-zinb_log_pmf(y, mu, a * T_star, g).mean())

    # If LBFGS made things worse for some pathological reason, fall back to grid.
    if nll_after > grid_nlls[j] + 1e-6:
        T_star = T_grid
        nll_after = grid_nlls[j]
        print("  (LBFGS did not improve over grid; using grid value)")

    print(f"  T*               : {T_star:.4f}")
    print(f"  val NLL after    : {nll_after:.6f}  (Δ={nll_after - nll_before:+.4f})")
    return T_star, nll_before, nll_after


# ═════════════════════════════════════════════════════════════════
# 4. Fix 2 — Zero-threshold tuning
# ═════════════════════════════════════════════════════════════════
def fit_zero_threshold(val, T_star):
    """
    Sweep τ ∈ [0.05, 0.95] in 0.025 steps, compute F1 on val,
    return the maximising τ.
    """
    print("\n══════════ Fix 2: Zero-threshold tuning ══════════")
    y     = val["target"]
    p_zero = zinb_prob_zero(val["mu"], val["alpha"] * T_star, val["gate"])

    is_zero_t = y < 0.5
    n_pos = int(is_zero_t.sum())
    print(f"  empirical zero rate (val): {is_zero_t.float().mean():.4f}  "
          f"(n_true_zeros = {n_pos:,})")
    print(f"  default τ = 0.5 →  P(zero)>τ rate = "
          f"{(p_zero > 0.5).float().mean():.4f}")

    taus, f1s, precs, recs = [], [], [], []
    for tau in np.arange(0.05, 0.96, 0.025):
        is_zero_p = p_zero > float(tau)
        rec = (is_zero_p & is_zero_t).float().sum() / max(int(is_zero_t.sum()), 1)
        prec = (is_zero_p & is_zero_t).float().sum() / max(int(is_zero_p.sum()), 1)
        rec = float(rec); prec = float(prec)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        taus.append(float(tau)); f1s.append(f1); precs.append(prec); recs.append(rec)

    j = int(np.argmax(f1s))
    print(f"  best τ*          : {taus[j]:.3f}")
    print(f"  val zero P/R/F1  : P={precs[j]:.3f}  R={recs[j]:.3f}  F1={f1s[j]:.3f}")
    return float(taus[j]), float(f1s[j]), {
        "taus": taus, "f1s": f1s, "precs": precs, "recs": recs,
    }


# ═════════════════════════════════════════════════════════════════
# 5. Final test-set evaluation with the calibrated parameters
# ═════════════════════════════════════════════════════════════════
def evaluate_calibrated(test, T_star, tau_star, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    y, mu, a, g, dt = (test["target"], test["mu"], test["alpha"],
                       test["gate"], test["dt"])

    # ─────────── Calibrated quantities ───────────
    a_cal = a * T_star                       # Fix 1
    p_zero = zinb_prob_zero(mu, a_cal, g)    # for Fix 2
    is_zero_p = p_zero > tau_star
    is_zero_t = y < 0.5

    expected = (1 - g) * mu
    err = expected - y
    n = expected.shape[0]

    main = {
        "n_test_readings": int(n),
        "fix_1_temperature_T":    float(T_star),
        "fix_2_zero_threshold":   float(tau_star),

        # Point-prediction metrics (unchanged by Fix 1/2)
        "mae":         float(err.abs().mean()),
        "rmse":        float((err ** 2).mean().sqrt()),
        "mape":        float((err.abs() / y.clamp(min=1)).mean() * 100),
        "log_mae":     float((torch.log1p(expected) - torch.log1p(y)).abs().mean()),
        "correlation": (float(torch.corrcoef(torch.stack([expected, y]))[0, 1])
                        if expected.std() > 1e-6 and y.std() > 1e-6 else 0.0),

        # Distributional metrics (Fix 1 acts here)
        "nll_mean_before_T":  float(-zinb_log_pmf(y, mu, a,     g).mean()),
        "nll_mean_after_T":   float(-zinb_log_pmf(y, mu, a_cal, g).mean()),
    }

    # ECE via PIT (before / after Fix 1)
    for label, alpha_use in (("before", a), ("after", a_cal)):
        pit = zinb_gaussian_pit(y, mu, alpha_use, g).numpy()
        counts, _ = np.histogram(pit, bins=np.linspace(0, 1, 11))
        ece = float(np.abs(counts - len(pit) / 10).sum() / len(pit))
        main[f"ece_{label}_T"] = ece
        for q in (0.5, 0.8, 0.9, 0.95):
            lo, hi = (1 - q) / 2, 1 - (1 - q) / 2
            cov = float(((pit >= lo) & (pit <= hi)).mean())
            main[f"picp_{int(100*q)}_{label}_T"] = cov

    # Zero F1 — before / after Fix 2 (both use Fix-1-calibrated α)
    main["zero_f1_default_tau"] = _f1(zinb_prob_zero(mu, a_cal, g) > 0.5, is_zero_t)
    main["zero_f1_tuned_tau"]   = _f1(is_zero_p, is_zero_t)

    p_default = zinb_prob_zero(mu, a_cal, g) > 0.5
    main["zero_precision_default"] = _prec(p_default, is_zero_t)
    main["zero_recall_default"]    = _rec (p_default, is_zero_t)
    main["zero_precision_tuned"]   = _prec(is_zero_p, is_zero_t)
    main["zero_recall_tuned"]      = _rec (is_zero_p, is_zero_t)

    main["empirical_zero_rate"] = float(is_zero_t.float().mean())
    main["predicted_zero_rate_default"] = float(p_default.float().mean())
    main["predicted_zero_rate_tuned"]   = float(is_zero_p.float().mean())

    with open(out / "main_table_calibrated.json", "w") as f:
        json.dump(main, f, indent=2)
    print(f"\n  wrote {out / 'main_table_calibrated.json'}")

    _make_figures(y, expected, mu, a_cal, g, dt, main, out)
    return main


def _f1(p, t):
    rec  = float((p & t).float().sum() / max(int(t.sum()), 1))
    prec = float((p & t).float().sum() / max(int(p.sum()), 1))
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

def _prec(p, t):
    return float((p & t).float().sum() / max(int(p.sum()), 1))

def _rec(p, t):
    return float((p & t).float().sum() / max(int(t.sum()), 1))


# ═════════════════════════════════════════════════════════════════
# 6. Figures
# ═════════════════════════════════════════════════════════════════
def _make_figures(y, expected, mu, a_cal, g, dt, main, out):
    figs = out / "figures"

    # ─── PIT histogram (after calibration) ───
    pit = zinb_gaussian_pit(y, mu, a_cal, g).numpy()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(pit, bins=20, range=(0, 1), color="#3b82c4",
            edgecolor="white", linewidth=0.3, density=True)
    ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.6,
               label="ideal (uniform)")
    ax.set_xlabel("PIT value"); ax.set_ylabel("density")
    ax.set_title(f"PIT histogram, calibrated  "
                 f"(ECE: {main['ece_before_T']:.3f} → {main['ece_after_T']:.3f})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figs / "pit_hist.png", dpi=160); plt.close(fig)

    # ─── Reliability diagram ───
    nominal = np.linspace(0.05, 0.95, 19)
    pit_before = zinb_gaussian_pit(y, mu, a_cal / main["fix_1_temperature_T"], g).numpy()
    emp_before, emp_after = [], []
    for q in nominal:
        lo, hi = (1 - q) / 2, 1 - (1 - q) / 2
        emp_before.append(((pit_before >= lo) & (pit_before <= hi)).mean())
        emp_after .append(((pit         >= lo) & (pit         <= hi)).mean())

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="ideal")
    ax.plot(nominal, emp_before, "o-", color="#888888", markersize=4,
            label="before T")
    ax.plot(nominal, emp_after,  "o-", color="#3b82c4", markersize=4,
            label="after T")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(figs / "reliability.png", dpi=160); plt.close(fig)

    # ─── Scatter (point predictions; unchanged by Fix 1/2) ───
    n = y.shape[0]
    sample = torch.randperm(n)[:50_000]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y[sample].clamp(min=1e-1),
               expected[sample].clamp(min=1e-1),
               s=1, alpha=0.2, color="#3b82c4")
    lo = max(1e-1, float(y[y > 0].min())); hi = float(y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="y=x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("actual (kWh)"); ax.set_ylabel("predicted (kWh)")
    ax.set_title("Predicted vs. actual (test set, log scale)")
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "scatter.png", dpi=160); plt.close(fig)

    # ─── Error vs gap (unchanged by Fix 1/2 but regenerated for completeness) ───
    gap_bins = [(0, 50), (50, 70), (70, 100), (100, 200), (200, 365)]
    rows = []
    for low, hi_ in gap_bins:
        sel = (dt >= low) & (dt < hi_)
        if int(sel.sum()) < 30: continue
        rows.append({
            "gap_range": f"[{low},{hi_})",
            "n":   int(sel.sum()),
            "mae": float((expected[sel] - y[sel]).abs().mean()),
        })
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        xs = [r["gap_range"] for r in rows]
        ys = [r["mae"]       for r in rows]
        ns = [r["n"]         for r in rows]
        ax.bar(xs, ys, color="#c44d3b")
        for i, (y_, n_) in enumerate(zip(ys, ns)):
            ax.text(i, y_, f"n={n_:,}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("inter-reading gap (days)")
        ax.set_ylabel("MAE (kWh)")
        ax.set_title("Test MAE by reading-gap length")
        fig.tight_layout()
        fig.savefig(figs / "error_vs_gap.png", dpi=160); plt.close(fig)

    print(f"  wrote 4 figures to {figs}/")


# ═════════════════════════════════════════════════════════════════
# 7. Main
# ═════════════════════════════════════════════════════════════════
def main():
    cfg = CONFIG
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # ─── Load cardinalities + datasets ───
    with open(Path(cfg["data_dir"]) / cfg["cardinalities_json"]) as f:
        cards = json.load(f, object_pairs_hook=OrderedDict)
    print(f"cardinalities: {dict(cards)}")

    DS = p.ImprovedElectricityMeterDataset
    val_ds  = DS(str(Path(cfg["data_dir"]) / cfg["val_npz"]),  cards)
    test_ds = DS(str(Path(cfg["data_dir"]) / cfg["test_npz"]), cards)
    print(f"sizes: val={len(val_ds):,}  test={len(test_ds):,}\n")

    common = dict(batch_size=cfg["batch_size"], collate_fn=orig.collate_fn,
                  num_workers=cfg["num_workers"], pin_memory=True,
                  shuffle=False)
    val_loader  = DataLoader(val_ds,  **common)
    test_loader = DataLoader(test_ds, **common)

    # ─── Build model + load checkpoint ───
    p.orig.AbsoluteTimeEmbedding = lambda d_model: p.AbsoluteTimeEmbedding(
        d_model, n_years=cfg["n_years"])
    model = p.ImprovedZINBElectricityMeterEncoder(
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], static_cardinalities=cards,
        dropout=cfg["dropout"],
        use_time_aware_attention=cfg["use_time_aware_attention"],
        default_rate=val_ds.rate_mean,
    ).to(device)

    ckpt = Path(cfg["checkpoint_dir"]) / cfg["checkpoint_name"]
    if not ckpt.exists():
        alt = Path(cfg["checkpoint_dir"]) / "best.pt"
        if alt.exists():
            ckpt = alt
        else:
            raise FileNotFoundError(f"No checkpoint at {ckpt} or {alt}")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state)
    print(f"loaded {ckpt}")

    # ─── Collect predictions on val + test ───
    print("\nCollecting val predictions...")
    val = collect_predictions(model, val_loader, device, tag="val")
    print("\nCollecting test predictions...")
    test = collect_predictions(model, test_loader, device, tag="test")

    # ─── Fix 1: temperature scaling on val ───
    T_star, nll_before, nll_after = fit_temperature_scaling(val)

    # ─── Fix 2: threshold tuning on val (using calibrated α) ───
    tau_star, f1_val, sweep = fit_zero_threshold(val, T_star)

    # ─── Final test evaluation ───
    print("\n══════════ Test evaluation (calibrated) ══════════")
    main_results = evaluate_calibrated(test, T_star, tau_star, cfg["out_dir"])

    # ─── Save sweep + fit summary ───
    summary = {
        "T_star":          T_star,
        "tau_star":        tau_star,
        "val_nll_before":  nll_before,
        "val_nll_after":   nll_after,
        "val_zero_f1_at_tau_star": f1_val,
        "tau_sweep": sweep,
    }
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    with open(out / "fit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ─── Console summary ───
    print("\n" + "═" * 64)
    print("CALIBRATED PAPER METRICS  (test set)")
    print("═" * 64)
    m = main_results
    print(f"  n test readings : {m['n_test_readings']:,}")
    print(f"  MAE             : {m['mae']:.2f} kWh        (unchanged)")
    print(f"  RMSE            : {m['rmse']:.2f} kWh        (unchanged)")
    print(f"  correlation     : {m['correlation']:.4f}     (unchanged)")
    print(f"  ─── Fix 1: temperature scaling T = {T_star:.3f} ───")
    print(f"  NLL    before T : {m['nll_mean_before_T']:.4f}")
    print(f"  NLL    after  T : {m['nll_mean_after_T']:.4f}")
    print(f"  ECE    before T : {m['ece_before_T']:.4f}")
    print(f"  ECE    after  T : {m['ece_after_T']:.4f}")
    print(f"  PICP@90 before  : {m['picp_90_before_T']:.4f}")
    print(f"  PICP@90 after   : {m['picp_90_after_T']:.4f}  (ideal 0.90)")
    print(f"  ─── Fix 2: zero-threshold τ = {tau_star:.3f} ───")
    print(f"  zero F1 @ τ=0.5 : {m['zero_f1_default_tau']:.4f}  "
          f"(P={m['zero_precision_default']:.3f}  R={m['zero_recall_default']:.3f})")
    print(f"  zero F1 @ τ*    : {m['zero_f1_tuned_tau']:.4f}  "
          f"(P={m['zero_precision_tuned']:.3f}  R={m['zero_recall_tuned']:.3f})")
    print("═" * 64)
    print(f"\nAll outputs under: {out}/\n")


if __name__ == "__main__":
    main()
