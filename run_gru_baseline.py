"""
GRU baseline for the paper's baseline comparison.
==================================================
This is a recognized neural sequence baseline (a recurrent encoder)
that consumes the SAME inputs, heads, ZINB output, and GLM baseline as
the main model. The ONLY change is the sequence encoder: a bidirectional
... no -- a *causal* (unidirectional) multi-layer GRU replaces the
time-aware transformer encoder.

Why this is a fair baseline:
  * identical four-stream input embedding
  * identical static-conditioned heads (correction/scale/alpha/gate)
  * identical frozen per-tariff GLM baseline
  * identical loss, optimiser schedule, epochs, data split
  * the GRU sees dt as an input feature (via the embedding) but, unlike
    the transformer, has NO time-aware attention bias. So comparing it
    to the full model isolates the value of (a) attention and (b) the
    time-aware bias jointly, while the no-time-bias ablation isolates
    (b) alone. Reporting both gives a clean decomposition.

NOTE ON FAIRNESS: a GRU is inherently sequential and cannot attend to
arbitrary past readings the way attention can; this is exactly the
capability under test, so the comparison is appropriate, not rigged.

USAGE:
    python run_gru_baseline.py --seed 42
    python v5b_segmented_eval.py --checkpoint checkpoints_gru/best_ema.pt \
        --agg peer_avg_aggregates.pkl   (NOTE: see eval caveat below)

EVAL CAVEAT: v5b_segmented_eval.py loads the transformer class by name.
To evaluate this GRU model with the same script, either (a) point the
eval's model constructor at GRUEncoderModel, or (b) use the test-MAE that
this script prints at the end via evaluate_for_paper(). Option (b) needs
no eval changes and gives the comparable headline number.
"""
import argparse
import torch
import torch.nn as nn

import run_paper_experiments as R
# (model construction is delegated to run_paper_experiments.build_model,
#  so we don't need to import the model module directly here.)


class GRUEncoder(nn.Module):
    """Causal multi-layer GRU with a final LayerNorm, matching the
    transformer encoder's (B, L, d_model) -> (B, L, d_model) interface."""

    def __init__(self, d_model, n_layers, dropout=0.05):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,        # causal: only past -> present
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, times=None, mask=None):
        # times/mask accepted for interface compatibility; GRU ignores
        # the pairwise time bias (that is the capability under test).
        out, _ = self.gru(x)
        return self.final_norm(out)


def build_gru_model(cfg, cards, default_rate, device):
    """Build the full model via run_paper_experiments (which handles the
    AbsoluteTimeEmbedding patching correctly), then replace its encoder
    with a GRU. This avoids duplicating the monkey-patch logic."""
    # Use the main builder so all the import/patching matches exactly.
    model = R.build_model(cfg, cards, default_rate, device)

    # Swap the encoder for a causal GRU. The main model's forward calls
    # self.encoder(x, times, mask) on the time-aware path; our GRUEncoder
    # accepts that signature and ignores the time bias (the capability
    # under test). Force the instance onto that call path.
    model.use_time_aware_attention = True
    model.encoder = GRUEncoder(
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    n_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    print(f"[GRU baseline] model parameters: {n_params:,}")
    return model


def build_optimizer_simple(model, cfg):
    """A single-group AdamW — avoids the transformer-specific parameter
    grouping in run_paper_experiments (which references encoder/attention
    submodules that no longer exist)."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.98), eps=1e-8,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_dir", default="checkpoints_gru")
    ap.add_argument("--results_dir", default="results_gru")
    args = ap.parse_args()

    cfg = R.build_default_config()
    cfg.run.seed = args.seed
    cfg.run.checkpoint_dir = args.ckpt_dir
    cfg.run.results_dir = args.results_dir

    print("=" * 60)
    print("BASELINE: GRU sequence encoder (no attention)")
    print(f"  seed={cfg.run.seed}  dirs={args.ckpt_dir}/{args.results_dir}")
    print("=" * 60)

    device, scaler = R.setup_run(cfg)
    cards, train_ds, train_loader, val_loader, test_loader = R.load_splits(cfg)
    model = build_gru_model(cfg, cards, train_ds.rate_mean, device)

    R.orig.sanity_check(model, train_loader, device)
    optimizer = build_optimizer_simple(model, cfg)
    model, ema = R.train(cfg, model, train_loader, val_loader,
                         optimizer, device, scaler)

    from pathlib import Path
    ckpt_dir = Path(cfg.run.checkpoint_dir)
    for name in ("best_ema.pt", "best.pt"):
        p = ckpt_dir / name
        if p.exists():
            model.load_state_dict(torch.load(p, map_location=device,
                                             weights_only=True))
            print(f"[gru-eval] loaded {name}")
            break

    R.evaluate_for_paper(model, test_loader, device,
                         out_dir=cfg.run.results_dir,
                         static_feature_names=list(cards.keys()))


if __name__ == "__main__":
    main()