"""
Ablation runner: ZINB transformer WITHOUT time-aware attention.
================================================================
Reuses run_paper_experiments.py exactly, changing only:
  - use_time_aware_attention = False   (plain nn.TransformerEncoder)
  - checkpoint_dir / results_dir       (so it doesn't clobber the main run)
  - seed (optional)

Everything else (data, features, loss weights, optimiser, epochs, GLM
baseline) is identical to the main model, so the comparison isolates the
single effect of the time-aware attention bias.

USAGE:
    python run_ablation_no_timebias.py
    python run_ablation_no_timebias.py --seed 42

After it finishes, evaluate exactly like the main model:
    python v5b_segmented_eval.py ^
        --checkpoint checkpoints_ablation_notime/best_ema.pt ^
        --agg peer_avg_aggregates.pkl

(Use ^ for line continuation in PowerShell, or put it all on one line.)
"""
import argparse
import run_paper_experiments as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_dir", default="checkpoints_ablation_notime")
    ap.add_argument("--results_dir", default="results_ablation_notime")
    args = ap.parse_args()

    # Build the SAME default config as the main experiment, then flip
    # exactly one switch.
    cfg = R.build_default_config()
    cfg.model.use_time_aware_attention = False      # <-- the ablation
    cfg.run.seed = args.seed
    cfg.run.checkpoint_dir = args.ckpt_dir
    cfg.run.results_dir = args.results_dir

    print("=" * 60)
    print("ABLATION: time-aware attention DISABLED")
    print(f"  seed           = {cfg.run.seed}")
    print(f"  checkpoint_dir = {cfg.run.checkpoint_dir}")
    print(f"  results_dir    = {cfg.run.results_dir}")
    print(f"  d_model/h/L    = {cfg.model.d_model}/"
          f"{cfg.model.n_heads}/{cfg.model.n_layers}")
    print("=" * 60)

    # Replicate main()'s body with our modified cfg.
    device, scaler = R.setup_run(cfg)
    cards, train_ds, train_loader, val_loader, test_loader = R.load_splits(cfg)
    model = R.build_model(cfg, cards, train_ds.rate_mean, device)
    R.orig.sanity_check(model, train_loader, device)
    optimizer = R.build_optimizer(model, cfg)
    model, ema = R.train(cfg, model, train_loader, val_loader,
                         optimizer, device, scaler)

    from pathlib import Path
    import torch
    ckpt_dir = Path(cfg.run.checkpoint_dir)
    best_ema = ckpt_dir / "best_ema.pt"
    best = ckpt_dir / "best.pt"
    if best_ema.exists():
        model.load_state_dict(torch.load(best_ema, map_location=device,
                                          weights_only=True))
        print("[ablation-eval] loaded best EMA weights")
    elif best.exists():
        model.load_state_dict(torch.load(best, map_location=device,
                                          weights_only=True))
        print("[ablation-eval] loaded best weights")

    R.evaluate_for_paper(model, test_loader, device,
                         out_dir=cfg.run.results_dir,
                         static_feature_names=list(cards.keys()))


if __name__ == "__main__":
    main()
