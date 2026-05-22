"""
EXACT drop-in for YOUR run_paper_experiments.py
================================================
Two edits to make:

EDIT 1 — at the top of run_paper_experiments.py, with the other imports
         (around line 21, near `import json`), add ONE line:

    import argparse

EDIT 2 — replace your existing main() (lines 348-380) with the main()
         below. Everything inside is identical to yours except:
           • it reads --seed from the command line
           • it sets cfg.run.seed and isolates the output dirs per seed
           • it captures the test MAE from evaluate_for_paper (return value
             OR the JSON the evaluator writes OR a clearly-printed line)
             and writes results_paper_s{seed}/seed_result.json

Nothing about the model, data, or training changes. The data split is
fixed inside the NPZ files (preprocessor SPLIT_SEED), so every seed uses
the SAME test meters — the runs are directly comparable. Only the model
init + batch-shuffle seed varies.

THEN run three times:
    python run_paper_experiments.py --seed 42
    python run_paper_experiments.py --seed 43
    python run_paper_experiments.py --seed 44

(You can skip 42 if you trust your existing 129.5 run; two more seeds is
enough for n=3.)

Finally:
    python aggregate_seeds.py --seeds 42 43 44
or just:
    python aggregate_seeds.py --vals 129.5 <run43_mae> <run44_mae>
"""

# ============================================================
# vvv  COPY FROM HERE  vvv
# ============================================================
def main():
    import argparse, json, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="model init + shuffle seed (data split is fixed "
                         "in the NPZ and does NOT change)")
    args = ap.parse_args()

    cfg = build_default_config()

    # Apply the seed and isolate output dirs so the runs don't overwrite.
    cfg.run.seed = args.seed
    cfg.run.checkpoint_dir = f"checkpoints_paper_s{args.seed}"
    cfg.run.results_dir    = f"results_paper_s{args.seed}"
    print(f"\n*** MULTI-SEED RUN: seed={args.seed} ***")
    print(f"    checkpoints -> {cfg.run.checkpoint_dir}")
    print(f"    results     -> {cfg.run.results_dir}\n")

    cfg.report()

    device, scaler = setup_run(cfg)
    cards, train_ds, train_loader, val_loader, test_loader = load_splits(cfg)
    model = build_model(cfg, cards, train_ds.rate_mean, device)

    # Sanity check
    orig.sanity_check(model, train_loader, device)

    optimizer = build_optimizer(model, cfg)
    model, ema = train(cfg, model, train_loader, val_loader,
                       optimizer, device, scaler)

    # Final test eval — prefer EMA
    ckpt_dir = Path(cfg.run.checkpoint_dir)
    best_ema = ckpt_dir / "best_ema.pt"
    best = ckpt_dir / "best.pt"
    if best_ema.exists():
        model.load_state_dict(torch.load(best_ema, map_location=device,
                                         weights_only=True))
        print("[paper-eval] loaded best EMA weights")
    elif best.exists():
        model.load_state_dict(torch.load(best, map_location=device,
                                         weights_only=True))
        print("[paper-eval] loaded best weights")
    else:
        print("[paper-eval] no checkpoint found — using current weights")

    metrics = evaluate_for_paper(model, test_loader, device,
                                 out_dir=cfg.run.results_dir,
                                 static_feature_names=list(cards.keys()))

    # ── capture the test MAE for this seed (robust to evaluator shape) ──
    test_mae = None
    # (a) return value is a dict?
    if isinstance(metrics, dict):
        for k in ("mae", "test_mae", "MAE", "test_MAE"):
            if k in metrics and isinstance(metrics[k], (int, float)):
                test_mae = float(metrics[k]); break
    # (b) return value is a bare number?
    elif isinstance(metrics, (int, float)):
        test_mae = float(metrics)
    # (c) scan any JSON the evaluator wrote into results_dir
    if test_mae is None:
        for jf in glob.glob(str(Path(cfg.run.results_dir) / "*.json")):
            try:
                with open(jf) as f:
                    d = json.load(f)
            except Exception:
                continue
            if isinstance(d, dict):
                for k in ("mae", "test_mae", "MAE", "test_MAE"):
                    if k in d and isinstance(d[k], (int, float)):
                        test_mae = float(d[k]); break
            if test_mae is not None:
                break

    result = {"seed": args.seed, "test_mae": test_mae}
    with open(Path(cfg.run.results_dir) / "seed_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 56)
    if test_mae is not None:
        print(f"  SEED {args.seed}: TEST MAE = {test_mae:.3f}")
    else:
        print(f"  SEED {args.seed}: TEST MAE not auto-captured — read it "
              f"from the\n  evaluate_for_paper output above and record it "
              f"manually.")
    print(f"  (saved to {cfg.run.results_dir}/seed_result.json)")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
# ============================================================
# ^^^  COPY TO HERE  ^^^
# ============================================================
