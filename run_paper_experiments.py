"""
Paper-Experiment Runner v5 — Electricity ZINB Transformer
==========================================================
Entry point that:
  1. Loads the v5 splits (with peer_avg, seasonality, rare-tariff
     bucketing, physical-rate guard).
  2. Builds the patched model wired to the 5-channel value schema.
  3. Trains with EMA + AMP + early stopping.
  4. Runs the comprehensive paper evaluator on the held-out test set.

Run as:
    python run_paper_experiments.py

Config is split into four dataclasses:
  • RunConfig   — paths, runtime, reproducibility
  • ModelConfig — architecture + ablation knobs
  • TrainConfig — optimization + EMA + early stopping
  • LossConfig  — loaded from peer_avg_config.json with overrides
"""

import json
import argparse
import logging
import glob
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

import Improved_embeding as orig
import electricity_zinb_patches_v5b as p
from paper_evaluator import evaluate_for_paper
from improved_combined_loss_v2 import ImprovedCombinedLoss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Paths, output dirs, reproducibility, runtime."""
    data_dir:            str = "."
    train_npz:           str = "meters_electricity_train.npz"
    val_npz:             str = "meters_electricity_val.npz"
    test_npz:            str = "meters_electricity_test.npz"
    cardinalities_json:  str = "static_cardinalities_ramz.json"
    peer_avg_config:     str = "peer_avg_config.json"
    checkpoint_dir:      str = "checkpoints_paper_s43"
    results_dir:         str = "results_paper"
    seed:                int = 42
    num_workers:         int = 4
    log_interval:        int = 500


@dataclass
class ModelConfig:
    """Architecture + ablation knobs."""
    d_model:                   int   = 192
    n_heads:                   int   = 6
    n_layers:                  int   = 5
    dropout:                   float = 0.05
    n_years:                   int   = 12
    use_time_aware_attention:  bool  = True


@dataclass
class TrainConfig:
    """Optimization + EMA + early stopping + ablation knobs."""
    batch_size:        int   = 256
    lr:                float = 3e-4
    weight_decay:      float = 1e-4
    max_epochs:        int   = 500
    warmup_epochs:     int   = 5
    patience:          int   = 60
    max_grad_norm:     float = 1.0
    ema_decay:         float = 0.995
    use_ema:           bool  = True
    use_augmentation:  bool  = False


@dataclass
class LossConfig:
    """Loss-side hyperparameters."""
    train_target_clip:  Optional[float] = None
    raw_huber_weight:   float = 1.0
    log_huber_weight:   float = 3.0
    zinb_weight:        float = 0.1
    gate_bce_weight:    float = 3.0
    calibration_weight: float = 2.0
    use_zinb_warmup:    bool  = True
    warmup_zinb_epoch:  int   = 20
    zinb_detach_epochs: int   = 20

    @classmethod
    def from_preprocessor_config(
        cls,
        path: str,
        overrides: Optional[dict] = None,
        use_clip: bool = False,
    ) -> "LossConfig":
        """Load train_target_clip from peer_avg_config.json.

        Parameters
        ----------
        path : str
            Path to the preprocessor config JSON file.
        overrides : dict, optional
            Key-value pairs to override after loading.
        use_clip : bool, default=False
            If True, load the recommended training clip value.

        Returns
        -------
        LossConfig
            Configured loss hyperparameters.
        """
        with open(path) as f:
            prep = json.load(f)
        clip = prep["recommended_train_target_clip"] if use_clip else None
        cfg = cls(train_target_clip=clip)
        if overrides:
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    logger.info(f"  unknown LossConfig key: {k}")
        return cfg

    def report(self) -> None:
        """Log the loss configuration."""
        logger.info("Loss config:")
        for k, v in asdict(self).items():
            if k == "train_target_clip":
                logger.info(f"  {k:<20s} = "
                            f"{f'{v:.0f} kWh' if v is not None else 'None (disabled)'}")
            else:
                logger.info(f"  {k:<20s} = {v}")

    def build(self) -> ImprovedCombinedLoss:
        """Construct the loss function instance from this configuration."""
        kwargs = dict(
            raw_huber_weight=self.raw_huber_weight,
            log_huber_weight=self.log_huber_weight,
            zinb_weight=self.zinb_weight,
            gate_bce_weight=self.gate_bce_weight,
            calibration_weight=self.calibration_weight,
            warmup_zinb_epoch=(self.warmup_zinb_epoch
                               if self.use_zinb_warmup else 0),
            zinb_detach_epochs=self.zinb_detach_epochs,
        )
        if self.train_target_clip is not None:
            kwargs["train_target_clip"] = self.train_target_clip
        return ImprovedCombinedLoss(**kwargs)


@dataclass
class ExperimentConfig:
    """Top-level container for all experiment configuration."""
    run:   RunConfig   = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss:  LossConfig  = field(default_factory=LossConfig)

    def report(self) -> None:
        """Log the entire experiment configuration."""
        logger.info("=" * 64)
        logger.info("EXPERIMENT CONFIG")
        logger.info("=" * 64)
        for name, sub in [("run", self.run), ("model", self.model),
                          ("train", self.train)]:
            logger.info(f"\n[{name}]")
            for k, v in asdict(sub).items():
                logger.info(f"  {k:<26s} = {v}")
        logger.info("")
        self.loss.report()
        logger.info("=" * 64)


def setup_run(cfg: ExperimentConfig) -> Tuple[torch.device, Optional[torch.amp.GradScaler]]:
    """
    Prepare directories, set random seeds, and configure device and AMP scaler.

    Parameters
    ----------
    cfg : ExperimentConfig
        Full experiment configuration.

    Returns
    -------
    device : torch.device
        'cuda' if available else 'cpu'.
    scaler : GradScaler or None
        Automatic Mixed Precision scaler if CUDA is used, else None.
    """
    Path(cfg.run.checkpoint_dir).mkdir(exist_ok=True)
    Path(cfg.run.results_dir).mkdir(exist_ok=True)
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    scaler = (torch.amp.GradScaler('cuda') if device.type == "cuda"
              else None)
    return device, scaler


def load_splits(
    cfg: ExperimentConfig,
) -> Tuple[OrderedDict, p.ImprovedElectricityMeterDataset, DataLoader, DataLoader, DataLoader]:
    """
    Load static cardinalities and create datasets/data loaders for train/val/test.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuration containing data paths and batch settings.

    Returns
    -------
    cards : OrderedDict
        Static feature cardinalities.
    train_ds : ImprovedElectricityMeterDataset
        Training dataset.
    train_loader : DataLoader
        Training data loader (shuffled, drop_last).
    val_loader : DataLoader
        Validation data loader.
    test_loader : DataLoader
        Test data loader.
    """
    data_dir = Path(cfg.run.data_dir)
    with open(data_dir / cfg.run.cardinalities_json) as f:
        cards = json.load(f, object_pairs_hook=OrderedDict)
    logger.info(f"cardinalities: {dict(cards)}")

    DS = p.ImprovedElectricityMeterDataset
    train_ds = DS(str(data_dir / cfg.run.train_npz), cards)
    val_ds = DS(str(data_dir / cfg.run.val_npz), cards)
    test_ds = DS(str(data_dir / cfg.run.test_npz), cards)

    logger.info(f"sizes: train={len(train_ds):,}  "
                f"val={len(val_ds):,}  test={len(test_ds):,}")
    logger.info(f"Rate stats: mean={train_ds.rate_mean:.1f}, "
                f"std={train_ds.rate_std:.1f}")

    common = dict(
        batch_size=cfg.train.batch_size,
        collate_fn=orig.collate_fn,
        num_workers=cfg.run.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return cards, train_ds, train_loader, val_loader, test_loader


def build_model(
    cfg: ExperimentConfig,
    cards: OrderedDict,
    default_rate: float,
    device: torch.device,
) -> p.ImprovedZINBElectricityMeterEncoder:
    """
    Instantiate the patched transformer model.

    Parameters
    ----------
    cfg : ExperimentConfig
        Model hyperparameters.
    cards : OrderedDict
        Static feature cardinalities.
    default_rate : float
        Global mean rate (kWh per 30 min) used as fallback.
    device : torch.device
        Device to place the model on.

    Returns
    -------
    model : ImprovedZINBElectricityMeterEncoder
        The constructed model.
    """
    n_years = cfg.model.n_years
    p.orig.AbsoluteTimeEmbedding = lambda d_model: p.AbsoluteTimeEmbedding(
        d_model, n_years=n_years)

    model = p.ImprovedZINBElectricityMeterEncoder(
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        static_cardinalities=cards,
        dropout=cfg.model.dropout,
        use_time_aware_attention=cfg.model.use_time_aware_attention,
        default_rate=default_rate,
    ).to(device)
    n_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    logger.info(f"model parameters: {n_params:,}")
    return model


def build_optimizer(
    model: p.ImprovedZINBElectricityMeterEncoder,
    cfg: ExperimentConfig,
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with layer‑specific learning rates.

    Parameters
    ----------
    model : ImprovedZINBElectricityMeterEncoder
        The model whose parameters will be optimized.
    cfg : ExperimentConfig
        Training configuration (learning rate, weight decay).

    Returns
    -------
    optimizer : torch.optim.Optimizer
        Configured AdamW optimizer.
    """
    lr = cfg.train.lr
    embed_p = list(model.emb.parameters())
    encoder_p = list(model.encoder.parameters())
    static_p = (
        list(model.static_head_embeddings.parameters())
        + list(model.static_ctx_proj.parameters())
        + list(model.default_rate_by_tariff.parameters())
        + list(model.default_rate_by_type.parameters())
        + list(model.default_rate_by_urban.parameters())
        + list(model.default_rate_by_region.parameters())
        + list(model.default_rate_by_phase.parameters())
        + list(model.default_rate_by_amper.parameters())
        # + list(model.baseline_blend_tariff.parameters())
        # + list(model.baseline_blend_urban.parameters())
    )
    segment_p = (
        list(model.tariff_output_log_scale.parameters())
        + list(model.meter_type_output_log_scale.parameters())
        + list(model.segment_gate_bias_tariff.parameters())
        + list(model.segment_gate_bias_urban.parameters())
        + list(model.segment_alpha_bias.parameters())
    )
    head_p = (
        list(model.correction_head.parameters())
        + list(model.scale_head.parameters())
        + list(model.alpha_head.parameters())
        + list(model.gate_head.parameters())
    )
    return torch.optim.AdamW(
        [
            {"params": embed_p, "lr": lr},
            {"params": encoder_p, "lr": lr},
            {"params": static_p, "lr": lr * 3, "weight_decay": 0.0},
            {"params": segment_p, "lr": lr * 5, "weight_decay": 0.01},
            {"params": head_p, "lr": lr * 3},
        ],
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8,
    )


def train(
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
) -> Tuple[torch.nn.Module, Optional[orig.ExponentialMovingAverage]]:
    """
    Execute the training loop with early stopping, EMA, and checkpointing.

    Parameters
    ----------
    cfg : ExperimentConfig
        Training hyperparameters (epochs, patience, etc.).
    model : torch.nn.Module
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : torch.device
        Device for computation.
    scaler : GradScaler or None
        AMP scaler (if CUDA).

    Returns
    -------
    model : torch.nn.Module
        The trained model (best weights restored if early stopping).
    ema : ExponentialMovingAverage or None
        The EMA handler (if used) with shadow weights restored.
    """
    scheduler = orig.WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.max_epochs,
        warmup_lr=1e-7,
        min_lr=1e-6,
    )

    loss_fn = cfg.loss.build()

    early = orig.EarlyStopping(
        patience=cfg.train.patience,
        mode="min",
        restore_best=True,
    )
    grad_acc = orig.GradientAccumulator(1)
    ema = (
        orig.ExponentialMovingAverage(model, cfg.train.ema_decay)
        if cfg.train.use_ema
        else None
    )
    augmenter = (
        orig.TimeSeriesAugmenter(
            noise_std=0.05, scale_range=(0.9, 1.1), mask_prob=0.1
        )
        if cfg.train.use_augmentation
        else None
    )

    training_logger = orig.TrainingLogger(cfg.run.checkpoint_dir, "paper_run")
    training_logger.log_config(
        {
            "run": asdict(cfg.run),
            "model": asdict(cfg.model),
            "train": asdict(cfg.train),
            "loss": asdict(cfg.loss),
        }
    )

    ckpt_dir = Path(cfg.run.checkpoint_dir)

    for epoch in range(cfg.train.max_epochs):
        scheduler.step(epoch)
        lr = scheduler.get_lr()
        tr = orig.train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            grad_acc,
            ema=ema,
            augmenter=augmenter,
            max_grad_norm=cfg.train.max_grad_norm,
            log_interval=cfg.run.log_interval,
            scaler=scaler,
        )
        va = orig.validate(
            model,
            val_loader,
            loss_fn,
            device,
            epoch,
            use_ema=cfg.train.use_ema,
            ema=ema,
        )
        tr["lr"] = lr
        training_logger.log(tr, epoch, "train")
        training_logger.log(va, epoch, "val")
        training_logger.print_summary(epoch, tr, va, lr)

        if early(va["mae"], model, epoch):
            logger.info(
                f"[INFO] early stop at epoch {epoch} "
                f"(best={early.best_score:.4f} @ {early.best_epoch})"
            )
            break
        if early.improved:
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), ckpt_dir / "best_ema.pt")
                ema.restore()

    training_logger.save_history()
    return model, ema


def build_default_config() -> ExperimentConfig:
    """
    Default v5 experiment configuration.

    Loss is the simple working version (raw=1.0, log=3.0, zinb=0.1),
    no target clipping. The two falsified hypotheses from prior runs
    (scale-aware clipping, zinb_weight=0.3) have been retired.

    Returns
    -------
    ExperimentConfig
        Fully configured experiment.
    """
    cfg = ExperimentConfig()
    cfg.loss = LossConfig.from_preprocessor_config(
        cfg.run.peer_avg_config,
        use_clip=False,   # disabled: prior experiments showed it hurts
        overrides={
            "raw_huber_weight": 1.0,
            "log_huber_weight": 3.0,
            "zinb_weight": 0.1,
            "gate_bce_weight": 3.0,
            "calibration_weight": 2.0,
            "warmup_zinb_epoch": 20,
            "use_zinb_warmup": True,
        },
    )
    return cfg


def main() -> None:
    """Parse command-line arguments and run the full experiment pipeline."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="model init + shuffle seed (data split is fixed "
        "in the NPZ and does NOT change)",
    )
    args = ap.parse_args()

    cfg = build_default_config()

    # Apply the seed and isolate output dirs so the runs don't overwrite.
    cfg.run.seed = args.seed
    cfg.run.checkpoint_dir = f"checkpoints_paper_s{args.seed}"
    cfg.run.results_dir = f"results_paper_s{args.seed}"
    logger.info(f"\n*** MULTI-SEED RUN: seed={args.seed} ***")
    logger.info(f"    checkpoints -> {cfg.run.checkpoint_dir}")
    logger.info(f"    results     -> {cfg.run.results_dir}\n")

    cfg.report()

    device, scaler = setup_run(cfg)
    cards, train_ds, train_loader, val_loader, test_loader = load_splits(cfg)
    model = build_model(cfg, cards, train_ds.rate_mean, device)

    # Sanity check
    orig.sanity_check(model, train_loader, device)

    optimizer = build_optimizer(model, cfg)
    model, ema = train(
        cfg, model, train_loader, val_loader, optimizer, device, scaler
    )

    # Final test eval — prefer EMA
    ckpt_dir = Path(cfg.run.checkpoint_dir)
    best_ema = ckpt_dir / "best_ema.pt"
    best = ckpt_dir / "best.pt"
    if best_ema.exists():
        model.load_state_dict(
            torch.load(best_ema, map_location=device, weights_only=True)
        )
        logger.info("[paper-eval] loaded best EMA weights")
    elif best.exists():
        model.load_state_dict(
            torch.load(best, map_location=device, weights_only=True)
        )
        logger.info("[paper-eval] loaded best weights")
    else:
        logger.info("[paper-eval] no checkpoint found — using current weights")

    metrics = evaluate_for_paper(
        model,
        test_loader,
        device,
        out_dir=cfg.run.results_dir,
        static_feature_names=list(cards.keys()),
    )

    # -- capture the test MAE for this seed (robust to evaluator shape) --
    test_mae = None
    # (a) return value is a dict?
    if isinstance(metrics, dict):
        for k in ("mae", "test_mae", "MAE", "test_MAE"):
            if k in metrics and isinstance(metrics[k], (int, float)):
                test_mae = float(metrics[k])
                break
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
                        test_mae = float(d[k])
                        break
            if test_mae is not None:
                break

    result = {"seed": args.seed, "test_mae": test_mae}
    with open(Path(cfg.run.results_dir) / "seed_result.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("\n" + "=" * 56)
    if test_mae is not None:
        logger.info(f"  SEED {args.seed}: TEST MAE = {test_mae:.3f}")
    else:
        logger.info(
            f"  SEED {args.seed}: TEST MAE not auto-captured — read it "
            f"from the\n  evaluate_for_paper output above and record it "
            f"manually."
        )
    logger.info(f"  (saved to {cfg.run.results_dir}/seed_result.json)")
    logger.info("=" * 56 + "\n")


if __name__ == "__main__":
    main()