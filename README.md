# Zero-Inflated Negative-Binomial Transformer for Irregular, Human-Read Electricity-Meter Forecasting

Code for the paper *"A Zero-Inflated Negative-Binomial Transformer for
Forecasting Irregular, Human-Read Electricity-Meter Consumption."*

The model forecasts per-meter electricity consumption from irregularly
spaced, manually-read meter sequences. It combines a transformer
sequence encoder, a zero-inflated negative-binomial (ZINB) output head,
and a frozen per-tariff generalised-linear prior, fed by a
domain-specific preprocessing pipeline (peer-average empirical-Bayes
shrinkage, quarterly aggregation, rare-category bucketing, seasonality
encoding).

## ⚠️ Data availability

The underlying meter dataset is **proprietary** (real utility consumption
records) and is **not** included in this repository. The `.npz` data
files and derived aggregates are git-ignored. To reproduce results you
must supply your own data in the expected format (see
`electricity_preprocessor_v5b.py` and `split_dataset.py` for the schema:
per-meter sequences of `[consumption, dt, ...]` value channels with
validity masks, plus seven static categorical attributes).

## Repository layout

### Model and training
- `Improved_embeding.py` — the model: four-stream input embedding,
  transformer encoder (optional time-aware attention bias), ZINB heads,
  per-tariff GLM baseline.
- `improved_combined_loss_v2.py` — composite training loss
  (robust regression + log-space + ZINB NLL + gate BCE + calibration).
- `run_paper_experiments.py` — main training entry point.

### Preprocessing
- `electricity_preprocessor_v5b.py` — the v5b preprocessing pipeline
  (the configuration reported in the paper).
- `electricity_zinb_patches_v5b.py` — associated patches.
- `split_dataset.py` — meter-level train/val/test split.

### Baselines and ablations
- `lightgbm_baseline.py` — gradient-boosting baseline on the same features.
- `run_ablation_no_timebias.py` — ablation: transformer without the
  time-aware attention bias.
- `run_gru_baseline.py` — GRU sequence-encoder baseline.
- `eval_gru.py` — standalone test-MAE evaluation for the GRU checkpoint.

### Evaluation and analysis
- `v5b_segmented_eval.py` — segmented evaluation (by gap, volatility,
  history) + baseline comparison + triage inputs.
- `deployment_reframing.py` — deployment-triage table and figure.
- `report_dataset_stats.py` — dataset statistics + characterisation figure.
- `make_prediction_figures.py` — predicted-vs-actual and example
  trajectory figures.
- `aggregate_seeds.py` — multi-seed mean ± std aggregation.
- `diagnose_meter.py` — per-meter head-output diagnostic.

### Config
- `static_cardinalities_ramz.json`, `tariff_code_to_index.json`,
  `peer_avg_config.json` — schema/config files.

## Reproducing the main result

```bash
# 1. Preprocess your data into the expected .npz format
python split_dataset.py            # (adapt paths to your data)

# 2. Train the main model (repeat with --seed 43, 44 for multi-seed)
python run_paper_experiments.py    # writes checkpoints_paper/ and results_paper/

# 3. Evaluate (segmented + baselines)
python v5b_segmented_eval.py --checkpoint checkpoints_paper/best_ema.pt \
    --agg peer_avg_aggregates.pkl

# 4. Baselines / ablations
python lightgbm_baseline.py
python run_ablation_no_timebias.py --seed 42
python run_gru_baseline.py --seed 42
python eval_gru.py --checkpoint checkpoints_gru/best_ema.pt

# 5. Aggregate multi-seed results
python aggregate_seeds.py --vals 129.6 130.1 129.7
```

## Headline results (test set, 887,108 readings)

| Predictor | MAE (kWh) | RMSE (kWh) |
|---|---|---|
| Persistence | 188.6 | 660.7 |
| LightGBM | 146.7 | 563.9 |
| GRU encoder | 135.5 | 599.6 |
| Transformer (no time-aware) | 129.9 | 531 |
| **ZINB transformer (ours)** | **129.8 ± 0.3** | **527** |

## Requirements

Python 3.12, PyTorch, NumPy, scikit-learn, LightGBM, matplotlib.
(A `requirements.txt` / environment file should be added before release.)

## Citation

```bibtex
@article{TODO,
  title   = {A Zero-Inflated Negative-Binomial Transformer for Forecasting
             Irregular, Human-Read Electricity-Meter Consumption},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```
