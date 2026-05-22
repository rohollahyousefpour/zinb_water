# Data Pipeline — What Changed and How to Run It

This pack replaces three files from your previous workflow:

| Old (water-named, buggy)              | New (clean, electricity) |
|---------------------------------------|--------------------------|
| `WaterMeterDataPreprocessor.py`       | `electricity_preprocessor.py` |
| `convert_to_numy.py`                  | folded into `eda_electricity.py` (was redundant) |
| no split                              | `split_dataset.py` |
| no EDA                                | `eda_electricity.py` |

## Bugs fixed (from the analysis above the code)

| # | Bug                                                                | Where it lived                             | Fix |
|---|--------------------------------------------------------------------|--------------------------------------------|-----|
| 1 | `compute_global_peer_averages` defined but never called            | `preprocess_large_csv_parallel`            | Pipeline now calls it in Pass 1 and saves a 3rd value channel `peer_avg`. |
| 2 | NPZ schema mismatch between preprocessor and `convert_to_numy.py`  | both files                                 | Unified schema: `values=(T,3)`, plus `abs_dates` field. |
| 3 | First reading of every meter silently corrupted (`rate=0, dt=0`)   | `process_subchunk`                         | Configurable `FIRST_READ_POLICY = "drop" \| "impute"`, default `"drop"`. |
| 4 | Categorical encoding order-dependent across chunks                 | `encode_static`                            | New `build_global_cat_maps()` pre-pass with explicit `<UNK>=0`. |
| 5 | First sub-chunk processed twice (debug leftover)                   | `process_chunk_parallel`                   | Removed. |
| 6 | No minimum-reads filter                                            | `process_subchunk`                         | `MIN_VALID_READS = 4` (configurable). |
| 7 | No train / val / test split                                        | missing                                    | `split_dataset.py` — temporal-per-meter (preferred) or by-meter. |
| 8 | No EDA / dataset statistics                                        | missing                                    | `eda_electricity.py` generates JSON + markdown + figures. |

## Run order

```bash
# 1) preprocess wide CSV → ragged NPZ (Passes 0, 1, 2)
python electricity_preprocessor.py
# produces: meters_electricity_ready.npz, static_cardinalities.json

# 2) compute all stats needed for the paper
python eda_electricity.py meters_electricity_ready.npz
# produces: eda_summary.json, eda_summary.md, eda_figures/*.png

# 3) split temporally for the experiments
python split_dataset.py meters_electricity_ready.npz --mode temporal
# produces: split_train.npz, split_val.npz, split_test.npz
```

## Three things to verify on your side after the first run

1. **`eda_summary.md` headline numbers look sane.**
   Expected for residential electricity in Mazandaran, bimonthly:
   - median gap ≈ 55–65 days
   - zero rate: probably 5–15% (driven by Caspian vacation homes)
   - dispersion index: should be ≥ 5, likely much higher
   If any number looks impossible (e.g. zero rate > 50% or median gap < 10), tell me — likely a data-loading issue.

2. **`tariff_code` cardinality.** Your model file expects 72.
   After the preprocessor's Pass 0, `static_cardinalities.json` will tell you the real count (could be more if rare codes only show up later in the CSV). Update the model accordingly.

3. **Geography.** Your `region_in` has cardinality 2, which is too coarse if you have two cities and also rural vs urban. Check whether city is captured by `region_in` or whether you need to add a `city` column. For the paper we want to be able to report per-city results.

## What's still missing (please confirm)

- **Static numeric features**: `STATIC_NUM_COLS = []`. Do you have any numeric static features (e.g., contracted load, customer age, building floor area)? If yes, add them — they will improve the model.
- **Holiday / Nowruz calendar**: Iranian calendar effects (Nowruz, Ramadan, summer migration) are likely material for Mazandaran. Worth adding as a binary feature at the embedding stage.

## What I need next to draft the methods section

Once you've run these three scripts, send me back:

- `eda_summary.md` (or the headline numbers from it)
- `static_cardinalities.json`
- Confirmation of `region_in` / city encoding
- Which baselines you have results for: any subset of
  {ARIMA, Prophet, LSTM-NB, DeepAR-NB, vanilla Transformer-ZINB, mTAN, GRU-D}

With those in hand I can:
- Fill in **Paper 1 §3 (Background)** with real numbers
- Draft **Paper 1 §4 (Method)** from your model file (renaming Water → Electricity throughout)
- Draft **Paper 1 §5 (Experimental Setup)**
