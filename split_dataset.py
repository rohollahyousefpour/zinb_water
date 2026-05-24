"""
Train / Validation / Test Split — Electricity Meters
====================================================

Two strategies (configurable):

  (A) temporal_per_meter   — for each meter, the EARLIEST 80% of its
                             readings → train, next 10% → val,
                             last 10% → test.   ✔ default; preferred
                             for forecasting evaluation; this is what
                             reviewers will expect at TMLR.

  (B) by_meter             — random split of meters into 70/15/15;
                             each meter goes entirely to one split.
                             Useful only as a robustness check.

Output: one NPZ per split with the same schema as the preprocessor's
output, so the existing ImprovedWaterMeterDataset (rename!) can load
them directly.

Usage:
    python split_dataset.py meters_electricity_ready.npz --mode temporal
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def temporal_per_meter_split(
    values: np.ndarray,
    times: np.ndarray,
    masks: np.ndarray,
    abs_dates: Optional[np.ndarray],
    frac_train: float = 0.80,
    frac_val: float = 0.10,
) -> Tuple[Dict[str, List[List[np.ndarray]]], List[List[int]]]:
    """
    Split each meter's reading sequence chronologically.

    For every meter, the sequence is divided into three consecutive blocks:
        - training: first `frac_train` fraction of readings
        - validation: next `frac_val` fraction
        - test: remaining readings

    Meters with too few readings (<5 total or <1 test sample) are dropped.

    Parameters
    ----------
    values : np.ndarray
        Array of meter value sequences (object dtype, each element is 1D array).
    times : np.ndarray
        Array of time step sequences (object dtype).
    masks : np.ndarray
        Array of mask sequences (object dtype, 1=observed, 0=missing).
    abs_dates : Optional[np.ndarray]
        Array of absolute date sequences (object dtype) or None.
    frac_train : float, default=0.80
        Fraction of each meter's data to assign to training.
    frac_val : float, default=0.10
        Fraction of each meter's data to assign to validation.

    Returns
    -------
    out : dict of lists
        Keys: "times", "values", "masks", "abs_dates" (if provided).
        Each value is a list of three lists (train, val, test) containing
        the sequence arrays for each split.
    keep_idx : list of three lists
        For each split (train, val, test), the original meter indices that
        contributed at least one reading to that split. Useful for subsetting
        static features.
    """
    out = {k: ([], [], []) for k in ("times", "values", "masks", "abs_dates")}
    keep_idx = [[], [], []]   # train, val, test → meter indices
    keep_lens = [[], [], []]  # number of readings kept per split per meter

    for i in range(len(values)):
        T = values[i].shape[0]
        if T < 5:
            # Skip meters that are too short to split meaningfully
            continue

        # Compute split sizes
        n_tr = max(1, int(np.floor(T * frac_train)))
        n_va = max(1, int(np.floor(T * frac_val)))
        n_te = T - n_tr - n_va
        if n_te < 1:
            # Not enough readings left for a test set
            continue

        # Define slices for train, validation, test
        splits = [
            slice(0, n_tr),                     # train
            slice(n_tr, n_tr + n_va),           # validation
            slice(n_tr + n_va, T),              # test
        ]

        for s, sl in enumerate(splits):
            out["times"][s].append(times[i][sl])
            out["values"][s].append(values[i][sl])
            out["masks"][s].append(masks[i][sl])
            if abs_dates is not None:
                out["abs_dates"][s].append(abs_dates[i][sl])

            keep_idx[s].append(i)
            keep_lens[s].append(sl.stop - sl.start)

    # Remove unused keys if abs_dates was None
    if abs_dates is None:
        del out["abs_dates"]

    return out, keep_idx


def by_meter_split(
    N: int,
    seed: int = 0,
    frac_train: float = 0.70,
    frac_val: float = 0.15,
) -> Dict[str, List[int]]:
    """
    Split meters randomly into train/val/test sets.

    Each meter is assigned wholly to one split. No chronological ordering
    is applied within meters.

    Parameters
    ----------
    N : int
        Total number of meters.
    seed : int, default=0
        Random seed for reproducibility.
    frac_train : float, default=0.70
        Fraction of meters assigned to training.
    frac_val : float, default=0.15
        Fraction of meters assigned to validation.

    Returns
    -------
    dict
        Keys: "train", "val", "test". Values are lists of meter indices.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_tr = int(frac_train * N)
    n_va = int(frac_val * N)

    return {
        "train": perm[:n_tr].tolist(),
        "val": perm[n_tr:n_tr + n_va].tolist(),
        "test": perm[n_tr + n_va:].tolist(),
    }


def values_array_subset(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Extract a subset of an object array using indices.

    Parameters
    ----------
    arr : np.ndarray
        Object array where each element is a sequence (e.g., values, times).
    idx : np.ndarray
        Integer indices of meters to keep.

    Returns
    -------
    np.ndarray
        New object array containing only the selected meters.
    """
    return np.array([arr[i] for i in idx], dtype=object)


def save_split(
    npz_in: str,
    mode: str = "temporal",
    out_prefix: str = "split",
    seed: int = 0,
    frac_train: float = 0.80,
    frac_val: float = 0.10,
) -> None:
    """
    Load preprocessed NPZ file and save three split NPZ files.

    The output NPZs have exactly the same structure as the input,
    making them directly compatible with the dataset class.

    Parameters
    ----------
    npz_in : str
        Path to input NPZ file produced by preprocessor.
    mode : str, default="temporal"
        Splitting strategy: "temporal" (chronological per meter) or "by_meter".
    out_prefix : str, default="split"
        Prefix for output files (e.g., "split_train.npz").
    seed : int, default=0
        Random seed for "by_meter" mode.
    frac_train : float, default=0.80
        Fraction for training split (temporal mode: per meter; by_meter: total meters).
    frac_val : float, default=0.10
        Fraction for validation split.
    """
    data = np.load(npz_in, allow_pickle=True)
    values = data["values"]
    times = data["times"]
    masks = data["masks"]
    static = data["static"]
    ramz = data["ramz"]
    abs_dates = data["abs_dates"] if "abs_dates" in data.files else None
    N = len(values)

    logger.info(f"splitting N={N:,} meters in mode={mode} ...")
    split_names = ["train", "val", "test"]

    if mode == "temporal":
        # Perform chronological split per meter
        bundles, keep_idx = temporal_per_meter_split(
            values, times, masks, abs_dates,
            frac_train=frac_train, frac_val=frac_val,
        )

        for s, name in enumerate(split_names):
            idx = np.asarray(keep_idx[s], dtype=np.int64)

            payload = {
                "times": np.array(bundles["times"][s], dtype=object),
                "values": np.array(bundles["values"][s], dtype=object),
                "masks": np.array(bundles["masks"][s], dtype=object),
                "static": static[idx],
                "ramz": ramz[idx],
            }
            if abs_dates is not None:
                payload["abs_dates"] = np.array(bundles["abs_dates"][s], dtype=object)

            out_file = f"{out_prefix}_{name}.npz"
            np.savez_compressed(out_file, **payload)
            logger.info(f"   {name:5s}: {len(idx):,} meters -> {out_file}")

    elif mode == "by_meter":
        # Split meters randomly
        idx_map = by_meter_split(
            N, seed=seed,
            frac_train=frac_train, frac_val=frac_val,
        )

        for name in split_names:
            idx = np.asarray(idx_map[name], dtype=np.int64)

            payload = {
                "times": values_array_subset(times, idx),
                "values": values_array_subset(values, idx),
                "masks": values_array_subset(masks, idx),
                "static": static[idx],
                "ramz": ramz[idx],
            }
            if abs_dates is not None:
                payload["abs_dates"] = values_array_subset(abs_dates, idx)

            out_file = f"{out_prefix}_{name}.npz"
            np.savez_compressed(out_file, **payload)
            logger.info(f"   {name:5s}: {len(idx):,} meters -> {out_file}")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    logger.info("split complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split preprocessed electricity meter data into train/val/test."
    )
    parser.add_argument("npz_in", help="Input NPZ file from preprocessor")
    parser.add_argument(
        "--mode",
        default="temporal",
        choices=["temporal", "by_meter"],
        help="Splitting strategy: 'temporal' (chronological per meter, default) or "
             "'by_meter' (random assignment of whole meters).",
    )
    parser.add_argument(
        "--out_prefix",
        default="split",
        help="Prefix for output files (e.g., 'split_train.npz').",
    )
    parser.add_argument(
        "--frac_train",
        type=float,
        default=0.80,
        help="Fraction for training (temporal: per meter; by_meter: of meters).",
    )
    parser.add_argument(
        "--frac_val",
        type=float,
        default=0.10,
        help="Fraction for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for 'by_meter' mode.",
    )
    args = parser.parse_args()

    save_split(
        args.npz_in,
        mode=args.mode,
        out_prefix=args.out_prefix,
        seed=args.seed,
        frac_train=args.frac_train,
        frac_val=args.frac_val,
    )