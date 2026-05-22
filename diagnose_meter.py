"""
Head-output diagnostic for a single meter.
Confirms WHY a near-zero meter is being over-predicted by dumping the
gate (pi), the mean (mu), the baseline, and the final (1-pi)*mu for
every reading of one meter.

Run on the meter id that looked wrong in example_trajectories.png
(e.g. 17232).

Usage:
    python diagnose_meter.py --checkpoint checkpoints_paper_s42/best_ema.pt \
        --agg peer_avg_aggregates.pkl --meter 17232
"""
import argparse
import numpy as np
import torch

# This reuses the loading machinery from your eval script. Adjust the
# import to match wherever build_reading_records / load_model live.
from v5b_segmented_eval import (  # noqa
    load_everything,   # however your eval exposes model + data; rename as needed
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--agg", required=True)
    ap.add_argument("--meter", type=int, required=True)
    args = ap.parse_args()

    # --- You will likely need to adapt these two lines to your eval API. ---
    # The goal: get the model and the single-meter batch, run forward, and
    # read out["gate"], out["mu"], out["baseline_mu"], out["target"], mask.
    model, batches, meta = load_everything(args.checkpoint, args.agg)

    model.eval()
    with torch.no_grad():
        for batch in batches:
            ids = batch["meter_ids"].cpu().numpy()  # adapt key name
            if args.meter not in ids:
                continue
            out = model(batch)
            row = int(np.where(ids == args.meter)[0][0])
            m = out["mask"][row].bool().cpu().numpy()
            pi = out["gate"][row].cpu().numpy()[m]
            mu = out["mu"][row].cpu().numpy()[m]
            base = out.get("baseline_mu", out["mu"])[row].cpu().numpy()[m]
            tgt = out["target"][row].cpu().numpy()[m]
            pred = (1 - pi) * mu

            print(f"\nMeter {args.meter}: {m.sum()} valid readings")
            print(f"{'idx':>3} {'target':>9} {'pred':>9} "
                  f"{'pi(gate)':>9} {'mu':>9} {'baseline':>9}")
            for i in range(len(tgt)):
                print(f"{i:>3} {tgt[i]:>9.1f} {pred[i]:>9.1f} "
                      f"{pi[i]:>9.3f} {mu[i]:>9.1f} {base[i]:>9.1f}")

            print("\n--- summary ---")
            print(f"  mean target      : {tgt.mean():.2f}")
            print(f"  mean prediction  : {pred.mean():.2f}")
            print(f"  mean gate (pi)   : {pi.mean():.3f}  "
                  f"(near 1 => predicting zero; near 0 => predicting nonzero)")
            print(f"  mean mu          : {mu.mean():.2f}")
            print(f"  mean baseline    : {base.mean():.2f}")
            frac_zero = float((tgt < 0.5).mean())
            print(f"  frac target zero : {frac_zero:.2f}")
            print("\nINTERPRETATION:")
            if frac_zero > 0.5 and pi.mean() < 0.5:
                print("  -> Gate is NOT firing on a mostly-zero meter:")
                print("     the zero-inflation head is under-confident here.")
            if base.mean() > 5 * max(tgt.mean(), 0.1):
                print("  -> Baseline/default-rate FLOOR is far above the")
                print("     true level: the default rate is propping up mu.")
            return
    print(f"Meter {args.meter} not found in any batch.")


if __name__ == "__main__":
    main()
