#!/usr/bin/env python3
"""
Bootstrap SOV-SVO BPEC gaps for languages common across all Goldfish sizes.

Outputs:
  <DATA_ROOT>/flores_clustering/bpec_svo_sov_bootstrap_common.csv
  <DATA_ROOT>/flores_clustering/bpec_svo_sov_bootstrap_common.txt
"""

import csv
import json
import os
import random
import statistics
from pathlib import Path


DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

ENCODING_CSV = DATA_ROOT / "flores_clustering/flores200_encoding.csv"
GOLDFISH_BASE = DATA_ROOT / "results_natural_langs_flores"
OUT_CSV = DATA_ROOT / "flores_clustering/bpec_svo_sov_bootstrap_common.csv"
OUT_TXT = DATA_ROOT / "flores_clustering/bpec_svo_sov_bootstrap_common.txt"

SIZES = ["5mb", "10mb", "100mb", "1000mb"]
SIZE_LABELS = ["5 MB", "10 MB", "100 MB", "1000 MB"]
N_BOOT = 50000
SEED = 20260510


def percentile(sorted_values, q):
    """Linear-interpolated percentile for q in [0, 1]."""
    if not sorted_values:
        raise ValueError("percentile() needs at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def load_word_orders():
    with ENCODING_CSV.open(newline="") as f:
        return {
            row["flores_code"]: row["base_word_order"]
            for row in csv.DictReader(f)
            if row["base_word_order"] in {"SVO", "SOV"}
        }


def load_size(size, word_orders):
    records = []
    for path in (GOLDFISH_BASE / size).glob("*_devtest.json"):
        code = path.name.replace("_devtest.json", "")
        order = word_orders.get(code)
        if order is None:
            continue
        with path.open() as f:
            bpec = float(json.load(f)["bpec"])
        records.append({"flores_code": code, "base_word_order": order, "bpec": bpec})
    return records


def bootstrap_median_gap(svo, sov, rng):
    diffs = []
    for _ in range(N_BOOT):
        svo_sample = [rng.choice(svo) for _ in svo]
        sov_sample = [rng.choice(sov) for _ in sov]
        diffs.append(statistics.median(sov_sample) - statistics.median(svo_sample))
    diffs.sort()
    return {
        "ci_low": percentile(diffs, 0.025),
        "ci_high": percentile(diffs, 0.975),
        "boot_mean": statistics.mean(diffs),
        "p_le_zero": sum(d <= 0 for d in diffs) / len(diffs),
        "p_ge_zero": sum(d >= 0 for d in diffs) / len(diffs),
    }


def main():
    rng = random.Random(SEED)
    word_orders = load_word_orders()
    datasets = [load_size(size, word_orders) for size in SIZES]

    common = set(row["flores_code"] for row in datasets[0])
    for rows in datasets[1:]:
        common &= set(row["flores_code"] for row in rows)

    results = []
    for size, label, rows in zip(SIZES, SIZE_LABELS, datasets):
        svo = sorted(row["bpec"] for row in rows
                     if row["flores_code"] in common and row["base_word_order"] == "SVO")
        sov = sorted(row["bpec"] for row in rows
                     if row["flores_code"] in common and row["base_word_order"] == "SOV")
        obs = statistics.median(sov) - statistics.median(svo)
        boot = bootstrap_median_gap(svo, sov, rng)
        results.append({
            "size": size,
            "label": label,
            "n_svo": len(svo),
            "n_sov": len(sov),
            "median_svo": statistics.median(svo),
            "median_sov": statistics.median(sov),
            "median_diff_sov_minus_svo": obs,
            **boot,
            "effective_choices_ratio": 2 ** obs,
            "effective_choices_pct_increase": (2 ** obs - 1) * 100,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    with OUT_TXT.open("w") as f:
        f.write(f"Bootstrap SOV-SVO BPEC median gaps, common languages only\n")
        f.write(f"N_BOOT = {N_BOOT}, seed = {SEED}\n")
        f.write(f"Common SVO/SOV languages: {results[0]['n_svo']} SVO, {results[0]['n_sov']} SOV\n\n")
        for row in results:
            f.write(
                f"{row['label']}: diff={row['median_diff_sov_minus_svo']:+.6f}, "
                f"95% CI [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}], "
                f"P(diff <= 0)={row['p_le_zero']:.4f}, "
                f"2^diff={row['effective_choices_ratio']:.4f} "
                f"({row['effective_choices_pct_increase']:+.2f}%)\n"
            )

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_TXT}")
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
