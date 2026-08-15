#!/usr/bin/env python3
"""Print per-size FLORES word order BPEC stats: N per order, median BPEC per order,
deltas from the best order, and the SVO-vs-SOV gap, on the languages common to all
four Goldfish training sizes."""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

ENCODING_CSV  = DATA_ROOT / "flores_clustering/flores200_encoding.csv"
GOLDFISH_BASE = DATA_ROOT / "results_natural_langs_flores"
SIZES       = ["5mb", "10mb", "100mb", "1000mb"]
SIZE_LABELS = ["5 MB", "10 MB", "100 MB", "1000 MB"]
ORDER_DISPLAY = ["SVO", "SOV", "VSO", "VOS", "OVS", "OSV", "NoDominant"]

def load_encoding():
    df = pd.read_csv(ENCODING_CSV)[["flores_code", "base_word_order"]]
    return df[df["base_word_order"].isin(ORDER_DISPLAY)]

def load_size(size, enc):
    records = []
    for path in (GOLDFISH_BASE / size).glob("*_devtest.json"):
        code = path.stem.replace("_devtest", "")
        with open(path) as f:
            records.append({"flores_code": code, "bpec": json.load(f)["bpec"]})
    return pd.DataFrame(records).merge(enc, on="flores_code", how="inner")

enc      = load_encoding()
datasets_all = [load_size(s, enc) for s in SIZES]

# common languages (present in all 4 sizes)
common = set(datasets_all[0]["flores_code"])
for df in datasets_all[1:]:
    common &= set(df["flores_code"])
print(f"Common languages: {len(common)}")

datasets = [df[df["flores_code"].isin(common)].copy() for df in datasets_all]

print("N per word order per size")
for sl, df in zip(SIZE_LABELS, datasets):
    counts = df.groupby("base_word_order").size().reindex(ORDER_DISPLAY).dropna().astype(int)
    total  = len(df)
    parts  = ", ".join(f"{wo}: {n}" for wo, n in counts.items())
    print(f"  {sl:10s}  total={total:3d}   {parts}")

print("Median BPEC per word order per size (lower = better)")
focus = ["SVO", "SOV", "VSO", "VOS", "NoDominant"]
for sl, df in zip(SIZE_LABELS, datasets):
    medians = {wo: df[df["base_word_order"] == wo]["bpec"].median()
               for wo in focus if wo in df["base_word_order"].values}
    best_wo  = min(medians, key=medians.get)
    best_med = medians[best_wo]
    print(f"\n  {sl}")
    for wo, med in sorted(medians.items(), key=lambda x: x[1]):
        marker = " ← best" if wo == best_wo else ""
        print(f"    {wo:12s}  median={med:.4f}{marker}")

print("Delta: best vs each other order")
for sl, df in zip(SIZE_LABELS, datasets):
    medians = {wo: df[df["base_word_order"] == wo]["bpec"].median()
               for wo in focus if wo in df["base_word_order"].values}
    best_wo  = min(medians, key=medians.get)
    best_med = medians[best_wo]
    print(f"\n  {sl}  (best: {best_wo} = {best_med:.4f})")
    for wo, med in sorted(medians.items(), key=lambda x: x[1]):
        if wo != best_wo:
            print(f"    Δ({wo} − {best_wo}) = {med - best_med:+.4f}")

print("SVO vs SOV delta specifically")
for sl, df in zip(SIZE_LABELS, datasets):
    svo = df[df["base_word_order"] == "SVO"]["bpec"].median()
    sov = df[df["base_word_order"] == "SOV"]["bpec"].median()
    print(f"  {sl:10s}  SVO={svo:.4f}  SOV={sov:.4f}  Δ(SOV−SVO)={sov-svo:+.4f}")
