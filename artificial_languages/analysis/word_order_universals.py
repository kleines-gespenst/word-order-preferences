#!/usr/bin/env python3
"""
Greenberg / Dryer word order universals on the artificial-language LMs.

Fetches the final test perplexity of every trained grammar from Weights & Biases
and tests several typological universals by comparing perplexity across groups of
grammars:

  1      Subject precedes object (SVO/SOV/VSO) vs follows it (VOS/OVS/OSV)  [Greenberg U1]
  3      VSO grammars split by the s4 parameter
  4      SOV grammars split by the s4 parameter
  17     VSO grammars split by adjective order s5 (Adj-N vs N-Adj)          [Greenberg U17]
  dryer  OV/VO order vs relative-clause order s6                           [Dryer 1992]

Run names encode the grammar as e.g. "VSO_s2True_s3False_s4True_s5False_s6True".
The W&B project is read from --entity/--project (or the WANDB_ENTITY / WANDB_PROJECT
environment variables).

Usage:
  python word_order_universals.py                  # run every universal
  python word_order_universals.py --universal 17   # run just one
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import wandb
from scipy import stats

BASE_ORDERS = ["SVO", "SOV", "VSO", "VOS", "OVS", "OSV"]
S_BEFORE_O  = {"SVO", "SOV", "VSO"}   # subject precedes object
OV_ORDERS   = {"OVS", "SOV", "OSV"}   # object precedes verb


def parse_run_name(name):
    """('VSO', {'s2': True, ...}) from 'VSO_s2True_...'; (None, {}) if unparseable."""
    m = re.match(r"^([A-Z]{3})", name)
    if not m or m.group(1) not in BASE_ORDERS:
        return None, {}
    flags = {}
    for i in range(2, 7):
        hit = re.search(rf"s{i}(True|False)", name)
        if hit:
            flags[f"s{i}"] = hit.group(1) == "True"
    return m.group(1), flags


def fetch_runs(entity, project):
    """One row per run: base_order, s2..s6, ppl (final test perplexity)."""
    api = wandb.Api()
    rows = []
    for run in api.runs(f"{entity}/{project}"):
        base, flags = parse_run_name(run.name)
        if base is None:
            continue
        hist = run.history(keys=["final/test_perplexity"], samples=1, pandas=True)
        if not len(hist) or "final/test_perplexity" not in hist.columns:
            continue
        ppl = hist["final/test_perplexity"].iloc[-1]
        if pd.isna(ppl):
            continue
        rows.append({"base_order": base, "ppl": float(ppl), **flags})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No runs with final/test_perplexity found.")
    return df


def _compare(a, b, label_a, label_b, expect=None):
    """Report means + Mann-Whitney U for two perplexity groups; expect = the label
    that word order typology predicts should have the *lower* perplexity."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        print(f"  (missing data: {label_a}={len(a)}, {label_b}={len(b)})")
        return
    print(f"  {label_a:32s} mean PPL = {a.mean():8.4f}  (n={len(a)})")
    print(f"  {label_b:32s} mean PPL = {b.mean():8.4f}  (n={len(b)})")
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    lower = label_a if a.mean() < b.mean() else label_b
    print(f"  lower perplexity: {lower}   (Mann-Whitney U={u:.0f}, p={p:.3g})")
    if expect is not None:
        print(f"  -> {'SUPPORTS' if lower == expect else 'CONTRADICTS'} "
              f"(typology predicts lower: {expect})")


def universal_1(df):
    print("\n[Universal 1] Subject before vs after object")
    before = df[df.base_order.isin(S_BEFORE_O)].ppl
    after  = df[~df.base_order.isin(S_BEFORE_O)].ppl
    _compare(before, after, "S before O (SVO/SOV/VSO)", "S after O (VOS/OVS/OSV)",
             expect="S before O (SVO/SOV/VSO)")


def _split_by_flag(df, base, flag, labels, name, expect=None):
    print(f"\n[{name}] {base} grammars split by {flag}")
    sub = df[df.base_order == base]
    if flag not in sub.columns:
        print(f"  (no {flag} data for {base})")
        return
    _compare(sub[sub[flag] == False].ppl, sub[sub[flag] == True].ppl,
             labels[0], labels[1], expect=expect)


def universal_3(df):
    _split_by_flag(df, "VSO", "s4", ("s4=False", "s4=True"), "Universal 3")


def universal_4(df):
    _split_by_flag(df, "SOV", "s4", ("s4=False", "s4=True"), "Universal 4")


def universal_17(df):
    # s5=False -> NP: Adj Noun; s5=True -> NP: Noun Adj. VSO is predicted to prefer N-Adj.
    _split_by_flag(df, "VSO", "s5", ("NP: Adj Noun (s5=False)", "NP: Noun Adj (s5=True)"),
                   "Universal 17", expect="NP: Noun Adj (s5=True)")


def dryer(df):
    # s6=False -> Rel-VP Noun (RelN); s6=True -> Noun Rel-VP (NRel).
    # Harmonic (Dryer): OV pairs with RelN, VO pairs with NRel.
    print("\n[Dryer] OV/VO order vs relative-clause order (s6)")
    if "s6" not in df.columns:
        print("  (no s6 data)")
        return
    ov = df.base_order.isin(OV_ORDERS)
    harmonic = (ov & (df.s6 == False)) | (~ov & (df.s6 == True))
    _compare(df[harmonic].ppl, df[~harmonic].ppl,
             "harmonic (OV+RelN, VO+NRel)", "disharmonic",
             expect="harmonic (OV+RelN, VO+NRel)")


UNIVERSALS = {"1": universal_1, "3": universal_3, "4": universal_4,
              "17": universal_17, "dryer": dryer}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universal", choices=list(UNIVERSALS) + ["all"], default="all")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"),
                    help="W&B entity (default: $WANDB_ENTITY)")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT"),
                    help="W&B project (default: $WANDB_PROJECT)")
    args = ap.parse_args()
    if not args.entity or not args.project:
        ap.error("set --entity/--project or the WANDB_ENTITY / WANDB_PROJECT env vars")

    df = fetch_runs(args.entity, args.project)
    print(f"Fetched {len(df)} runs from {args.entity}/{args.project}")
    todo = UNIVERSALS if args.universal == "all" else {args.universal: UNIVERSALS[args.universal]}
    for fn in todo.values():
        fn(df)


if __name__ == "__main__":
    main()
