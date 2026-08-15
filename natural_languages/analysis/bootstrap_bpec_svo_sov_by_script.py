#!/usr/bin/env python3
"""
Script-controlled SOV-SVO BPEC gap analysis.

Real languages differ in script/orthography, which could confound the SOV-vs-SVO
BPEC comparison. This re-runs the SOV-SVO median-gap bootstrap on script-restricted
subsets so all compared languages share a common orthographic style:

  * "romanized"  : Latin script only            (Latn)
  * "alphabetic" : segmental alphabets           (Latn, Grek, Cyrl, Armn, Geor)
  * "all"        : full set (reference / baseline)

Covers both the Goldfish monolingual models (4 training sizes, languages common to
all sizes, mirroring bootstrap_bpec_svo_sov_common.py) and the multilingual models.

Outputs to <BASE>/flores_clustering/:
  bpec_svo_sov_by_script.csv   (one row per model/size x subset)
  bpec_svo_sov_by_script.txt   (human-readable summary)
  bpec_svo_sov_by_script_langs.txt  (per-subset language lists for transparency)
"""

import csv
import json
import os
import random
import statistics
from pathlib import Path

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

BASE = DATA_ROOT
ENCODING_CSV = BASE / "flores_clustering" / "flores200_encoding.csv"
GOLDFISH_BASE = BASE / "results_natural_langs_flores"
OUT_CSV = BASE / "flores_clustering" / "bpec_svo_sov_by_script.csv"
OUT_TXT = BASE / "flores_clustering" / "bpec_svo_sov_by_script.txt"
OUT_LANGS = BASE / "flores_clustering" / "bpec_svo_sov_by_script_langs.txt"

SIZES = ["5mb", "10mb", "100mb", "1000mb"]
SIZE_LABELS = {"5mb": "5 MB", "10mb": "10 MB", "100mb": "100 MB", "1000mb": "1000 MB"}

# Script subsets. Alphabetic = true segmental alphabets (separate vowel+consonant
# letters). Excludes abjads (Arab, Hebr), abugidas (Deva, Beng, Ethi, Taml, ...),
# and syllabaries/logographies (Hani/Hans/Hant, Jpan, Hang).
LATIN = {"Latn"}
ALPHABETIC = {"Latn", "Grek", "Cyrl", "Armn", "Geor"}
SUBSETS = ["all", "romanized", "alphabetic"]

MULTILINGUAL = [
    ("BLOOM-560m", "results_bloom_flores"),
    ("BLOOM-1b7", "results_bloom_1b7_flores"),
    ("BLOOM-3b", "results_bloom_3b_flores"),
    ("BLOOM-7b1", "results_bloom_7b1_flores"),
    ("mGPT", "results_mgpt_flores"),
    ("mGPT-13b", "results_mgpt13b_flores"),
    ("XGLM-564m", "results_xglm_564m_flores"),
    ("XGLM-1.7b", "results_xglm_1b7_flores"),
    ("Llama-3.1-8B", "results_llama3_1_8b_flores"),
    ("Mistral-7B-v0.3", "results_mistral_7b_v03_flores"),
    ("Qwen2.5-7B", "results_qwen2_5_7b_flores"),
    ("tiny-aya", "results_tiny_aya_flores"),
]

N_BOOT = 50000
SEED = 20260510


def script_of(code):
    return code.split("_")[1] if "_" in code else "?"


def in_subset(code, subset):
    if subset == "all":
        return True
    if subset == "romanized":
        return script_of(code) in LATIN
    if subset == "alphabetic":
        return script_of(code) in ALPHABETIC
    raise ValueError(subset)


def percentile(sorted_values, q):
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


def load_goldfish_size(size, word_orders):
    recs = {}
    for path in (GOLDFISH_BASE / size).glob("*_devtest.json"):
        code = path.name.replace("_devtest.json", "")
        if code not in word_orders:
            continue
        recs[code] = float(json.load(path.open())["bpec"])
    return recs


def load_multilingual(results_dir, word_orders):
    recs = {}
    for path in (BASE / results_dir).glob("bpec_*_devtest.json"):
        code = path.name.replace("bpec_", "").replace("_devtest.json", "")
        if code not in word_orders:
            continue
        b = json.loads(path.read_text()).get("bpec")
        if b is not None:
            recs[code] = float(b)
    return recs


def bootstrap_gap(svo, sov, rng):
    """Median(SOV) - Median(SVO) with 95% bootstrap CI."""
    diffs = []
    for _ in range(N_BOOT):
        s_svo = [rng.choice(svo) for _ in svo]
        s_sov = [rng.choice(sov) for _ in sov]
        diffs.append(statistics.median(s_sov) - statistics.median(s_svo))
    diffs.sort()
    return {
        "ci_low": percentile(diffs, 0.025),
        "ci_high": percentile(diffs, 0.975),
        "p_le_zero": sum(d <= 0 for d in diffs) / len(diffs),
    }


def analyze(label, group, codes_bpec, word_orders, subset, rng, langs_out):
    codes = [c for c in codes_bpec if in_subset(c, subset)]
    svo = sorted(codes_bpec[c] for c in codes if word_orders[c] == "SVO")
    sov = sorted(codes_bpec[c] for c in codes if word_orders[c] == "SOV")
    langs_out.append(
        f"[{group} | {label} | {subset}] "
        f"SVO(n={len(svo)}), SOV(n={len(sov)}): "
        f"SOV langs = {sorted(c for c in codes if word_orders[c]=='SOV')}"
    )
    row = {
        "group": group, "label": label, "subset": subset,
        "n_svo": len(svo), "n_sov": len(sov),
        "median_svo": statistics.median(svo) if svo else "",
        "median_sov": statistics.median(sov) if sov else "",
    }
    if len(svo) >= 2 and len(sov) >= 2:
        obs = statistics.median(sov) - statistics.median(svo)
        boot = bootstrap_gap(svo, sov, rng)
        row.update({
            "median_diff_sov_minus_svo": obs,
            "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "p_le_zero": boot["p_le_zero"],
            "pct_more_choices": (2 ** obs - 1) * 100,
            "significant_sov_gt_svo": boot["ci_low"] > 0,
        })
    else:
        row.update({k: "" for k in
                    ["median_diff_sov_minus_svo", "ci_low", "ci_high",
                     "p_le_zero", "pct_more_choices", "significant_sov_gt_svo"]})
    return row


def main():
    word_orders = load_word_orders()
    langs_out = []
    results = []

    # ── Goldfish: languages common to all 4 sizes (mirrors existing bootstrap) ──
    gf = {s: load_goldfish_size(s, word_orders) for s in SIZES}
    common = set.intersection(*[set(gf[s]) for s in SIZES])
    for s in SIZES:
        rng = random.Random(SEED)
        codes_bpec = {c: gf[s][c] for c in common}
        for subset in SUBSETS:
            results.append(analyze(SIZE_LABELS[s], "Goldfish (common)",
                                   codes_bpec, word_orders, subset, rng, langs_out))

    # ── Multilingual models ────────────────────────────────────────────────────
    for name, rdir in MULTILINGUAL:
        codes_bpec = load_multilingual(rdir, word_orders)
        if not codes_bpec:
            continue
        rng = random.Random(SEED)
        for subset in SUBSETS:
            results.append(analyze(name, "Multilingual",
                                   codes_bpec, word_orders, subset, rng, langs_out))

    # ── write outputs ───────────────────────────────────────────────────────────
    fields = ["group", "label", "subset", "n_svo", "n_sov", "median_svo",
              "median_sov", "median_diff_sov_minus_svo", "ci_low", "ci_high",
              "p_le_zero", "pct_more_choices", "significant_sov_gt_svo"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    with OUT_TXT.open("w") as f:
        f.write("Script-controlled SOV-SVO BPEC median gaps (SOV - SVO)\n")
        f.write(f"N_BOOT={N_BOOT}, seed={SEED}. Positive = SOV harder to compress.\n")
        f.write("Subsets: romanized=Latin only; alphabetic=Latn/Grek/Cyrl/Armn/Geor.\n\n")
        cur = None
        for r in results:
            if r["group"] != cur:
                cur = r["group"]
                f.write(f"\n=== {cur} ===\n")
            if r["median_diff_sov_minus_svo"] == "":
                f.write(f"  {r['label']:16} {r['subset']:10} "
                        f"n_svo={r['n_svo']:>2} n_sov={r['n_sov']:>2}  "
                        f"(too few for bootstrap)\n")
                continue
            sig = "  *SIG*" if r["significant_sov_gt_svo"] else ""
            f.write(f"  {r['label']:16} {r['subset']:10} "
                    f"n_svo={r['n_svo']:>2} n_sov={r['n_sov']:>2}  "
                    f"diff={r['median_diff_sov_minus_svo']:+.4f}  "
                    f"95%CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]  "
                    f"P(<=0)={r['p_le_zero']:.4f}  "
                    f"({r['pct_more_choices']:+.1f}% choices){sig}\n")

    OUT_LANGS.write_text("\n".join(langs_out) + "\n")

    print(OUT_TXT.read_text())
    print(f"\nSaved: {OUT_CSV}\nSaved: {OUT_TXT}\nSaved: {OUT_LANGS}")


if __name__ == "__main__":
    main()
