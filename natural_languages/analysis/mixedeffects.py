#!/usr/bin/env python3
"""
Mixed-effects models of Goldfish BPEC: does the SOV x training-size interaction
survive controlling for morphological complexity, language resourcedness, and
training-data composition? Reproduces the paper's mixed-effects table.

Data: 62-language set, SVO vs SOV, repeated over the 4 Goldfish training sizes.
Covariates (z-scored):
  mattr_z       morphological complexity (subword MATTR; static -> level)
  resource_lvl  language resourcedness (Joshi et al. 2020,
                aclanthology.org/2020.acl-main.560; also x size)
  data_comp     training data composition (OSCAR share; also x size)
  lz            log10(size_mb), z-scored (the within-language "size" axis)

The ladder (random intercept per language, ML fits):
  M1  is_SOV * lz                                    (no covariates)
  M2  + mattr_z                                      (morphology, level)
  M3  + resource_lvl                                 (resourcedness, level)
  M4  + data_comp                                    (data composition, level)
  M5  + mattr_z + resource_lvl + data_comp           (all three levels)
  M6  + mattr_z * is_SOV                             (morphology x word order)
  M7  + resource_lvl * lz + mattr_z                  (resourcedness x size)
  M8  + data_comp * lz + mattr_z                     (data composition x size)
  M9  + resource_lvl * lz + data_comp * lz + mattr_z (full)
Robustness: maximal random-effects structure (random size slope per language)
for M1/M7/M9, and resourcedness as a binary high/low factor.

Output: <DATA_ROOT>/flores_clustering/morph_complexity/mixedeffects.txt
Run (needs statsmodels/pandas): python mixedeffects.py
"""

import os
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))
MORPH = DATA_ROOT / "flores_clustering/morph_complexity"
FLC   = DATA_ROOT / "flores_clustering"
INFO  = MORPH / "goldfish_data_info.tsv"          # from github.com/tylerachang/goldfish
SIZE_COLS = {5: "bpec_5mb", 10: "bpec_10mb", 100: "bpec_100mb", 1000: "bpec_1000mb"}
WEB = {"oscar"}                                    # training-data composition = OSCAR share


def webshare(prop):
    """OSCAR share from a 'corpus:frac,corpus:frac' proportions string."""
    if not isinstance(prop, str):
        return np.nan
    d = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in prop.split(",")}
    return sum(v for s, v in d.items() if s in WEB)


def build_long():
    df  = pd.read_csv(MORPH / "morph_goldfish_62.csv")
    tok = pd.read_csv(MORPH / "morph_tokenizer_goldfish.csv")[["flores_code", "subword_mattr_1000"]]
    res = pd.read_csv(FLC / "languages_resource_table.csv")[["flores_code", "joshi_class"]]
    res.loc[res.flores_code == "swh_Latn", "joshi_class"] = 2.0   # Swahili: class 2 (missing in table)
    info = pd.read_csv(INFO, sep="\t")
    prop = dict(zip(info.lang, info.proportions))

    df = df.merge(tok, on="flores_code").merge(res, on="flores_code")
    df = df[df.base_word_order.isin(["SVO", "SOV"])].copy()
    gl = df.flores_code.str.lower().replace({"swh_latn": "swa_latn"})   # Goldfish uses swa_
    df["web"] = gl.map(lambda c: webshare(prop.get(c)))
    df = df.dropna(subset=["subword_mattr_1000", "joshi_class", "web"])

    long = df.melt(id_vars=["flores_code", "base_word_order", "subword_mattr_1000",
                            "joshi_class", "web"],
                   value_vars=list(SIZE_COLS.values()),
                   var_name="sc", value_name="bpec").dropna(subset=["bpec"])
    long["size_mb"] = long.sc.map({v: k for k, v in SIZE_COLS.items()})
    z = lambda s: (s - s.mean()) / s.std()
    long["is_SOV"]       = (long.base_word_order == "SOV").astype(float)
    long["lz"]           = z(np.log10(long.size_mb))
    long["mattr_z"]      = z(long.subword_mattr_1000)
    long["resource_lvl"] = z(long.joshi_class)
    long["data_comp"]    = z(long.web)
    long["high_res"]     = (long.joshi_class >= 3).astype(float)   # high(>=3) vs low
    return long


def fit(long, formula, maximal=False):
    """Random-intercept LMM (or maximal RE if maximal=True); ML fit; first converged."""
    kw = dict(re_formula="~lz") if maximal else {}
    for method in ("lbfgs", "bfgs", "cg", "powell"):
        try:
            r = smf.mixedlm(formula, long, groups=long.flores_code,
                            **kw).fit(method=method, reml=False, maxiter=3000)
            if r.converged:
                return r, method
        except Exception:
            continue
    return None, None


# terms to display (statsmodels names interactions in either order, so list both)
SHOW = ["is_SOV", "is_SOV:lz", "lz:is_SOV",
        "mattr_z", "mattr_z:is_SOV", "is_SOV:mattr_z",
        "resource_lvl", "resource_lvl:lz", "lz:resource_lvl",
        "data_comp", "data_comp:lz", "lz:data_comp",
        "high_res", "high_res:lz", "lz:high_res"]


def report(long, tag, formula, out, maximal=False):
    r, method = fit(long, formula, maximal=maximal)
    if r is None:
        out.append(f"\n{tag}: FIT FAILED"); return
    out.append(f"\n{tag}   [{'maximal RE' if maximal else 'rand-int'}; {method}; N={int(r.nobs)}]")
    for t in SHOW:
        if t in r.params.index:
            star = "*" if r.pvalues[t] < .05 else ""
            out.append(f"    {t:16s} beta={r.params[t]:+.4f}  se={r.bse[t]:.4f}  p={r.pvalues[t]:.3g} {star}")


LADDER = [
    ("M1  base",                              "bpec ~ is_SOV*lz"),
    ("M2  + morphological complexity",        "bpec ~ is_SOV*lz + mattr_z"),
    ("M3  + resourcedness",                   "bpec ~ is_SOV*lz + resource_lvl"),
    ("M4  + training-data composition",       "bpec ~ is_SOV*lz + data_comp"),
    ("M5  + all three levels",                "bpec ~ is_SOV*lz + mattr_z + resource_lvl + data_comp"),
    ("M6  + morphology x word order",         "bpec ~ is_SOV*lz + mattr_z*is_SOV"),
    ("M7  + resourcedness x size",            "bpec ~ is_SOV*lz + resource_lvl*lz + mattr_z"),
    ("M8  + data composition x size",         "bpec ~ is_SOV*lz + data_comp*lz + mattr_z"),
    ("M9  full",                              "bpec ~ is_SOV*lz + resource_lvl*lz + data_comp*lz + mattr_z"),
]


def main():
    long = build_long()
    n_sov = int(long[long.is_SOV == 1].flores_code.nunique())
    n_svo = int(long[long.is_SOV == 0].flores_code.nunique())
    out = [f"Mixed-effects ladder — Goldfish BPEC, 62-set (SVO vs SOV)",
           f"N={len(long)} obs / {long.flores_code.nunique()} langs (SOV {n_sov}, SVO {n_svo}); "
           f"data_comp = OSCAR share; ML fits.",
           "=" * 74]

    for tag, formula in LADDER:
        report(long, tag, formula, out)

    out.append("\n" + "=" * 74)
    out.append("ROBUSTNESS: maximal random-effects (random size slope per language)")
    out.append("=" * 74)
    report(long, "M1 maximal RE", "bpec ~ is_SOV*lz", out, maximal=True)
    report(long, "M7 maximal RE", "bpec ~ is_SOV*lz + resource_lvl*lz + mattr_z", out, maximal=True)
    report(long, "M9 maximal RE", "bpec ~ is_SOV*lz + resource_lvl*lz + data_comp*lz + mattr_z", out, maximal=True)

    out.append("\nROBUSTNESS: resourcedness as binary high(>=3)/low factor")
    report(long, "binary res.", "bpec ~ is_SOV*lz + high_res*lz + mattr_z", out)

    txt = "\n".join(out) + "\n"
    MORPH.mkdir(parents=True, exist_ok=True)
    (MORPH / "mixedeffects.txt").write_text(txt)
    print(txt)
    print(f"Saved: {MORPH / 'mixedeffects.txt'}")


if __name__ == "__main__":
    main()
