#!/usr/bin/env python3
"""
Morphological-complexity scores (type-token ratio family) for every language
set used in the BPEC word order plots, so lower BPEC can be disentangled from
morphological complexity.

Metrics are computed on the FLORES-200 *devtest* split — the SAME text BPEC was
evaluated on, and a parallel corpus (identical 1012 sentences in every
language), so cross-language differences reflect morphology / tokenization, not
content.

Per language we report:
  n_tokens, n_types         raw counts (whitespace tokens, case-folded)
  ttr                       types / tokens                        (length-sensitive)
  root_ttr                  types / sqrt(tokens)   (Guiraud R)
  mattr_100                 moving-average TTR, window 100         (length-robust; preferred)
  mean_word_len             mean characters per token              (morphology proxy)
  mean_sent_len_tok         mean tokens per sentence
  whitespace_delimited      False for scriptio-continua scripts where word TTR
                            is unreliable (Jpan/Hani/Hans/Hant/Thai/Mymr/…)

Language sets (reconstructed exactly as in the plot scripts):
  goldfish_62  common across all 4 Goldfish sizes ∩ resolved rightness score
               (figure panels c/d)
  goldfish_68  common across all 4 Goldfish sizes ∩ base_word_order label
               (the "common 68" set)
  mgpt/bloom/xglm  each model's FLORES-mapped training-language set
               (multilingual comparison)

Outputs (to <ARCHIVE>/flores_clustering/morph_complexity/):
  morph_all_flores.csv            master table, one row per FLORES language
  morph_<set>.csv                 per-set table joined to that set's BPEC + word order
  morph_bpec_correlations.csv     Spearman corr(metric, BPEC) overall & within SVO/SOV

OPTIONAL tokenizer pass (additive; does NOT touch the MATTR outputs above):
  python compute_morphological_complexity.py --tokenizer
    → morph_tokenizer_goldfish.csv   subword metrics from each language's
      monolingual Goldfish-1000mb tokenizer (fertility, tok/char, subword MATTR).
      Script-agnostic, so it recovers jpn/zho/tha that whitespace MATTR drops.
      Downloads tokenizers into the HF cache dir (HF_HOME), not home.
"""

import glob
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

# ── paths ───────────────────────────────────────────────────────────────────────
DATA_ROOT     = Path(os.environ.get("WORD_ORDER_DATA", "data"))
HF_CACHE      = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
ARCHIVE       = DATA_ROOT
ENCODING_CSV  = ARCHIVE / "flores_clustering" / "flores200_encoding.csv"
GOLDFISH_BASE = ARCHIVE / "results_natural_langs_flores"
FLORES_CACHE  = HF_CACHE / "datasets" / "facebook___flores"
OUT_DIR       = ARCHIVE / "flores_clustering" / "morph_complexity"

SIZES = ["5mb", "10mb", "100mb", "1000mb"]
POS_COLS = ["pos1_vp_comp", "pos2_comp", "pos3_pp", "pos4_np", "pos5_rel"]
ORDER_DISPLAY = ["SVO", "SOV", "VSO", "VOS", "OVS", "OSV", "NoDominant"]

# scripts written without whitespace word boundaries → word-level TTR unreliable
NON_WS_SCRIPTS = {"Jpan", "Hani", "Hans", "Hant", "Thai", "Mymr",
                  "Khmr", "Laoo", "Tibt", "Yiii"}

MATTR_WINDOW = 100

# ── multilingual model training-language sets (from the multilingual plot) ──────
BLOOM_TRAINED = {
    "arb_Arab","asm_Beng","bam_Latn","ben_Beng","cat_Latn","eng_Latn","eus_Latn","fon_Latn","fra_Latn","guj_Gujr","hin_Deva","ibo_Latn","ind_Latn","kan_Knda","kik_Latn","kin_Latn","lin_Latn","lug_Latn","mal_Mlym","mar_Deva","npi_Deva","nso_Latn","nya_Latn","ory_Orya","pan_Guru","por_Latn","run_Latn","sna_Latn","sot_Latn","spa_Latn","swh_Latn","tam_Taml","tel_Telu","tsn_Latn","tso_Latn","tum_Latn","twi_Akua","urd_Arab","vie_Latn","wol_Latn","xho_Latn","yor_Latn","zho_Hans","zho_Hant","zul_Latn",
}
XGLM_TRAINED = {
    "deu_Latn","eng_Latn","fra_Latn","gla_Latn","guj_Gujr","hin_Deva","ind_Latn","ita_Latn","jpn_Jpan","kor_Hang","mal_Mlym","mya_Mymr","nld_Latn","npi_Deva","pan_Guru","pol_Latn","por_Latn","ron_Latn","rus_Cyrl","spa_Latn","swe_Latn","tha_Thai","tur_Latn","uig_Arab","ukr_Cyrl","urd_Arab","vie_Latn","zho_Hans","zho_Hant",
}
MGPT_TRAINED = {
    "afr_Latn","apc_Arab","arb_Arab","arz_Arab","azb_Arab","azj_Latn","bak_Cyrl","bel_Cyrl","ben_Beng","bul_Cyrl","dan_Latn","deu_Latn","ell_Grek","eng_Latn","est_Latn","eus_Latn","fin_Latn","fra_Latn","heb_Hebr","hin_Deva","hun_Latn","hye_Armn","ind_Latn","ita_Latn","jav_Latn","jpn_Jpan","kat_Geor","kaz_Cyrl","khk_Cyrl","kir_Cyrl","kor_Hang","lit_Latn","lvs_Latn","mal_Mlym","mar_Deva","min_Latn","mya_Mymr","pes_Arab","pol_Latn","por_Latn","ron_Latn","rus_Cyrl","spa_Latn","swe_Latn","swh_Latn","tam_Taml","tat_Cyrl","tel_Telu","tgk_Cyrl","tgl_Latn","tha_Thai","tuk_Latn","tur_Latn","ukr_Cyrl","urd_Arab","uzn_Latn","vie_Latn","yor_Latn","zsm_Latn",
}
MODEL_RESULT_DIRS = {
    "mgpt":  ARCHIVE / "results_mgpt_flores",
    "bloom": ARCHIVE / "results_bloom_flores",
    "xglm":  ARCHIVE / "results_xglm_564m_flores",
}
MODEL_TRAINED = {"mgpt": MGPT_TRAINED, "bloom": BLOOM_TRAINED, "xglm": XGLM_TRAINED}


# ── word order scoring ───────────────────────────
def _score_from_row(row):
    bits = [row[c] for c in POS_COLS]
    if any(b == "?" for b in bits):
        return None
    n_r = sum(1 for b in bits if b == "R")
    n_l = sum(1 for b in bits if b == "L")
    if n_r >= 3 and n_r > n_l:
        majority = "R"
    elif n_l >= 3 and n_l > n_r:
        majority = "L"
    else:
        return None
    resolved = [majority if b == "N" else b for b in bits]
    return sum(1 for b in resolved if b == "R")


# ── FLORES text loading ─────────────────────────────────────────────────────────
def load_devtest_sentences(flores_code: str):
    hits = glob.glob(str(FLORES_CACHE / flores_code / "*" / "*" / "flores-devtest.arrow"))
    if not hits:
        return None
    with pa.memory_map(hits[0]) as src:
        try:
            tbl = ipc.RecordBatchFileReader(src).read_all()
        except pa.ArrowInvalid:
            src.seek(0)
            tbl = ipc.RecordBatchStreamReader(src).read_all()
    return tbl.column("sentence").to_pylist()


_STRIP = "".join(chr(c) for c in range(0x20))  # placeholder; real strip below
def _tokenize(sentence: str):
    """Whitespace split, case-fold, strip leading/trailing punctuation & symbols."""
    toks = []
    for raw in sentence.split():
        t = raw.strip()
        # strip surrounding punctuation / symbols (keep word-internal apostrophes etc.)
        while t and unicodedata.category(t[0])[0] in ("P", "S"):
            t = t[1:]
        while t and unicodedata.category(t[-1])[0] in ("P", "S"):
            t = t[:-1]
        if t:
            toks.append(t.casefold())
    return toks


def _mattr(tokens, window):
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens) if tokens else float("nan")
    ratios = []
    counts = Counter(tokens[:window])
    distinct = len(counts)
    ratios.append(distinct / window)
    for i in range(window, len(tokens)):
        out_tok, in_tok = tokens[i - window], tokens[i]
        if out_tok != in_tok:
            counts[out_tok] -= 1
            if counts[out_tok] == 0:
                del counts[out_tok]
                distinct -= 1
            if in_tok not in counts:
                distinct += 1
            counts[in_tok] += 1
        ratios.append(distinct / window)
    return float(np.mean(ratios))


def _mtld_pass(tokens, threshold):
    """One directional MTLD pass: mean token length of a run before TTR decays to threshold."""
    factors, start = 0.0, 0
    types = set()
    for i, tok in enumerate(tokens):
        types.add(tok)
        ttr = len(types) / (i - start + 1)
        if ttr <= threshold:
            factors += 1
            start = i + 1
            types = set()
    # partial factor for the trailing incomplete run
    if start < len(tokens):
        ttr = len(types) / (len(tokens) - start)
        factors += (1 - ttr) / (1 - threshold)
    return len(tokens) / factors if factors else float("nan")


def _mtld(tokens, threshold=0.72):
    """Measure of Textual Lexical Diversity — bidirectional mean. Length-independent."""
    if len(tokens) < 10:
        return float("nan")
    fwd = _mtld_pass(tokens, threshold)
    bwd = _mtld_pass(tokens[::-1], threshold)
    return float(np.mean([fwd, bwd]))


def _hdd(tokens, sample=42):
    """HD-D (vocd-D): expected TTR for a random sample of `sample` tokens, via the
    hypergeometric distribution. Standard length-robust diversity index."""
    n = len(tokens)
    if n <= sample:
        return float("nan")
    freqs = Counter(tokens)
    contrib = 0.0
    for f in freqs.values():
        if n - f >= sample:
            # P(type absent) = C(n-f, sample) / C(n, sample) = prod (n-f-i)/(n-i)
            p_absent = 1.0
            for i in range(sample):
                p_absent *= (n - f - i) / (n - i)
        else:
            p_absent = 0.0
        contrib += (1 - p_absent)
    return contrib / sample


def compute_metrics(flores_code: str):
    sents = load_devtest_sentences(flores_code)
    if sents is None:
        return None
    all_tokens, sent_lens = [], []
    for s in sents:
        toks = _tokenize(s)
        all_tokens.extend(toks)
        sent_lens.append(len(toks))
    n_tok = len(all_tokens)
    if n_tok == 0:
        return None
    n_typ = len(set(all_tokens))
    script = flores_code.split("_")[-1]
    return {
        "flores_code": flores_code,
        "script": script,
        "whitespace_delimited": script not in NON_WS_SCRIPTS,
        "n_tokens": n_tok,
        "n_types": n_typ,
        "ttr": n_typ / n_tok,
        "root_ttr": n_typ / math.sqrt(n_tok),
        "mattr_100": _mattr(all_tokens, MATTR_WINDOW),
        "mattr_1000": _mattr(all_tokens, 1000),
        "mtld": _mtld(all_tokens),
        "hdd_42": _hdd(all_tokens),
        "mean_word_len": float(np.mean([len(t) for t in all_tokens])),
        "mean_sent_len_tok": float(np.mean(sent_lens)),
    }


# ── BPEC loaders ────────────────────────────────────────────────────────────────
def load_goldfish_bpec(size: str):
    rows = {}
    for p in (GOLDFISH_BASE / size).glob("*_devtest.json"):
        code = p.stem.replace("_devtest", "")
        d = json.loads(p.read_text())
        if "bpec" in d and d["bpec"] is not None:
            rows[code] = float(d["bpec"])
    return rows


def load_model_bpec(results_dir: Path):
    rows = {}
    for p in results_dir.glob("bpec_*_devtest.json"):
        code = p.stem.replace("bpec_", "").replace("_devtest", "")
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("bpec") is not None:
            rows[code] = float(d["bpec"])
    return rows


# ── language-set reconstruction ─────────────────────────────────────────────────
def common_across_sizes():
    sets = []
    for size in SIZES:
        codes = {p.stem.replace("_devtest", "")
                 for p in (GOLDFISH_BASE / size).glob("*_devtest.json")}
        sets.append(codes)
    return set.intersection(*sets)


def build_sets(enc: pd.DataFrame):
    enc = enc.set_index("flores_code")
    common = common_across_sizes()

    # goldfish_68: common ∩ base_word_order in ORDER_DISPLAY
    wo_ok = set(enc[enc["base_word_order"].isin(ORDER_DISPLAY)].index)
    goldfish_68 = common & wo_ok

    # goldfish_62: common ∩ resolved rightness score
    score = enc.apply(_score_from_row, axis=1)
    score_ok = set(score.dropna().index)
    goldfish_62 = common & score_ok

    return {
        "goldfish_62": goldfish_62,
        "goldfish_68": goldfish_68,
        "mgpt": MGPT_TRAINED,
        "bloom": BLOOM_TRAINED,
        "xglm": XGLM_TRAINED,
    }


# ── main ─────────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = pd.read_csv(ENCODING_CSV)
    enc["score"] = enc.apply(_score_from_row, axis=1)

    sets = build_sets(enc)
    print("Language-set sizes:")
    for name, codes in sets.items():
        print(f"  {name}: {len(codes)}")

    # union of every code we need + all cached flores langs → master table
    all_codes = sorted(set().union(*sets.values()) |
                       {p.name for p in FLORES_CACHE.iterdir() if p.is_dir()})
    recs, missing = [], []
    for code in all_codes:
        m = compute_metrics(code)
        if m is None:
            missing.append(code)
        else:
            recs.append(m)
    if missing:
        print(f"  no FLORES devtest text for: {sorted(missing)}")
    morph = pd.DataFrame(recs)
    enc_cols = ["flores_code", "language_name", "base_word_order", "score"]
    morph = morph.merge(enc[enc_cols], on="flores_code", how="left")
    morph.to_csv(OUT_DIR / "morph_all_flores.csv", index=False)
    print(f"Saved: {OUT_DIR/'morph_all_flores.csv'} ({len(morph)} langs)")

    # BPEC sources per set
    gf_bpec = {s: load_goldfish_bpec(s) for s in SIZES}
    model_bpec = {m: load_model_bpec(d) for m, d in MODEL_RESULT_DIRS.items()}

    metric_cols = ["ttr", "root_ttr", "mattr_100", "mattr_1000", "mtld", "hdd_42", "mean_word_len"]
    corr_rows = []

    for name, codes in sets.items():
        sub = morph[morph["flores_code"].isin(codes)].copy()
        if name.startswith("goldfish"):
            for s in SIZES:
                sub[f"bpec_{s}"] = sub["flores_code"].map(gf_bpec[s])
            bpec_for_corr = "bpec_1000mb"   # 1000mb is where the SVO/SOV effect appears
        else:
            sub["bpec"] = sub["flores_code"].map(model_bpec[name])
            bpec_for_corr = "bpec"
        sub = sub.sort_values("mattr_100", ascending=False)
        sub.to_csv(OUT_DIR / f"morph_{name}.csv", index=False)
        print(f"Saved: morph_{name}.csv ({len(sub)} langs)")

        # correlations metric vs BPEC — overall and within SVO / SOV, whitespace langs only
        valid = sub[sub["whitespace_delimited"] & sub[bpec_for_corr].notna()]
        for scope, dfx in [("all", valid),
                           ("SVO", valid[valid["base_word_order"] == "SVO"]),
                           ("SOV", valid[valid["base_word_order"] == "SOV"])]:
            if len(dfx) < 4:
                continue
            for mc in metric_cols:
                rho = dfx[mc].rank().corr(dfx[bpec_for_corr].rank())  # Spearman = Pearson of ranks
                corr_rows.append({"set": name, "bpec_col": bpec_for_corr,
                                  "scope": scope, "n": len(dfx), "metric": mc,
                                  "spearman_rho": round(rho, 3)})

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT_DIR / "morph_bpec_correlations.csv", index=False)
    print(f"\nSaved: morph_bpec_correlations.csv")
    print("\n=== Spearman corr(morph metric, BPEC), scope=all, whitespace langs ===")
    show = corr[corr["scope"] == "all"].pivot(index="set", columns="metric",
                                              values="spearman_rho")
    print(show.to_string())


# ── OPTIONAL: tokenizer-space complexity (additive; separate CSV, MATTR untouched)─
# Uses each language's monolingual Goldfish-1000mb tokenizer (BPE, vocab 50k) on
# the same FLORES devtest. Script-agnostic → recovers jpn/zho/tha. Entangled with
# BPEC (same tokenizer), so it is a mechanism/robustness check, not the primary
# independent morphology measure (that stays whitespace mattr_100/hdd_42).
TOK_OUT_CSV = OUT_DIR / "morph_tokenizer_goldfish.csv"
SHARE_CACHE = str(HF_CACHE)


def _accessor_variety(ids, window=1000, step=250):
    """Successor Accessor Variety (AV) and its Shannon efficiency (eta), à la
    Poelman et al. (2025) / Tatariya et al. (2025). Right accessor (successor).
    For each token, AV = number of DISTINCT types that follow it; eta = normalized
    entropy of that successor distribution. Averaged token-weighted within sliding
    1000-token windows (like MATTR, to remove length dependence), then over windows.
    (Non-boundary-reset flat stream; step<window overlapping windows.)"""
    n = len(ids)
    starts = ([0] if n <= window else list(range(0, n - window + 1, step)))
    av_w, eta_w = [], []
    for s in starts:
        seg = ids[s:s + window] if n > window else ids
        succ = {}
        for i in range(len(seg) - 1):
            succ.setdefault(seg[i], Counter())[seg[i + 1]] += 1
        av_sum = eta_sum = cnt = 0
        for i in range(len(seg) - 1):
            c = succ[seg[i]]
            k = len(c)
            av_sum += k
            if k > 1:
                tot = sum(c.values())
                H = -sum((v / tot) * math.log(v / tot) for v in c.values())
                eta_sum += H / math.log(k)
            cnt += 1
        if cnt:
            av_w.append(av_sum / cnt)
            eta_w.append(eta_sum / cnt)
    return (float(np.mean(av_w)) if av_w else float("nan"),
            float(np.mean(eta_w)) if eta_w else float("nan"))


def _tokenizer_metrics(code, tok):
    sents = load_devtest_sentences(code)
    if sents is None:
        return None
    n_sub = n_char = n_word = 0
    ids_all = []
    for s in sents:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        ids_all.extend(ids)
        n_sub += len(ids)
        n_char += sum(1 for ch in s if not ch.isspace())
        n_word += len(_tokenize(s))
    if n_sub == 0 or n_char == 0:
        return None
    av, eta = _accessor_variety(ids_all, window=1000, step=250)
    return {
        "flores_code": code,
        "n_subword": n_sub,
        "fertility": (n_sub / n_word) if n_word else float("nan"),  # tokens/word
        "tok_per_char": n_sub / n_char,                              # script-agnostic
        "subword_ttr": len(set(ids_all)) / n_sub,
        "subword_mattr_100": _mattr(ids_all, MATTR_WINDOW),
        "subword_mattr_1000": _mattr(ids_all, 1000),
        "subword_av": av,                                           # accessor variety
        "subword_eta": eta,                                        # AV Shannon efficiency
    }


def tokenizer_pass():
    os.environ.setdefault("HF_HOME", SHARE_CACHE)
    os.environ.setdefault("HF_HUB_CACHE", f"{SHARE_CACHE}/hub")
    os.makedirs(f"{SHARE_CACHE}/hub", exist_ok=True)
    from transformers import AutoTokenizer
    # FLORES code → Goldfish repo prefix (e.g. swh_Latn → swa_Latn); identity otherwise
    from goldfish_flores_lang import goldfish_checkpoint_prefix

    base = pd.read_csv(OUT_DIR / "morph_goldfish_68.csv")   # superset of the 62 set
    codes = sorted(base["flores_code"].tolist())

    recs, failed = [], []
    for i, code in enumerate(codes, 1):
        repo = f"goldfish-models/{goldfish_checkpoint_prefix(code)}_1000mb"
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception as e:
            failed.append((code, type(e).__name__)); continue
        m = _tokenizer_metrics(code, tok)
        if m is None:
            failed.append((code, "no_text")); continue
        recs.append(m)

    keep = ["flores_code", "language_name", "base_word_order", "score",
            "whitespace_delimited", "mattr_100", "bpec_5mb", "bpec_1000mb"]
    out = pd.DataFrame(recs).merge(base[keep], on="flores_code", how="left")
    out.to_csv(TOK_OUT_CSV, index=False)          # separate file — MATTR outputs untouched
    print(f"\nSaved: {TOK_OUT_CSV} ({len(out)} langs; {len(failed)} failed: {failed})")

    print("\n=== SVO vs SOV medians (tokenizer space, ALL scripts incl. jpn/zho/tha) ===")
    for mc in ["tok_per_char", "subword_mattr_100", "subword_mattr_1000",
               "fertility", "subword_av", "subword_eta"]:
        svo = out[out.base_word_order == "SVO"][mc].median()
        sov = out[out.base_word_order == "SOV"][mc].median()
        print(f"  {mc:20s} SVO={svo:.3f}  SOV={sov:.3f}  Δ={sov-svo:+.3f}")
    print("\ncorr(metric, bpec_1000mb) within SVO / SOV (all scripts):")
    for mc in ["subword_mattr_1000", "subword_av", "subword_eta"]:
        for g in ["SVO", "SOV"]:
            gd = out[out.base_word_order == g]
            rho = gd[mc].rank().corr(gd["bpec_1000mb"].rank())
            print(f"  {mc:20s} {g}: n={gd[mc].notna().sum():2d}  rho={rho:+.3f}")


if __name__ == "__main__":
    if "--tokenizer" in sys.argv:
        tokenizer_pass()          # additive pass; leaves MATTR CSVs untouched
    else:
        main()                    # default: whitespace MATTR (unchanged)
