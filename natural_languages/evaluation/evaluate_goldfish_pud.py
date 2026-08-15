#!/usr/bin/env python3
"""
Evaluate goldfish models on PUD (Parallel Universal Dependencies) treebanks.
Computes BPEC (bits per English character) the same way as FLORES:
- Model is run on the language's PUD sentences.
- Normalisation is by English PUD character count so BPEC is comparable across languages.

Usage:
  # Single config + model
  python evaluate_goldfish_pud.py --model goldfish-models/eng_latn_1000mb --pud-config en_pud

  # All available PUD configs × model sizes (skips missing models)
  python evaluate_goldfish_pud.py --run-all --results-dir data/results_natural_langs_pud
"""

import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

_share_cache = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", _share_cache)
os.environ.setdefault("HF_HUB_CACHE", f"{_share_cache}/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", _share_cache)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_share_cache}/hub")
os.makedirs(_share_cache, exist_ok=True)
os.makedirs(f"{_share_cache}/hub", exist_ok=True)

import torch
from torch import nn

from goldfish_flores_lang import goldfish_checkpoint_prefix
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

DATASET_NAME = "universal-dependencies/universal_dependencies"
RESULTS_BASE_DEFAULT = str(DATA_ROOT / "results_natural_langs_pud")
MODEL_SIZES = ("5mb", "10mb", "100mb", "1000mb")

# PUD config (e.g. en_pud) -> FLORES/goldfish language code (e.g. eng_Latn)
# Used to select goldfish model: goldfish-models/{flores_lang}_{size}
# Include all PUD treebanks that have a goldfish model (FLORES 200).
PUD_CONFIG_TO_FLORES: Dict[str, str] = {
    "ar_pud": "arb_Arab",
    "bn_pud": "ben_Beng",    # Bengali
    "cs_pud": "ces_Latn",
    "de_pud": "deu_Latn",
    "el_pud": "ell_Grek",
    "en_pud": "eng_Latn",
    "es_pud": "spa_Latn",
    "fi_pud": "fin_Latn",
    "fr_pud": "fra_Latn",
    "gl_pud": "glg_Latn",   # Galician
    "hi_pud": "hin_Deva",
    "hr_pud": "hrv_Latn",
    "id_pud": "ind_Latn",
    "is_pud": "isl_Latn",
    "it_pud": "ita_Latn",
    "ja_pud": "jpn_Jpan",
    "ko_pud": "kor_Hang",
    "mag_pud": "mag_Deva",  # Magahi
    "nl_pud": "nld_Latn",
    "pl_pud": "pol_Latn",
    "pt_pud": "por_Latn",
    "ru_pud": "rus_Cyrl",
    "sv_pud": "swe_Latn",
    "th_pud": "tha_Thai",
    "tr_pud": "tur_Latn",
    "zh_pud": "zho_Hans",
}


def get_pud_configs() -> List[str]:
    """Return list of PUD config names from the dataset."""
    try:
        configs = get_dataset_config_names(DATASET_NAME)
        return sorted(c for c in configs if c.endswith("_pud"))
    except Exception:
        return list(PUD_CONFIG_TO_FLORES.keys())


def load_pud_sentences(
    hf_config: str,
    split: Optional[str] = None,
    dataset_name: str = DATASET_NAME,
) -> List[str]:
    """Load PUD treebank and return list of sentence strings (for BPEC)."""
    full_ds = load_dataset(dataset_name, hf_config, trust_remote_code=True)
    for split_name in (split or "test", "train", "validation", "test"):
        if split_name in full_ds:
            ds = full_ds[split_name]
            break
    else:
        ds = full_ds[list(full_ds.keys())[0]]

    sentences = []
    for row in ds:
        if "text" in ds.column_names and row.get("text"):
            sentences.append(row["text"].strip())
        elif "tokens" in ds.column_names and row.get("tokens"):
            sentences.append(" ".join(str(t) for t in row["tokens"]))
        else:
            continue
    return sentences


def compute_bpec(
    model,
    tokenizer,
    sentences: List[str],
    reference_char_count: int,
    device: str = "cuda",
    max_length: int = 512,
    verbose: bool = False,
) -> float:
    """
    Compute length-normalised bits per character (BPEC), same as FLORES.
    BPEC = -sum(log p(token_t | context)) / (reference_char_count * log(2))
    """
    model.eval()
    start_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else None
    total_log_prob = 0.0

    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Computing BPEC", disable=not verbose):
            encoded = tokenizer(sentence, add_special_tokens=False)
            input_ids = list(encoded["input_ids"])
            if start_token_id is not None:
                input_ids.insert(0, start_token_id)
            if len(input_ids) < 2:
                continue
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            input_ids_t = torch.tensor([input_ids], device=device)
            outputs = model(input_ids=input_ids_t, return_dict=True)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            for t in range(1, len(input_ids)):
                token_id = input_ids[t]
                log_prob = log_probs[0, t - 1, token_id].item()
                total_log_prob += log_prob

    if reference_char_count > 0:
        return -total_log_prob / (reference_char_count * math.log(2))
    return 0.0


def run_one(
    model_name: str,
    pud_config: str,
    results_dir: Path,
    split: str = "test",
    max_length: int = 512,
    device: str = "cuda",
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run BPEC evaluation for one model and one PUD config.
    Returns result dict or None on failure.
    """
    flores_lang = PUD_CONFIG_TO_FLORES.get(pud_config)
    if not flores_lang:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)
    except Exception as e:
        if verbose:
            print(f"  Model load failed: {e}")
        return None

    try:
        sentences = load_pud_sentences(pud_config, split=split)
    except Exception as e:
        if verbose:
            print(f"  PUD load failed: {e}")
        return None

    if not sentences:
        if verbose:
            print(f"  No sentences for {pud_config}")
        return None

    try:
        en_sentences = load_pud_sentences("en_pud", split=split)
    except Exception:
        en_sentences = []
    if not en_sentences:
        if verbose:
            print("  English PUD (en_pud) required for reference character count.")
        return None
    reference_char_count = sum(len(s) for s in en_sentences)

    bpec = compute_bpec(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        reference_char_count=reference_char_count,
        device=device,
        max_length=max_length,
        verbose=verbose,
    )

    return {
        "model": model_name,
        "dataset": DATASET_NAME,
        "pud_config": pud_config,
        "language": flores_lang,
        "split": split,
        "num_sentences": len(sentences),
        "reference_char_count": reference_char_count,
        "bpec": bpec,
        "max_length": max_length,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate goldfish models on PUD treebanks (BPEC, same as FLORES)."
    )
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model (e.g. goldfish-models/eng_latn_1000mb)")
    parser.add_argument("--pud-config", type=str, default=None, help="PUD config (e.g. en_pud, ru_pud)")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run over all PUD configs × model sizes (ignores --model/--pud-config)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_BASE_DEFAULT,
        help=f"Results base directory (default: {RESULTS_BASE_DEFAULT})",
    )
    parser.add_argument("--split", type=str, default="test", help="PUD split (default: test)")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated PUD configs for --run-all (default: all from HF)",
    )
    args = parser.parse_args()

    results_base = Path(args.results_dir)
    results_base.mkdir(parents=True, exist_ok=True)

    if args.run_all:
        configs = [c.strip() for c in args.configs.split(",")] if args.configs else get_pud_configs()
        print(f"PUD BPEC evaluation: {len(configs)} configs × {len(MODEL_SIZES)} model sizes")
        print(f"Results dir: {results_base}")
        evaluated = 0
        skipped_no_mapping = 0
        skipped_no_model = 0
        failed = 0
        for model_size in MODEL_SIZES:
            size_dir = results_base / model_size
            size_dir.mkdir(parents=True, exist_ok=True)
            for pud_config in configs:
                flores_lang = PUD_CONFIG_TO_FLORES.get(pud_config)
                if not flores_lang:
                    skipped_no_mapping += 1
                    continue
                model_name = (
                    f"goldfish-models/{goldfish_checkpoint_prefix(flores_lang)}_{model_size}"
                )
                out_name = f"{pud_config}_{args.split}.json"
                out_path = size_dir / out_name
                if out_path.exists():
                    evaluated += 1
                    continue
                try:
                    res = run_one(
                        model_name=model_name,
                        pud_config=pud_config,
                        results_dir=size_dir,
                        split=args.split,
                        max_length=args.max_length,
                        device=args.device,
                        verbose=args.verbose,
                    )
                except Exception as e:
                    if args.verbose:
                        print(f"  Error {pud_config} {model_size}: {e}")
                    failed += 1
                    continue
                if res is None:
                    skipped_no_model += 1
                    continue
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=2)
                evaluated += 1
                print(f"  {pud_config} {model_size} -> BPEC {res['bpec']:.4f} -> {out_path}")
        print(f"Done. Evaluated: {evaluated}, skipped (no mapping): {skipped_no_mapping}, skipped (no model): {skipped_no_model}, failed: {failed}")
        return

    if not args.model or not args.pud_config:
        parser.error("Use --model and --pud-config for single run, or --run-all")
    flores_lang = PUD_CONFIG_TO_FLORES.get(args.pud_config)
    if not flores_lang:
        print(f"Unknown PUD config (no FLORES mapping): {args.pud_config}")
        print("Known:", list(PUD_CONFIG_TO_FLORES.keys()))
        return 1
    res = run_one(
        model_name=args.model,
        pud_config=args.pud_config,
        results_dir=results_base,
        split=args.split,
        max_length=args.max_length,
        device=args.device,
        verbose=args.verbose,
    )
    if res is None:
        print("Evaluation failed.")
        return 1
    out_path = results_base / f"{args.pud_config}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"BPEC: {res['bpec']:.4f}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    exit(main() or 0)
