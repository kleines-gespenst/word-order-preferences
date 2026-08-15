#!/usr/bin/env python3
"""
Evaluate goldfish models on FLORES dataset.

Classical perplexity evaluation:
- Cross-entropy loss in nats
- Perplexity = exp(loss)
- Full sequence scoring
- Proper special token handling (CLS/BOS prepend, NO EOS - matching goldfish paper)
- UNK tokens assigned random-chance loss (like goldfish paper)
"""

import os

# Use /share for Hugging Face cache (home has limited quota)
_share_cache = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", _share_cache)
os.environ.setdefault("HF_HUB_CACHE", f"{_share_cache}/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", _share_cache)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_share_cache}/hub")
os.makedirs(_share_cache, exist_ok=True)
os.makedirs(f"{_share_cache}/hub", exist_ok=True)


def _dir_size(path: str) -> int:
    """Return total size in bytes of directory."""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _fmt_size(n: float) -> str:
    """Format bytes as human-readable size."""
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def _report_cache_sizes() -> None:
    """Print disk usage of Hugging Face cache (home and /share). Uses du if available for speed."""
    import subprocess
    home_cache = os.path.expanduser("~/.cache/huggingface")
    paths = [
        ("/share (this run uses)", _share_cache),
        ("home (~/.cache/huggingface)", home_cache),
    ]
    print("Hugging Face cache disk usage:")
    for label, path in paths:
        if not os.path.isdir(path):
            print(f"  {label}: (not present)")
            continue
        try:
            r = subprocess.run(
                ["du", "-sh", path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r.returncode == 0 and r.stdout:
                size_str = r.stdout.split()[0]
                print(f"  {label}: {size_str}")
            else:
                print(f"  {label}: {_fmt_size(_dir_size(path))}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  {label}: {_fmt_size(_dir_size(path))}")
    print()


_report_cache_sizes()

import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from datasets import DownloadMode, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from goldfish_flores_lang import facebook_flores_config_candidates


def compute_perplexity(
    model,
    tokenizer,
    sentences: List[str],
    device: str = "cuda",
    max_length: int = 512,
    only_second_half: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute classical perplexity with proper special token handling.
    
    - Prepends CLS/BOS token if available (goldfish uses CLS)
    - NO EOS appended (matching goldfish paper methodology)
    - Cross-entropy loss in nats
    - UNK tokens get random-chance loss: log(vocab_size)
    - Perplexity = exp(mean loss per token)
    - Optionally only score second half (matching goldfish paper)
    """
    model.eval()
    
    # Determine special tokens
    # Goldfish models use CLS as the start token
    if tokenizer.cls_token_id is not None:
        start_token_id = tokenizer.cls_token_id
        start_token_name = "CLS"
    else:
        start_token_id = None
        start_token_name = None
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    unk_token_id = tokenizer.unk_token_id
    vocab_size = tokenizer.vocab_size
    
    
    # Use reduction='none' so we can modify individual token losses
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
    
    total_loss = 0.0
    total_tokens = 0
    total_unk_tokens = 0
    per_sentence_losses = []
    per_sentence_tokens = []
    
    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Computing perplexity", disable=not verbose):
            # Tokenize without special tokens (we handle them manually)
            encoded = tokenizer(sentence, add_special_tokens=False)
            input_ids = list(encoded["input_ids"])
            
            # Prepend start token (CLS for goldfish)
            if start_token_id is not None:
                input_ids.insert(0, start_token_id)
            
            # NOTE: No EOS appended (matching goldfish paper methodology)
            
            # Skip sequences too short for next-token prediction
            if len(input_ids) < 2:
                continue
            
            # Truncate if needed
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            
            # Convert to tensor
            input_ids_t = torch.tensor([input_ids], device=device)
            
            # Forward pass
            outputs = model(input_ids=input_ids_t, return_dict=True)
            logits = outputs.logits  # (1, seq_len, vocab_size)
            
            # Shift for next-token prediction
            # Position i predicts token i+1
            shift_logits = logits[:, :-1, :].contiguous()  # (1, seq_len-1, vocab_size)
            shift_labels = input_ids_t[:, 1:].contiguous()  # (1, seq_len-1)
            
            # Compute per-token loss
            # Transpose for CrossEntropyLoss: (1, vocab_size, seq_len-1)
            logits_transposed = torch.transpose(shift_logits, 1, 2)
            losses = loss_fn(logits_transposed, shift_labels).cpu()  # (1, seq_len-1)
            
            # UNK tokens get random-chance loss (in nats)
            if unk_token_id is not None:
                unk_mask = shift_labels.cpu() == unk_token_id
                num_unk = unk_mask.sum().item()
                total_unk_tokens += num_unk
                losses[unk_mask] = np.log(vocab_size)
            
            # Only score second half (by characters) - matching goldfish paper
            if only_second_half:
                half_char_idx = len(sentence) // 2
                halfline = sentence[:half_char_idx]
                halfline_tokens = tokenizer(halfline, add_special_tokens=False)['input_ids']
                halfline_len_tokens = len(halfline_tokens)
                # Zero out first-half losses
                losses[0, :halfline_len_tokens] = 0.0
                num_tokens = losses.shape[1] - halfline_len_tokens
            else:
                num_tokens = shift_labels.numel()
            
            # Sum losses for this sentence
            sentence_loss = torch.sum(losses).item()
            
            per_sentence_losses.append(sentence_loss)
            per_sentence_tokens.append(num_tokens)
            
            total_loss += sentence_loss
            total_tokens += num_tokens
    
    # Aggregate
    mean_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(mean_loss)
    
    per_sentence_ppl = [
        np.exp(l / t) if t > 0 else float('inf')
        for l, t in zip(per_sentence_losses, per_sentence_tokens)
    ]
    
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "total_loss": total_loss,
        "total_tokens": total_tokens,
        "total_unk_tokens": total_unk_tokens,
        "num_sentences": len(per_sentence_losses),
        "per_sentence_ppl": per_sentence_ppl,
        "mean_sentence_ppl": float(np.mean(per_sentence_ppl)),
        "median_sentence_ppl": float(np.median(per_sentence_ppl)),
    }


def load_flores_dataset(
    split: str = "devtest",
    lang: str = "eng_Latn",
    dataset_name: str = "facebook/flores",
    force_redownload: bool = False,
) -> List[str]:
    """Load FLORES dataset sentences."""

    if dataset_name == "facebook/flores":
        candidates = facebook_flores_config_candidates(lang)
    else:
        candidates = [lang]

    load_kw = {"trust_remote_code": True}
    if force_redownload:
        load_kw["download_mode"] = DownloadMode.FORCE_REDOWNLOAD

    last_err: Optional[Exception] = None
    dataset = None
    for hub_lang in candidates:
        try:
            dataset = load_dataset(dataset_name, hub_lang, **load_kw)
            break
        except Exception as e:
            last_err = e
            continue
    if dataset is None:
        raise last_err  # type: ignore[misc]
    sentences = dataset[split]["sentence"]
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
    Compute length-normalised bits per character (BPEC).

    Log probs are from the model on **its language** (sentences).
    Normalisation is by a fixed reference length (English character count)
    so that BPEC is comparable across languages.

    BPEC = -sum(log p(token_t | context)) / (reference_char_count * log(2))
    """
    model.eval()

    if tokenizer.cls_token_id is not None:
        start_token_id = tokenizer.cls_token_id
    else:
        start_token_id = None

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

    import math
    if reference_char_count > 0:
        bpec = -total_log_prob / (reference_char_count * math.log(2))
    else:
        bpec = 0.0

    return bpec


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate goldfish model on FLORES (classical perplexity)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="goldfish-models/eng_latn_1000mb",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="facebook/flores",
        help="FLORES dataset to use",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng_Latn",
        help="Language code",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="devtest",
        choices=["dev", "devtest"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress bar",
    )
    parser.add_argument(
        "--only_second_half",
        action="store_true",
        help="Only score second half of each sentence (matching goldfish paper default)",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Pass download_mode=force_redownload to load_dataset (refresh HF dataset cache)",
    )
    args = parser.parse_args()
    if os.environ.get("FLORES_FORCE_REDOWNLOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        args.force_redownload = True
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Language: {args.lang}")
    print(f"  Split: {args.split}")
    print(f"  Device: {args.device}")
    print(f"  Max length: {args.max_length}")
    print(f"  Only second half: {args.only_second_half}")
    print(f"  Force HF redownload: {args.force_redownload}")
    
    # Load model and tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(args.device)
    model.eval()
    
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Inspect tokenizer
    inspect_tokenizer(tokenizer)
    
    # Load dataset
    print("\n" + "-" * 40)
    sentences = load_flores_dataset(
        split=args.split,
        lang=args.lang,
        dataset_name=args.dataset,
        force_redownload=args.force_redownload,
    )
    
    # Compute perplexity
    
    results = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        device=args.device,
        max_length=args.max_length,
        only_second_half=args.only_second_half,
        verbose=args.verbose,
    )
    
    # Compute BPEC: model on its language, normalised by English character length
    english_sentences = load_flores_dataset(
        split=args.split,
        lang="eng_Latn",
        dataset_name=args.dataset,
        force_redownload=args.force_redownload,
    )
    reference_char_count = sum(len(s) for s in english_sentences)
    print(f"  Reference (English) character count: {reference_char_count:,}")
    bpec = compute_bpec(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        reference_char_count=reference_char_count,
        device=args.device,
        max_length=args.max_length,
        verbose=args.verbose,
    )
    avg_tokens_per_sentence = (
    results["total_tokens"] / results["num_sentences"]
    if results["num_sentences"] > 0
    else 0.0
    )
    
    # Print results
    print("RESULTS")
    print(f"  Loss (nats/token):       {results['loss']:.4f}")
    print(f"  Perplexity:              {results['perplexity']:.4f}")
    print(f"  Total tokens:            {results['total_tokens']:,}")
    print(f"  UNK tokens:              {results['total_unk_tokens']:,}")
    print(f"  Sentences evaluated:     {results['num_sentences']:,}")
    print(f"  Avg tokens/sentence:    {avg_tokens_per_sentence:.2f}")
    print(f"  Mean sentence PPL:       {results['mean_sentence_ppl']:.4f}")
    print(f"  Median sentence PPL:     {results['median_sentence_ppl']:.4f}")
    print(f"  BPEC (model lang, norm by eng length): {bpec:.4f}")
    
    # Save results
    output_data = {
        "model": args.model,
        "dataset": args.dataset,
        "language": args.lang,
        "split": args.split,
        "max_length": args.max_length,
        "only_second_half": args.only_second_half,
        "timestamp": datetime.now().isoformat(),
        "loss": results["loss"],
        "perplexity": results["perplexity"],
        "total_tokens": results["total_tokens"],
        "total_unk_tokens": results["total_unk_tokens"],
        "num_sentences": results["num_sentences"],
        "mean_sentence_ppl": results["mean_sentence_ppl"],
        "median_sentence_ppl": results["median_sentence_ppl"],
        "bpec": bpec,
    }
    
    if args.output:
        output_path = args.output
    else:
        model_name = args.model.replace("/", "_")
        output_path = f"ppl_{model_name}_{args.lang}_{args.split}.json"
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n Results saved: {output_path}")


if __name__ == "__main__":
    main()