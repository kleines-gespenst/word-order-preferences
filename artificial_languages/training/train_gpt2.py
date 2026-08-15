#!/usr/bin/env python3
"""
Language Model Training — SentencePiece BPE tokenization (15k vocab).

Replaces the earlier word-level tokenization with
SentencePiece BPE and adds per-token surprisal output.

Tokenizer note:
  Languages differ only in word order → identical word-type inventory → identical
  BPE merges (which are character-level, not order-dependent). One SP tokenizer
  trained on any grammar's training file is sufficient for all grammars.
  Pass --tokenizer_path to reuse an already-trained model and skip re-training.

Extra outputs vs. goldfish:
  - Per-token surprisal in bits (−log₂ p(tₖ | t<k)) saved as JSON
  - BPC is kept (computed from raw character counts, same formula)
"""

import os
import json
import math
import argparse
import time
from itertools import chain
from datetime import datetime
from pathlib import Path

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

try:
    import sentencepiece as spm
except ImportError as e:
    raise ImportError("Install sentencepiece:  pip install sentencepiece") from e


# ─────────────────────────────────────────────
# Compute Cost Tracking  (unchanged from goldfish)
# ─────────────────────────────────────────────
class ComputeTracker:
    """Track computation costs: time, FLOPs, GPU hours."""

    def __init__(self):
        self.timings = {}
        self.start_times = {}
        self.gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def start(self, phase):
        self.start_times[phase] = time.time()

    def stop(self, phase):
        if phase in self.start_times:
            elapsed = time.time() - self.start_times[phase]
            self.timings[phase] = self.timings.get(phase, 0) + elapsed
            return elapsed
        return 0

    def get_hours(self, phase):
        return self.timings.get(phase, 0) / 3600

    def get_gpu_hours(self, phase):
        return self.get_hours(phase) * max(1, self.gpu_count)

    @staticmethod
    def estimate_training_flops(num_params, num_tokens):
        return 6 * num_params * num_tokens

    @staticmethod
    def estimate_inference_flops(num_params, num_tokens):
        return 2 * num_params * num_tokens

    @staticmethod
    def format_flops(flops):
        for unit, label in [(1e21, "ZFLOPs"), (1e18, "EFLOPs"), (1e15, "PFLOPs"),
                            (1e12, "TFLOPs"), (1e9,  "GFLOPs"), (1e6,  "MFLOPs")]:
            if flops >= unit:
                return f"{flops/unit:.2f} {label}"
        return f"{flops:.2e} FLOPs"


# ─────────────────────────────────────────────
# Fairseq-style logger  (unchanged from goldfish)
# ─────────────────────────────────────────────
class PplLoggingCallback(TrainerCallback):
    """Log eval_*_perplexity to W&B at every eval step (derived from eval_*_loss)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        ppl_logs = {}
        for key, val in logs.items():
            if key.startswith("eval_") and key.endswith("_loss"):
                try:
                    ppl_logs[key[:-len("_loss")] + "_perplexity"] = math.exp(float(val))
                except (ValueError, OverflowError):
                    pass
        if ppl_logs:
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log(ppl_logs, step=state.global_step)
            except ImportError:
                pass


class FairseqStyleLogger(TrainerCallback):
    def __init__(self, total_train_tokens, num_train_samples, log_every=50):
        self.total_train_tokens = total_train_tokens
        self.num_train_samples  = num_train_samples
        self.start_time         = None
        self.epoch_start_time   = None
        self.log_every          = log_every

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        ts = self.start_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{ts} | INFO | Starting training")
        print(f"{ts} | INFO | Total training samples: {self.num_train_samples:,}")
        print(f"{ts} | INFO | Total training tokens:  {self.total_train_tokens:,}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = datetime.now()
        epoch = float(state.epoch) if state.epoch else 0
        ts = self.epoch_start_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{ts} | INFO | Begin training epoch {epoch}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step  = state.global_step
        now   = datetime.now()
        ts    = now.strftime('%Y-%m-%d %H:%M:%S')
        epoch = float(state.epoch) if state.epoch else 0

        if "loss" in logs and "learning_rate" in logs and step % self.log_every == 0:
            loss = float(logs["loss"])
            ppl  = float(np.exp(loss))
            lr   = float(logs["learning_rate"])
            elapsed = (now - self.start_time).total_seconds() if self.start_time else 1
            wps = step * args.per_device_train_batch_size * args.gradient_accumulation_steps * 512 / elapsed
            print(f"{ts} | INFO | train | epoch {epoch} | step {step:,} | "
                  f"loss {loss:.3f} | ppl {ppl:.2f} | lr {lr:.5e} | wps {wps:.1f}")

        if "eval_validation_loss" in logs:
            eval_loss = float(logs["eval_validation_loss"])
            eval_ppl  = float(np.exp(eval_loss))
            print(f"{ts} | INFO | valid | epoch {epoch} | step {step:,} | "
                  f"valid_loss {eval_loss:.3f} | valid_ppl {eval_ppl:.2f}")

        if "eval_test_loss" in logs:
            test_loss = float(logs["eval_test_loss"])
            test_ppl  = float(np.exp(test_loss))
            print(f"{ts} | INFO | test  | epoch {epoch} | step {step:,} | "
                  f"test_loss {test_loss:.3f} | test_ppl {test_ppl:.2f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        now   = datetime.now()
        epoch = float(state.epoch) if state.epoch else 0
        ts    = now.strftime('%Y-%m-%d %H:%M:%S')
        if self.epoch_start_time:
            secs = (now - self.epoch_start_time).total_seconds()
            print(f"{ts} | INFO | End of epoch {epoch} (took {secs:.1f}s)")

    def on_train_end(self, args, state, control, **kwargs):
        now = datetime.now()
        ts  = now.strftime('%Y-%m-%d %H:%M:%S')
        if self.start_time:
            total = (now - self.start_time).total_seconds()
            print(f"\n{ts} | INFO | Done training in {total:.1f} seconds")


# ─────────────────────────────────────────────
# SentencePiece helpers
# ─────────────────────────────────────────────
SP_VOCAB_SIZE = 15_000


def train_sentencepiece(train_file: str, model_prefix: str,
                        vocab_size: int = SP_VOCAB_SIZE,
                        cross_word: bool = False,
                        model_type: str = "bpe") -> str:
    """
    Train a SentencePiece tokenizer (BPE or Unigram).

    model_type="bpe"      : greedy merge-based subwords.
    model_type="unigram"  : probabilistic subword LM (SP default).

    The word-boundary property is controlled by split_by_whitespace, NOT by
    model_type — so a Unigram tokenizer trained with cross_word=False is just as
    grammar-order-invariant as the word-boundary BPE, and ONE shared tokenizer is
    valid for every grammar.

    cross_word=False  (word-boundary):
        split_by_whitespace=True → pieces stay within words.
        Unaffected by word order → ONE tokenizer valid for all grammars.

    cross_word=True  (standard SP / cross-word):
        Spaces encoded as ▁, pieces can cross word boundaries.
        Cross-boundary statistics depend on word order → each grammar
        needs its own tokenizer AND tokenizations differ across grammars,
        making cross-grammar PPL comparison less clean.
    """
    mode_label = "cross-word" if cross_word else "word-boundary"
    print(f"  Training file : {train_file}")
    print(f"  Model prefix  : {model_prefix}")
    print(f"  Vocab size    : {vocab_size:,}")
    print(f"  Algorithm     : {model_type}")
    print(f"  Boundary mode : {mode_label}")

    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        pad_id=0,  pad_piece="<pad>",
        eos_id=1,  eos_piece="</s>",
        unk_id=2,  unk_piece="<unk>",
        bos_id=3,  bos_piece="<s>",
        # word-boundary mode: pre-tokenise on whitespace, BPE stays within words
        split_by_whitespace=not cross_word,
        # always add leading space so first word gets the same ▁-prefixed pieces
        # as mid-sentence words — otherwise position-0 arguments (e.g. S in SVO)
        # get different token IDs, biasing the model toward certain word orders
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
    )
    model_path = f"{model_prefix}.model"
    print(f" SP model saved: {model_path}")
    return model_path


def encode_file(sp: spm.SentencePieceProcessor, file_path: str, eos_id: int) -> list:
    """Encode all lines, appending EOS once per line; return flat id list."""
    all_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_ids.extend(sp.EncodeAsIds(line))
            all_ids.append(eos_id)
    return all_ids


# ─────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────
def to_fixed_chunks_from_ids(all_ids, block=512):
    """Cut flat id list into fixed-size chunks; drop tail (Fairseq behaviour)."""
    n    = len(all_ids) // block
    used = n * block
    if n == 0:
        return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []}), 0, 0
    chunks = [all_ids[i: i + block] for i in range(0, used, block)]
    attn   = [[1] * block] * n
    return Dataset.from_dict({"input_ids": chunks, "attention_mask": attn, "labels": chunks}), n, used


def count_characters_in_file(file_path: str) -> int:
    total = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total += len(line.strip())
    return total


# ─────────────────────────────────────────────
# Per-token surprisal
# ─────────────────────────────────────────────
@torch.no_grad()
def compute_per_token_surprisal(model, dataset, device, batch_size=8):
    """
    For every chunk compute per-token surprisal in bits: −log₂ p(tₖ | t<k).

    Each chunk of length T yields T−1 surprisal values (token 0 has no left
    context within the chunk and is therefore excluded).

    Returns:
        surprisals : list[list[float]]  — one inner list per chunk
        mean_bits  : float              — mean surprisal across all tokens
    """
    model.eval()
    model.to(device)

    all_surprisals  = []
    total_nll_nats  = 0.0
    total_tokens    = 0

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids":      torch.tensor([b["input_ids"]      for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        },
    )

    for batch in loader:
        input_ids = batch["input_ids"].to(device)       # (B, T)
        attn_mask = batch["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits  # (B, T, V)

        # logits[:, i] predicts input_ids[:, i+1]
        shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = input_ids[:, 1:].contiguous()     # (B, T-1)

        log_probs   = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_nll   = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1) nats
        token_bits  = token_nll / math.log(2)            # → bits

        for i in range(input_ids.size(0)):
            surp = token_bits[i].cpu().tolist()
            all_surprisals.append(surp)
            total_nll_nats += token_nll[i].sum().item()
            total_tokens   += len(surp)

    mean_bits = (total_nll_nats / total_tokens) / math.log(2) if total_tokens > 0 else float("nan")
    return all_surprisals, mean_bits


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 LM training with SentencePiece BPE (15k vocab)"
    )
    parser.add_argument("--grammar",        type=str, required=True)
    parser.add_argument("--split",          type=str, required=True)
    parser.add_argument("--cross_word_bpe", action="store_true", default=False,
                        help="Use cross-word BPE (standard SentencePiece: spaces → ▁, "
                             "merges can cross word boundaries). "
                             "Default: word-boundary BPE (split_by_whitespace=True, "
                             "merges stay within words, one tokenizer valid for all grammars). "
                             "Cross-word mode trains a separate tokenizer per grammar because "
                             "cross-boundary bigram frequencies depend on word order.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to a pre-trained SP .model file (skips tokenizer training). "
                             "For word-boundary BPE the script auto-reuses a shared tokenizer "
                             "from BASE_MODELS/shared_tokenizer/ if one exists.")
    parser.add_argument("--separate_tokenizer", action="store_true", default=False,
                        help="Word-boundary BPE: train a separate tokenizer for this grammar "
                             "(saved in output_dir) instead of using/creating the shared one. "
                             "Use this to verify that per-grammar word-boundary tokenizers "
                             "produce identical vocabularies across grammars.")
    parser.add_argument("--tokenizer_algo", type=str, default="bpe",
                        choices=["bpe", "unigram"],
                        help="SentencePiece algorithm. 'bpe' (default) reproduces the "
                             "existing runs; 'unigram' trains a Unigram LM tokenizer. "
                             "Word-boundary mode (the default) is order-invariant for both, "
                             "so a single shared tokenizer is used across all grammars either way.")
    parser.add_argument("--vocab_size",     type=int, default=SP_VOCAB_SIZE)
    parser.add_argument("--surprisal_batch_size", type=int, default=8)
    parser.add_argument("--tag",            type=str, default=None,
                        help="Optional suffix appended to output folder names (e.g. 'a40').")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default=None,
                        help="Root data directory containing permuted_splits/. "
                             "Defaults to $WORD_ORDER_DATA/word_order_data. "
                             "The basename is used to replace '5MB' in model/results dir names "
                             "(e.g. word_order_data_10MB → '10MB').")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Train for this many epochs instead of MAX_STEPS=5000. "
                             "steps_per_epoch is computed from the actual chunk count after "
                             "data loading, so MAX_STEPS and the LR scheduler are updated "
                             "consistently. WARMUP_STEPS stays at 10%% of MAX_STEPS.")
    parser.add_argument("--eval-steps", type=int, default=200,
                        help="Evaluate (and save) every N steps (default: 200). "
                             "Use a smaller value (e.g. 50) for finer training-dynamics plots.")
    parser.add_argument("--no-surprisal", action="store_true", default=False,
                        help="Skip per-token surprisal computation.")
    parser.add_argument("--no-test", action="store_true", default=False,
                        help="Skip test set evaluation (training + validation only).")
    args = parser.parse_args()

    compute_tracker = ComputeTracker()
    compute_tracker.start("total")

    # bpe_mode drives paths, W&B names, and tokenizer training behaviour
    bpe_mode = "crossword" if args.cross_word_bpe else "wordboundary"
    # algo selects the SentencePiece algorithm ("bpe" | "unigram") and is used in
    # the model/results dir names so unigram runs never collide with bpe runs, e.g.
    #   word_order_models_5MB_bpe15000_wordboundary
    #   word_order_models_5MB_unigram15000_wordboundary
    algo = args.tokenizer_algo

    print(f"Grammar  : {args.grammar}")
    print(f"Split    : {args.split}")
    print(f"BPE mode : {bpe_mode}  "
          f"({'cross-word: merges can span spaces' if args.cross_word_bpe else 'word-boundary: merges stay within words, shared tokenizer'})")

    # ── Paths ──
    # Naming convention mirrors goldfish (word-level) dirs for easy comparison:
    #   word_order_models_5MB            ← goldfish / word-level
    #   word_order_models_5MB_bpe15k_wordboundary  ← this script, word-boundary BPE
    #   word_order_models_5MB_bpe15k_crossword      ← this script, cross-word BPE
    BASE_DATA    = args.data_dir if args.data_dir else str(DATA_ROOT / "word_order_data")
    # Derive a size label from the data dir basename (e.g. "word_order_data_10MB" → "10MB",
    # "word_order_data" → "5MB" for backwards compatibility).
    import re as _re
    _basename = os.path.basename(BASE_DATA.rstrip("/"))
    _m = _re.search(r"_(\d+[A-Za-z]+)$", _basename)
    data_size_label = _m.group(1) if _m else "5MB"
    tag_suffix   = f"_{args.tag}" if args.tag else ""
    BASE_MODELS  = f"{DATA_ROOT}/word_order_models_{data_size_label}_{algo}{args.vocab_size}_{bpe_mode}{tag_suffix}"
    BASE_RESULTS = f"{DATA_ROOT}/word_order_models_{data_size_label}_{algo}{args.vocab_size}_{bpe_mode}{tag_suffix}_results"

    data_dir   = f"{BASE_DATA}/permuted_splits/{args.grammar}"
    train_file = f"{data_dir}/{args.split}.trn"
    valid_file = f"{data_dir}/{args.split}.dev"
    test_file  = f"{data_dir}/{args.split}.tst"

    output_dir    = f"{BASE_MODELS}/gpt2-checkpoints/{args.grammar}/{args.split}"
    results_dir   = f"{BASE_RESULTS}/gpt2-results"
    surprisal_dir = f"{BASE_RESULTS}/gpt2-surprisals/{args.grammar}"

    for d in (output_dir, results_dir, surprisal_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\nPaths:")
    print(f"  Train     : {train_file}")
    print(f"  Valid     : {valid_file}")
    print(f"  Test      : {test_file}")
    print(f"  Output    : {output_dir}")
    print(f"  Results   : {results_dir}")
    print(f"  Surprisals: {surprisal_dir}")

    # ── Hyperparameters  (same as goldfish) ──
    MAX_LENGTH    = 512
    BATCH_SIZE    = 4
    GRAD_ACCUM    = 16
    MAX_STEPS     = 5_000
    LEARNING_RATE = 1e-4
    WARMUP_STEPS  = int(0.1 * MAX_STEPS)  # 10% of total steps

    torch.manual_seed(1)
    np.random.seed(1)

    compute_tracker.start("preprocessing")

    # ── SentencePiece tokenizer ──
    if args.tokenizer_path is not None:
        # Explicit path always wins
        sp_model_path = args.tokenizer_path
        print(f"\n Using explicitly provided SP model: {sp_model_path}")

    elif not args.cross_word_bpe:
        if args.separate_tokenizer:
            # Train a per-grammar word-boundary tokenizer (for comparison purposes).
            # After training, compare its vocabulary to any existing shared tokenizer.
            sp_prefix     = f"{output_dir}/{algo}{args.vocab_size}_wordboundary"
            sp_model_path = train_sentencepiece(
                train_file, sp_prefix, args.vocab_size, cross_word=False, model_type=algo
            )
            # Compare vocab with shared tokenizer if it exists
            shared_tok_path = f"{BASE_MODELS}/shared_tokenizer/{algo}{args.vocab_size}_wordboundary.model"
            if os.path.exists(shared_tok_path):
                print("\n── Tokenizer vocab comparison ──")
                sp_ref = spm.SentencePieceProcessor()
                sp_ref.Load(shared_tok_path)
                sp_new = spm.SentencePieceProcessor()
                sp_new.Load(sp_model_path)
                n = sp_ref.GetPieceSize()
                mismatches = [(i, sp_ref.IdToPiece(i), sp_new.IdToPiece(i))
                              for i in range(n) if sp_ref.IdToPiece(i) != sp_new.IdToPiece(i)]
                if not mismatches:
                    print(f"   Vocabularies IDENTICAL to shared tokenizer ({n} pieces)")
                else:
                    print(f"   {len(mismatches)} mismatches out of {n} pieces")
                    for i, ref_piece, new_piece in mismatches[:20]:
                        print(f"    id={i}  shared={repr(ref_piece)}  this={repr(new_piece)}")
                    if len(mismatches) > 20:
                        print(f"    ... and {len(mismatches)-20} more")
                del sp_ref, sp_new
            else:
                # No shared tokenizer yet — save this one as the shared reference
                import shutil
                shared_tok_dir = f"{BASE_MODELS}/shared_tokenizer"
                os.makedirs(shared_tok_dir, exist_ok=True)
                shutil.copy(sp_model_path, shared_tok_path)
                vocab_src = sp_model_path.replace(".model", ".vocab")
                if os.path.exists(vocab_src):
                    shutil.copy(vocab_src, shared_tok_path.replace(".model", ".vocab"))
                print(f"\n   Saved as shared reference tokenizer: {shared_tok_path}")
        else:
            # Word-boundary BPE: tokenizer is grammar-order-invariant → share one
            # across all grammars.  Train it once; every subsequent run reuses it.
            shared_tok_dir  = f"{BASE_MODELS}/shared_tokenizer"
            shared_tok_path = f"{shared_tok_dir}/{algo}{args.vocab_size}_wordboundary.model"
            os.makedirs(shared_tok_dir, exist_ok=True)
            if os.path.exists(shared_tok_path):
                sp_model_path = shared_tok_path
                print(f"\n Reusing shared word-boundary tokenizer: {sp_model_path}")
            else:
                print(f"\n  No shared tokenizer found — training one on {train_file}")
                print(f"  (will be reused by all subsequent grammars)")
                # Train to a per-process temp prefix, then atomically rename so
                # concurrent array tasks never read a half-written file.
                tmp_prefix = f"{shared_tok_dir}/.tmp_{os.getpid()}_{algo}{args.vocab_size}_wordboundary"
                train_sentencepiece(train_file, tmp_prefix, args.vocab_size, cross_word=False, model_type=algo)
                tmp_model = f"{tmp_prefix}.model"
                tmp_vocab = f"{tmp_prefix}.vocab"
                os.replace(tmp_model, shared_tok_path)
                if os.path.exists(tmp_vocab):
                    os.replace(tmp_vocab, shared_tok_path.replace(".model", ".vocab"))
                sp_model_path = shared_tok_path

    else:
        # Cross-word BPE: cross-boundary merges depend on word order → train
        # a separate tokenizer for this grammar.
        sp_prefix     = f"{output_dir}/{algo}{args.vocab_size}_crossword"
        sp_model_path = train_sentencepiece(
            train_file, sp_prefix, args.vocab_size, cross_word=True, model_type=algo
        )

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)

    actual_vocab_size = sp.GetPieceSize()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()

    print(f"\nSP vocab size: {actual_vocab_size:,}  eos_id={eos_id}  pad_id={pad_id}  unk_id={sp.unk_id()}")

    # Save SP model path for reproducibility
    with open(f"{output_dir}/sp_model_path.txt", "w") as f:
        f.write(sp_model_path + "\n")

    # ── Character counts for BPC (same as goldfish) ──
    train_chars = count_characters_in_file(train_file)
    valid_chars = count_characters_in_file(valid_file)
    test_chars  = count_characters_in_file(test_file)
    print(f"  Train: {train_chars:,} characters")
    print(f"  Valid: {valid_chars:,} characters")
    print(f"  Test:  {test_chars:,} characters")

    # ── Inspect tokenization ──
    with open(train_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    sample_pieces = sp.EncodeAsPieces(first_line)
    sample_ids    = sp.EncodeAsIds(first_line)
    print(f"\nFirst line (raw):    {first_line[:120]}")
    print(f"First 20 pieces:     {sample_pieces[:20]}")
    print(f"First 20 token IDs:  {sample_ids[:20]}")

    # ── Encode datasets ──
    train_ids = encode_file(sp, train_file, eos_id)
    valid_ids = encode_file(sp, valid_file, eos_id)
    test_ids  = encode_file(sp, test_file,  eos_id)

    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Valid tokens: {len(valid_ids):,}")
    print(f"  Test  tokens: {len(test_ids):,}")

    # ── Chunk into fixed-length sequences ──
    train_chunked, n_train, used_train = to_fixed_chunks_from_ids(train_ids, MAX_LENGTH)
    valid_chunked, n_valid, used_valid = to_fixed_chunks_from_ids(valid_ids, MAX_LENGTH)
    test_chunked,  n_test,  used_test  = to_fixed_chunks_from_ids(test_ids,  MAX_LENGTH)

    print(f"\nChunking (block={MAX_LENGTH}):")
    print(f"  Train: {len(train_ids):,} tokens → {len(train_chunked):,} chunks  (discarded {len(train_ids)-used_train:,})")
    print(f"  Valid: {len(valid_ids):,} tokens → {len(valid_chunked):,} chunks  (discarded {len(valid_ids)-used_valid:,})")
    print(f"  Test:  {len(test_ids):,}  tokens → {len(test_chunked):,} chunks  (discarded {len(test_ids)-used_test:,})")

    if len(train_chunked) > 0:
        sample = train_chunked[0]
        print(f"\nSample chunk — first 20 IDs : {sample['input_ids'][:20]}")
        print(f"               last  20 IDs : {sample['input_ids'][-20:]}")

    compute_tracker.stop("preprocessing")

    # ── Resolve epoch-based budget (must happen after chunking) ──
    if args.max_epochs is not None:
        steps_per_epoch = math.ceil(len(train_chunked) / (BATCH_SIZE * GRAD_ACCUM))
        MAX_STEPS    = args.max_epochs * steps_per_epoch
        WARMUP_STEPS = int(0.1 * MAX_STEPS)
        print(f"\nEpoch budget: {args.max_epochs} epochs × {steps_per_epoch} steps/epoch "
              f"= {MAX_STEPS} steps  (warmup: {WARMUP_STEPS})")

    # ── Model config  (same as goldfish: 4 layers, n_inner=2048) ──
    config = GPT2Config(
        vocab_size=actual_vocab_size,
        n_positions=MAX_LENGTH,
        n_embd=512,
        n_layer=4,
        n_head=8,
        n_inner=2048,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        activation_function="relu",
        tie_word_embeddings=True,
        add_cross_attention=False,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    model     = GPT2LMHeadModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Vocab: {config.vocab_size}  Layers: {config.n_layer}  Heads: {config.n_head}  "
          f"Hidden: {config.n_embd}  n_inner: {config.n_inner}  Params: {num_params:,}")

    # ── W&B ──
    use_wandb = True
    try:
        import wandb
    except ImportError:
        use_wandb = False
        print("\n[WARNING] wandb not installed, proceeding without logging")

    wandb_project = os.environ.get("WANDB_PROJECT", f"word_order_lm_5MB_bpe_new_hyperparams")
    wandb_name    = f"{args.grammar}.{args.split}"

    # ── Training args  (same as goldfish) ──
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=MAX_STEPS,
        num_train_epochs=1000,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=0.0,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=1.0,
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_loss",
        greater_is_better=False,
        logging_first_step=True,
        logging_steps=50,
        disable_tqdm=True,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["wandb"] if use_wandb else "none",
        run_name=wandb_name if use_wandb else None,
        seed=1,
        data_seed=1,
    )

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            group=args.grammar,
            config={
                "grammar":        args.grammar,
                "split":          args.split,
                "tokenizer":      f"sentencepiece_{algo}_{bpe_mode}",
                "bpe_mode":       bpe_mode,
                "vocab_size":     actual_vocab_size,
                "sp_model":       sp_model_path,
                "max_length":     MAX_LENGTH,
                "batch_size":     BATCH_SIZE,
                "grad_accum":     GRAD_ACCUM,
                "effective_batch": BATCH_SIZE * GRAD_ACCUM,
                "max_steps":      MAX_STEPS,
                "learning_rate":  LEARNING_RATE,
                "warmup_steps":   WARMUP_STEPS,
                "train_chunks":   len(train_chunked),
                "valid_chunks":   len(valid_chunked),
                "test_chunks":    len(test_chunked),
                "eos_per_line":   True,
            },
        )

    # ── Optimizer + scheduler  (goldfish hyperparameters) ──
    from torch.optim import Adam
    from transformers import get_linear_schedule_with_warmup
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.0,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS,
    )

    # ── Train ──
    fairseq_logger = FairseqStyleLogger(
        total_train_tokens=len(train_ids),
        num_train_samples=len(train_chunked),
    )

    _eval_datasets = {"validation": valid_chunked}
    if not args.no_test:
        _eval_datasets["test"] = test_chunked

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_chunked,
        eval_dataset=_eval_datasets,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), fairseq_logger, PplLoggingCallback()],
        optimizers=(optimizer, scheduler),
    )

    compute_tracker.start("training")
    train_result = trainer.train()
    compute_tracker.stop("training")

    print(f"  Steps: {int(train_result.global_step):,}")
    print(f"  Training loss: {float(train_result.training_loss):.4f}")

    trainer.save_model(output_dir)
    print(f" Model saved: {output_dir}")

    # Remove intermediate checkpoint dirs — keep only the final best model in output_dir
    import shutil, glob as _glob
    for ckpt_dir in _glob.glob(f"{output_dir}/checkpoint-*"):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f" Checkpoint subdirectories removed (single-checkpoint mode)")

    # ── Standard evaluation (PPL + BPC) ──
    compute_tracker.start("evaluation")

    val_results = trainer.evaluate(eval_dataset=valid_chunked)
    val_loss = float(val_results["eval_loss"])
    val_ppl  = float(np.exp(val_loss))
    val_bpc  = (val_loss * len(valid_ids)) / (valid_chars * math.log(2)) if valid_chars > 0 else 0.0
    print(f"\nValidation | loss={val_loss:.4f}  ppl={val_ppl:.4f}  bpc={val_bpc:.4f}")

    if not args.no_test:
        test_results = trainer.evaluate(eval_dataset=test_chunked)
        test_loss = float(test_results["eval_loss"])
        test_ppl  = float(np.exp(test_loss))
        test_bpc  = (test_loss * len(test_ids)) / (test_chars * math.log(2)) if test_chars > 0 else 0.0
        print(f"Test       | loss={test_loss:.4f}  ppl={test_ppl:.4f}  bpc={test_bpc:.4f}")
    else:
        test_loss = test_ppl = test_bpc = None
        print("Test evaluation skipped (--no-test).")

    val_mean_bits = test_mean_bits = None

    if not args.no_surprisal:
        # ── Per-token surprisal ──
        print("surprisal = −log₂ p(tₖ | t<k),  unit: bits")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = trainer.model

        print("\nValidation set…")
        val_surprisals, val_mean_bits = compute_per_token_surprisal(
            model, valid_chunked, device, batch_size=args.surprisal_batch_size
        )
        print(f"  Mean surprisal: {val_mean_bits:.4f} bits/token  (PPL≈{2**val_mean_bits:.2f})")

        if not args.no_test:
            print("\nTest set…")
            test_surprisals, test_mean_bits = compute_per_token_surprisal(
                model, test_chunked, device, batch_size=args.surprisal_batch_size
            )
            print(f"  Mean surprisal: {test_mean_bits:.4f} bits/token  (PPL≈{2**test_mean_bits:.2f})")

        surprisal_meta = {
            "grammar":    args.grammar,
            "split":      args.split,
            "tokenizer":  f"sentencepiece_{algo}_{bpe_mode}",
            "bpe_mode":   bpe_mode,
            "vocab_size": actual_vocab_size,
            "sp_model":   sp_model_path,
            "unit":       "bits",
            "description": (
                "Per-token surprisal −log₂ p(tₖ | t<k) in bits. "
                "Each inner list corresponds to one 512-token chunk and contains "
                "511 values (token 0 has no within-chunk left context and is excluded)."
            ),
        }

        val_surp_path = f"{surprisal_dir}/{args.split}.dev.{bpe_mode}.surprisal.json"
        with open(val_surp_path, "w") as f:
            json.dump({
                "meta": {**surprisal_meta, "split_file": valid_file,
                         "n_chunks": len(val_surprisals),
                         "mean_surprisal_bits": val_mean_bits,
                         "ppl_from_surprisal": 2 ** val_mean_bits},
                "surprisals": val_surprisals,
            }, f)
        print(f"\n Val  surprisals → {val_surp_path}")

        if not args.no_test:
            test_surp_path = f"{surprisal_dir}/{args.split}.tst.{bpe_mode}.surprisal.json"
            with open(test_surp_path, "w") as f:
                json.dump({
                    "meta": {**surprisal_meta, "split_file": test_file,
                             "n_chunks": len(test_surprisals),
                             "mean_surprisal_bits": test_mean_bits,
                             "ppl_from_surprisal": 2 ** test_mean_bits},
                    "surprisals": test_surprisals,
                }, f)
            print(f" Test surprisals → {test_surp_path}")
    else:
        print("Surprisal computation skipped (--no-surprisal).")

    compute_tracker.stop("evaluation")
    compute_tracker.stop("total")

    # ── Compute costs  (same as goldfish) ──
    actual_steps   = int(train_result.global_step)
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * MAX_LENGTH
    total_tokens_processed = actual_steps * tokens_per_step

    training_flops  = ComputeTracker.estimate_training_flops(num_params, total_tokens_processed)
    inference_flops = ComputeTracker.estimate_inference_flops(num_params, len(valid_ids) + len(test_ids))
    total_flops     = training_flops + inference_flops

    print(f"\n  GPU: {compute_tracker.gpu_name}  ×{compute_tracker.gpu_count}")
    print(f"\n  Timing:")
    pre_s  = compute_tracker.timings.get("preprocessing", 0)
    trn_s  = compute_tracker.timings.get("training",      0)
    eva_s  = compute_tracker.timings.get("evaluation",    0)
    tot_s  = compute_tracker.timings.get("total",         0)
    print(f"    Preprocessing: {pre_s:.1f}s ({pre_s/60:.2f} min)")
    print(f"    Training:      {trn_s:.1f}s ({compute_tracker.get_gpu_hours('training'):.4f} GPU-hours)")
    print(f"    Evaluation:    {eva_s:.1f}s ({eva_s/60:.2f} min)")
    print(f"    Total:         {tot_s:.1f}s ({tot_s/3600:.4f} hours)")
    print(f"\n  FLOPs:")
    print(f"    Training:  {ComputeTracker.format_flops(training_flops)}  ({training_flops:.2e})")
    print(f"    Inference: {ComputeTracker.format_flops(inference_flops)}  ({inference_flops:.2e})")
    print(f"    Total:     {ComputeTracker.format_flops(total_flops)}  ({total_flops:.2e})")
    print(f"\n  Tokens processed: {total_tokens_processed:,} ({actual_steps:,} steps × {tokens_per_step:,})")

    # ── Save results JSON ──
    results = {
        "grammar":    args.grammar,
        "split":      args.split,
        "tokenizer":  f"sentencepiece_{algo}_{bpe_mode}",
        "bpe_mode":   bpe_mode,
        "vocab_size": actual_vocab_size,
        "sp_model":   sp_model_path,
        "num_parameters": num_params,
        "training": {
            "steps": actual_steps,
            "loss":  float(train_result.training_loss),
        },
        "validation": {
            "loss":                val_loss,
            "perplexity":          val_ppl,
            "bpc":                 val_bpc,
            "mean_surprisal_bits": val_mean_bits,
            "ppl_from_surprisal":  (2 ** val_mean_bits) if val_mean_bits is not None else None,
            "num_tokens":          len(valid_ids),
            "num_chars":           valid_chars,
        },
        "test": {
            "loss":                test_loss,
            "perplexity":          test_ppl,
            "bpc":                 test_bpc,
            "mean_surprisal_bits": test_mean_bits,
            "ppl_from_surprisal":  (2 ** test_mean_bits) if test_mean_bits is not None else None,
            "num_tokens":          len(test_ids) if not args.no_test else None,
            "num_chars":           test_chars if not args.no_test else None,
        },
        "compute_costs": {
            "gpu_name":              compute_tracker.gpu_name,
            "gpu_count":             compute_tracker.gpu_count,
            "training_flops":        training_flops,
            "inference_flops":       inference_flops,
            "total_flops":           total_flops,
            "training_gpu_hours":    compute_tracker.get_gpu_hours("training"),
            "evaluation_gpu_hours":  compute_tracker.get_gpu_hours("evaluation"),
            "total_gpu_hours":       compute_tracker.get_gpu_hours("total"),
            "preprocessing_seconds": pre_s,
            "training_seconds":      trn_s,
            "evaluation_seconds":    eva_s,
            "total_seconds":         tot_s,
            "tokens_processed":      total_tokens_processed,
        },
        "config": {
            "max_length":    MAX_LENGTH,
            "batch_size":    BATCH_SIZE,
            "grad_accum":    GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "warmup_steps":  WARMUP_STEPS,
            "optimizer":     "adam",
            "schedule":      "linear",
            "eos_per_line":  True,
        },
    }

    results_file = f"{results_dir}/{args.grammar}.{args.split}.{bpe_mode}.results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Results saved: {results_file}")

    if use_wandb:
        import wandb as _wandb
        wb_log = {
            "final/train_loss":           float(train_result.training_loss),
            "final/val_loss":             val_loss,
            "final/val_perplexity":       val_ppl,
            "final/val_bpc":              val_bpc,
            "compute/training_flops":     training_flops,
            "compute/total_flops":        total_flops,
            "compute/training_gpu_hours": compute_tracker.get_gpu_hours("training"),
            "compute/total_gpu_hours":    compute_tracker.get_gpu_hours("total"),
        }
        if test_loss is not None:
            wb_log.update({"final/test_loss": test_loss,
                           "final/test_perplexity": test_ppl,
                           "final/test_bpc": test_bpc})
        if val_mean_bits is not None:
            wb_log["final/val_surprisal_bits"] = val_mean_bits
        if test_mean_bits is not None:
            wb_log["final/test_surprisal_bits"] = test_mean_bits
        _wandb.log(wb_log)
        _wandb.run.summary["val_perplexity"] = val_ppl
        _wandb.run.summary["val_bpc"]        = val_bpc
        if test_ppl is not None:
            _wandb.run.summary["test_perplexity"] = test_ppl
            _wandb.run.summary["test_bpc"]        = test_bpc
        if val_mean_bits is not None:
            _wandb.run.summary["val_surprisal_bits"] = val_mean_bits
        if test_mean_bits is not None:
            _wandb.run.summary["test_surprisal_bits"] = test_mean_bits
        _wandb.run.summary["training_flops"]      = training_flops
        _wandb.run.summary["training_gpu_hours"]  = compute_tracker.get_gpu_hours("training")
        _wandb.finish()


if __name__ == "__main__":
    main()
