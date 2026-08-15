# Left-Branching Transformers Excel at Right-Branching Languages

This repository contains the experiments for the paper "Left-Branching
Transformers Excel at Right-Branching Languages: Data Shapes Word Order
Preferences in Language Models".

There are two sets of experiments:

- `artificial_languages/` trains one small GPT-2 on each of 192 controlled
  grammars (6 base orders by 32 branching configurations) and measures how word
  order affects learnability (perplexity, BPEC, per-constituent surprisal).
- `natural_languages/` asks the same question on FLORES-200 and PUD using
  monolingual Goldfish models, controlling for morphology, resourcedness,
  writing script, and training-data composition with mixed-effects models.

## Setup

```bash
pip install -r requirements.txt
export WORD_ORDER_DATA=/path/to/word_order   # data and models live here (default: ./data)
export PYTHONPATH=shared                      # some scripts import helpers from shared/
```

Data and models are not in the repository; see [docs/DATA.md](docs/DATA.md). The
Weights & Biases scripts read `WANDB_ENTITY` and `WANDB_PROJECT`; Hugging Face
downloads use `$HF_HOME`.

## Artificial languages

```bash
# vocabulary (optional: base_grammar.gr already contains the vocabulary)
python artificial_languages/vocab_generation/extract_english_words.py --pos noun

# corpora, then train (Slurm array over 192 grammars x 10 splits), then analyse
bash   artificial_languages/generation/generate_corpora.sh
sbatch artificial_languages/training/train_all_grammars.sh          # runs train_gpt2.py
python artificial_languages/analysis/cross_config_surprisal_analysis.py
python artificial_languages/analysis/argument_surprisal_by_word_order.py
python artificial_languages/analysis/word_order_universals.py       # reads from W&B
```

## Natural languages

```bash
python natural_languages/typology/encode_flores_word_orders.py
python natural_languages/evaluation/evaluate_goldfish_flores.py     # and evaluate_goldfish_pud.py
python natural_languages/analysis/compute_flores_bpec_stats.py      # SVO/SOV BPEC gap
python natural_languages/analysis/bootstrap_bpec_svo_sov_common.py  # and bootstrap_bpec_svo_sov_by_script.py
python natural_languages/analysis/compute_morphological_complexity.py
python natural_languages/analysis/mixedeffects.py                   # M1-M9 mixed-effects table
```

## Citation

The paper is under review; this entry will be updated on publication.

```bibtex
@unpublished{wordorderpreferences,
  title  = {Left-Branching Transformers Excel at Right-Branching Languages:
            Data Shapes Word Order Preferences in Language Models},
  author = {Arzt, Varvara and Hanbury, Allan and Blevins, Terra},
  year   = {2026},
  note   = {Under review}
}
```

## Acknowledgements

Allan Hanbury and Terra Blevins supervised this project, with most of the
supervision provided by Terra Blevins.

## Contact

If you have any questions, feel free to email me at
`varvara.arzt [at] tuwien [dot] ac [dot] at`.

See [LICENSE](LICENSE).
