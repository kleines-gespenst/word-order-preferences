# Data & models

Scripts read data and models from a directory you set:

```bash
export WORD_ORDER_DATA=/path/to/word_order   # defaults to ./data
```

`artificial_languages/generation/base_grammar.gr` holds the grammar and its
vocabulary. Paths below are relative to `WORD_ORDER_DATA`.

## Artificial languages

| Path | Produced by |
|---|---|
| `word_order_data/` (permuted samples and splits) | `generation/generate_corpora.sh` |
| `vocab_final/`, `vocab_for_wuggy/`, `nltk_data/` | `vocab_generation/` |
| `word_order_models_*bpe15000_wordboundary*/` (checkpoints, surprisals, `surprisal_by_category/`) | `training/train_all_grammars.sh` |
| `surprisal_cross_config/` | `analysis/cross_config_surprisal_analysis.py` |

The word order universals script reads run metrics from Weights & Biases.


## Natural languages

| Path | Contents |
|---|---|
| `flores_clustering/flores200_word_orders.csv` | Word order labels per FLORES-200 language |
| `flores_clustering/flores200_encoding.csv` | Written by `typology/encode_flores_word_orders.py` |
| `flores_clustering/morph_complexity/` | Morphological complexity (subword MATTR), language resourcedness ([Joshi et al., 2020](https://aclanthology.org/2020.acl-main.560/)), and training data composition (OSCAR share) for the mixed-effects models |
| `results_natural_langs_flores/`, `results_natural_langs_pud/` | Goldfish BPEC evaluation outputs |
