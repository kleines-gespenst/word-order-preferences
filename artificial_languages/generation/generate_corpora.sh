#!/bin/bash
# Generate the artificial-language corpora from the committed grammar.
# Run from this directory; outputs go under $WORD_ORDER_DATA (default: ./data).
cd "$(dirname "$0")"
DATA="${WORD_ORDER_DATA:-data}"

python sample_sentences.py -g base_grammar.gr -n 950000 -O "$DATA/word_order_data" -b True
python permute_sentences.py
python make_splits.py -S "$DATA/word_order_data/permuted_samples_plain/" -O "$DATA/word_order_data/permuted_splits/"
