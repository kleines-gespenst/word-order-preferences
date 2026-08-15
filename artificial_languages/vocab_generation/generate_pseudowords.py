#!/usr/bin/env python3
"""
Generate English pseudowords with Wuggy from word lists
"""

import os
from pathlib import Path

from wuggy import WuggyGenerator
from uwotm8 import convert_american_to_british_spelling

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

# Initialize generator
g = WuggyGenerator()
g.download_language_plugin("orthographic_english", auto_download=True)
g.load("orthographic_english")

# ============================================================
# DEFINE SMALL WORD LISTS
# ============================================================

# Personal pronouns nominative (8)
PRONOUNS_NOMINATIVE = [
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'one'
]

# Prepositions (30)
PREPOSITIONS = [
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'under', 'over', 'beside', 'behind', 'near', 'without', 'within', 'against',
    'along', 'among', 'around', 'across'
]

# Complementizer words (3)
COMP_WORDS = [
    'that', 'which', 'whether'
]

# Coordinating conjunctions (2)
CC_WORDS = [
    'and', 'or'
]

# ============================================================
# FILE PATHS
# ============================================================

base_dir = str(DATA_ROOT / "vocab_for_wuggy") + '/'

# Input/output configurations: (input_file, output_file, num_pseudowords)
FILE_CONFIGS = [
    # Large word lists - 5 pseudowords each
    (f'{base_dir}nouns_singular.txt', f'{base_dir}pseudowords_nouns_singular.txt', 7),
    (f'{base_dir}verbs_transitive_for_pseudo.txt', f'{base_dir}pseudowords_verbs_transitive.txt', 7),
    (f'{base_dir}verbs_intransitive_for_pseudo.txt', f'{base_dir}pseudowords_verbs_intransitive.txt', 7),
    (f'{base_dir}verbs_complementizer_for_pseudo.txt', f'{base_dir}pseudowords_verbs_complementizer.txt', 5),
    (f'{base_dir}adjectives.txt', f'{base_dir}pseudowords_adjectives.txt', 5),
    # Small word lists - 1 pseudoword each
    (f'{base_dir}pronouns_nominative.txt', f'{base_dir}pseudowords_pronouns_nominative.txt', 1),
    (f'{base_dir}prepositions.txt', f'{base_dir}pseudowords_prepositions.txt', 1),
    (f'{base_dir}comp_words.txt', f'{base_dir}pseudowords_comp_words.txt', 1),
    (f'{base_dir}cc_words.txt', f'{base_dir}pseudowords_cc_words.txt', 1),
]

# ============================================================
# SAVE SMALL WORD LISTS
# ============================================================

def save_word_list(words, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(f"{word}\n")
    print(f" Saved {len(words)} words to {filepath}")

save_word_list(PRONOUNS_NOMINATIVE, f'{base_dir}pronouns_nominative.txt')
save_word_list(PREPOSITIONS, f'{base_dir}prepositions.txt')
save_word_list(COMP_WORDS, f'{base_dir}comp_words.txt')
save_word_list(CC_WORDS, f'{base_dir}cc_words.txt')

# ============================================================
# PSEUDOWORD GENERATION FUNCTION
# ============================================================

def generate_pseudowords(input_file, output_file, num_per_word=5):
    """Generate pseudowords for words in input file"""

    # Load seed words
    with open(input_file, 'r', encoding='utf-8') as f:
        seed_words = [line.strip() for line in f if line.strip()]

    pseudowords = []
    failed_words = []
    
    for i, seed in enumerate(seed_words):
        # Try British spelling if available
        british_seed = convert_american_to_british_spelling(seed)
        if not british_seed:
            british_seed = seed
        
        try:
            matches = list(g.generate_classic([british_seed]))
            
            if matches:
                # Get up to num_per_word pseudowords
                count = 0
                for match in matches:
                    if count >= num_per_word:
                        break
                    pseudoword = match["pseudoword"]
                    pseudowords.append(pseudoword)
                    count += 1
            else:
                failed_words.append(seed)

        except Exception as e:
            failed_words.append(seed)
            continue
    
    # Save pseudowords
    with open(output_file, 'w', encoding='utf-8') as f:
        for w in pseudowords:
            f.write(f"{w}\n")
    
    print(f"\n Generated {len(pseudowords)} pseudowords")
    print(f" Saved to {output_file}")
    if failed_words:
        print(f"Warning: Failed for {len(failed_words)} words")
    
    return len(pseudowords), len(failed_words)

# ============================================================
# GENERATE PSEUDOWORDS FOR ALL FILES
# ============================================================

results = []

for input_file, output_file, num_per_word in FILE_CONFIGS:
    try:
        generated, failed = generate_pseudowords(input_file, output_file, num_per_word)
        results.append((input_file, generated, failed))
    except FileNotFoundError:
        print(f"\nWarning: File not found: {input_file}, skipping...")
        results.append((input_file, 0, -1))

# ============================================================
# SUMMARY
# ============================================================

print("SUMMARY")

for input_file, generated, failed in results:
    filename = input_file.split('/')[-1]
    if failed == -1:
        print(f"  {filename}: FILE NOT FOUND")
    else:
        print(f"  {filename}: {generated} pseudowords generated, {failed} failed")