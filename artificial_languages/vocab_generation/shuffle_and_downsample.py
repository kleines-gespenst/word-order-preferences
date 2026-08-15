import os
import random
from pathlib import Path

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

base_dir = str(DATA_ROOT / "vocab_for_wuggy") + '/'
output_dir = str(DATA_ROOT / "vocab_final") + '/'

Path(output_dir).mkdir(parents=True, exist_ok=True)

FILE_CONFIGS = [
    (f'{base_dir}pseudowords_nouns_singular.txt', 25_000//2),
    (f'{base_dir}pseudowords_verbs_transitive.txt', 8_000//4),
    (f'{base_dir}pseudowords_verbs_intransitive.txt', 8_000//4),
    (f'{base_dir}pseudowords_verbs_complementizer.txt', 4_000//4),
    (f'{base_dir}pseudowords_adjectives.txt', 5_000),
]

random.seed(42)

for filepath, n_samples in FILE_CONFIGS:
    with open(filepath, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    
    random.shuffle(words)
    
    if len(words) < n_samples:
        print(f"Warning: {filepath} has only {len(words)} words, need {n_samples}")
        sampled = words
    else:
        sampled = words[:n_samples]
    
    filename = Path(filepath).name
    output_path = Path(output_dir) / filename
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(sampled))
    
    print(f"{filename}: {len(sampled)} words saved")