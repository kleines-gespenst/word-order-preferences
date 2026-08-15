import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

vocab_dir = DATA_ROOT / "vocab_final"
grammar_path = DATA_ROOT / "conlangs_generation/base_grammar_old_vocab.gr"
output_path = grammar_path.with_name('base_grammar.gr')

ZIPF_ALPHA = 1.0

# Read vocab files
def read_vocab(filename):
    path = vocab_dir / filename
    if path.exists():
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    return []

nouns = read_vocab('pseudowords_nouns_singular.txt')
adjectives = read_vocab('pseudowords_adjectives.txt')
verbs_trans = read_vocab('pseudowords_verbs_transitive.txt')
verbs_intrans = read_vocab('pseudowords_verbs_intransitive.txt')
verbs_comp = read_vocab('pseudowords_verbs_complementizer.txt')
prepositions = read_vocab('pseudowords_prepositions.txt')
pronouns = read_vocab('pseudowords_pronouns_nominative.txt')
cc_words = read_vocab('pseudowords_cc_words.txt')
comp_words = read_vocab('pseudowords_comp_words.txt')

# Morphology
def add_s(word):
    if word.endswith(('s', 'sh', 'ch', 'x', 'z')):
        return word + 'es'
    elif word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    return word + 's'

def add_ed(word):
    if word.endswith('e'):
        return word + 'd'
    elif word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
        return word[:-1] + 'ied'
    return word + 'ed'

def verb_past_p(v):
    return add_ed(v) + 'a'


def compute_zipf_weights(words, alpha=ZIPF_ALPHA):
    """Sort by length (shorter = more frequent), assign Zipf weights."""
    if not words:
        return {}
    
    sorted_words = sorted(words, key=lambda w: (len(w), w))
    word_to_rank = {w: i + 1 for i, w in enumerate(sorted_words)}
    
    # Zipf: weight = 1 / rank^alpha
    weights = {w: 1.0 / (word_to_rank[w] ** alpha) for w in words}
    return weights


def make_rules_zipf(category, words, weights):
    return '\n'.join(f'{weights.get(w, 1):.6f}\t{category}\t{w}' for w in words)


def make_derived_rules_zipf(category, base_words, derive_fn, base_weights):
    return '\n'.join(
        f'{base_weights.get(w, 1):.6f}\t{category}\t{derive_fn(w)}' 
        for w in base_words
    )


vocab_section = []

# Compute Zipf weights
noun_weights = compute_zipf_weights(nouns)
adj_weights = compute_zipf_weights(adjectives)
trans_weights = compute_zipf_weights(verbs_trans)
intrans_weights = compute_zipf_weights(verbs_intrans)
comp_weights = compute_zipf_weights(verbs_comp)

# Nouns
vocab_section.append(make_rules_zipf('Noun_S', nouns, noun_weights))
vocab_section.append(make_derived_rules_zipf('Noun_P', nouns, add_s, noun_weights))

# Adjectives
vocab_section.append(make_rules_zipf('Adj', adjectives, adj_weights))

# Intransitive verbs
vocab_section.append(make_derived_rules_zipf('IVerb_Past_S', verbs_intrans, add_ed, intrans_weights))
vocab_section.append(make_derived_rules_zipf('IVerb_Past_P', verbs_intrans, verb_past_p, intrans_weights))
vocab_section.append(make_derived_rules_zipf('IVerb_Pres_S', verbs_intrans, add_s, intrans_weights))
vocab_section.append(make_rules_zipf('IVerb_Pres_P', verbs_intrans, intrans_weights))

# Transitive verbs
vocab_section.append(make_derived_rules_zipf('TVerb_Past_S', verbs_trans, add_ed, trans_weights))
vocab_section.append(make_derived_rules_zipf('TVerb_Past_P', verbs_trans, verb_past_p, trans_weights))
vocab_section.append(make_derived_rules_zipf('TVerb_Pres_S', verbs_trans, add_s, trans_weights))
vocab_section.append(make_rules_zipf('TVerb_Pres_P', verbs_trans, trans_weights))

# Complement-taking verbs
vocab_section.append(make_derived_rules_zipf('Verb_Comp_Past_S', verbs_comp, add_ed, comp_weights))
vocab_section.append(make_derived_rules_zipf('Verb_Comp_Past_P', verbs_comp, verb_past_p, comp_weights))
vocab_section.append(make_derived_rules_zipf('Verb_Comp_Pres_S', verbs_comp, add_s, comp_weights))
vocab_section.append(make_rules_zipf('Verb_Comp_Pres_P', verbs_comp, comp_weights))

# Closed classes (uniform)
vocab_section.append('\n'.join(f'1\tPrep\t{w}' for w in prepositions))
vocab_section.append('\n'.join(f'1\tCC\t{w}' for w in cc_words))
vocab_section.append('\n'.join(f'1\tComp\t{w}' for w in comp_words))

# Pronouns
mid = len(pronouns) // 2
vocab_section.append('\n'.join(f'1\tPronoun_S\t{w}' for w in (pronouns[:mid] if mid > 0 else pronouns)))
vocab_section.append('\n'.join(f'1\tPronoun_P\t{w}' for w in (pronouns[mid:] if mid > 0 else pronouns)))

# Fixed
vocab_section.append('1\tRel\trel')
vocab_section.append('1\tSubj\tsubj')
vocab_section.append('1\tObj\tobj')

# Read and combine
with open(grammar_path) as f:
    content = f.read()

vocab_marker = '# Vocabulary'
if vocab_marker in content:
    grammar_rules = content[:content.index(vocab_marker)]
else:
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '\tNoun_S\t' in line:
            grammar_rules = '\n'.join(lines[:i])
            break

new_grammar = grammar_rules + vocab_marker + '\n' + '\n\n'.join(vocab_section)

with open(output_path, 'w') as f:
    f.write(new_grammar)

# Stats
print(f"Saved to {output_path}")
print(f"Nouns: {len(nouns)}, Adj: {len(adjectives)}")
print(f"Verbs: trans={len(verbs_trans)}, intrans={len(verbs_intrans)}, comp={len(verbs_comp)}")
