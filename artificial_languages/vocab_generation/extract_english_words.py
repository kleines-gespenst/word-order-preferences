"""Open-class word-list generation for the artificial-language vocabulary.

Single entry point for all open-class POS:
  * nouns / adjectives -- most frequent English words, POS-filtered and
    lemmatised with spaCy (one lemma per source word).
  * verbs              -- additionally classified by transitivity
    (intransitive / transitive / complement-taking) using VerbNet + WordNet +
    Wiktionary, with British-spelling normalisation.

Generates noun, adjective, and verb word lists (dispatched by --pos).

Usage:
    python extract_english_words.py                 # noun + adj + verb
    python extract_english_words.py --pos noun adj  # just the simple lists
    python extract_english_words.py --pos verb      # just verbs
"""

import argparse
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

import spacy
from wordfreq import top_n_list

# pos -> (spaCy POS tag, target #lemmas, output filename)
POS_CONFIG = {
    "noun": ("NOUN", 10500, "nouns_singular.txt"),
    "adj":  ("ADJ",   5100, "adjectives.txt"),
}


def generate_pos_list(pos, out_dir, n_candidates=100000):
    """Simple frequency-ranked, POS-filtered, lemmatised list (nouns/adjectives)."""
    tag, target, fname = POS_CONFIG[pos]
    nlp = spacy.load("en_core_web_sm")
    common_words = top_n_list("en", n_candidates)
    lemmas, seen = [], set()
    for word in common_words:
        if len(lemmas) >= target:
            break
        for token in nlp(word):
            lemma = token.lemma_
            if token.pos_ == tag and lemma not in seen:
                lemmas.append(lemma)
                seen.add(lemma)
                break
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        for w in lemmas:
            f.write(f"{w}\n")
    print(f" {len(lemmas)} unique {pos} lemmas -> {path}")


def generate_verbs():
    """Build verb lists with transitivity classification (VerbNet/WordNet/Wiktionary)."""
    import nltk
    nltk.data.path.insert(0, str(DATA_ROOT / "nltk_data"))
    nltk.download('wordnet', quiet=True, download_dir=str(DATA_ROOT / "nltk_data"))

    import json
    import spacy
    from wordfreq import top_n_list
    from nltk.corpus import verbnet as vn
    from nltk.corpus import wordnet as wn
    from uwotm8 import convert_american_to_british_spelling

    nlp = spacy.load('en_core_web_sm')

    # ============================================================
    # SPELLING NORMALIZATION (to British)
    # ============================================================

    def normalize_to_british(word):
        """Normalize American to British spelling using uwotm8"""
        try:
            british = convert_american_to_british_spelling(word)
            return british if british else word
        except:
            return word

    # ============================================================
    # LOAD WIKTIONARY
    # ============================================================

    wiktionary_verbs = {}
    wiktionary_path = str(DATA_ROOT / "kaikki.org-dictionary-English-by-pos-verb.jsonl")

    try:
        with open(wiktionary_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                word = entry.get('word', '')
                wiktionary_verbs[word] = entry
        print(f" Loaded {len(wiktionary_verbs)} verbs from Wiktionary")
    except FileNotFoundError:
        print(f"Warning: Wiktionary file not found at {wiktionary_path}")
        wiktionary_verbs = {}

    # ============================================================
    # VERBNET FRAME DEFINITIONS
    # ============================================================

    INTRANSITIVE_FRAMES = {
        'Basic Intransitive',
        'Intransitive',
        'PP',
        'Locative Inversion',
        'There-insertion',
        'Middle Construction',
        'Apart Reciprocal Alternation Intransitive',
        'Simple Reciprocal Intransitive',
        'Together Reciprocal Alternation Intransitive',
        'Material/Product Alternation Intransitive',
        'Total Transformation Alternation Intransitive',
    }

    TRANSITIVE_FRAMES = {
        'Basic Transitive',
        'Transitive',
        'NP',
        'NP-NP',
        'NP-PP',
        'NP-PP-PP',
        'NP-ADJP',
        'NP-PART',
        'Dative',
        'Benefactive Alternation',
        'Conative',
        'Apart Reciprocal Alternation Transitive',
        'Simple Reciprocal Alternation Transitive',
        'Simple Reciprocal Transitive',
        'Together Reciprocal Alternation Transitive',
        'Material/Product Alternation Transitive',
        'Total Transformation Alternation Transitive',
        'Location Subject Alternation',
        'Unspecified Object',
    }

    COMPLEMENTIZER_FRAMES = {
        'S', 'THAT-S', 'NP-S', 'PP-S', 'PP-THAT-S', 'S-SUBJUNCT', 'PP-THAT-S-SUBJUNCT',
        'WH-S', 'WHAT-S', 'HOW-S', 'NP-WH-S', 'NP-WHAT-S', 'NP-HOW-S',
        'PP-WH-S', 'PP-WHAT-S', 'PP-HOW-S', 'P-WH-S', 'P-WHAT-S', 'PP-P-WH-S', 'PP-P-WHAT-S',
    }

    # ============================================================
    # WORDNET FRAME IDS
    # ============================================================

    WN_INTRANSITIVE_FRAMES = {1, 2, 3, 4, 6, 7, 22, 23}
    WN_TRANSITIVE_FRAMES = {8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 30, 31}
    WN_COMPLEMENTIZER_FRAMES = {26, 34}

    AUXILIARIES = {'be', 'have', 'do', 'will', 'would', 'could', 'should', 
                   'might', 'must', 'shall', 'can', 'may', "'s", "'ve", "'ll", "'d"}

    # ============================================================
    # CLASSIFICATION FUNCTIONS
    # ============================================================

    def classify_verb_verbnet(verb):
        """Classification based on VerbNet frames"""
        trans, intrans, comp = False, False, False
        
        try:
            for classid in vn.classids(verb):
                vnclass = vn.vnclass(classid)
                for frame in vnclass.findall('.//FRAME'):
                    desc = frame.find('DESCRIPTION')
                    if desc is None:
                        continue
                    f = desc.get('primary', '')
                    
                    if f in INTRANSITIVE_FRAMES:
                        intrans = True
                    if f in TRANSITIVE_FRAMES:
                        trans = True
                    if f in COMPLEMENTIZER_FRAMES:
                        comp = True
        except:
            pass
        
        return {'trans': trans, 'intrans': intrans, 'comp': comp}


    def classify_verb_wordnet(verb):
        """Classification based on WordNet frame IDs"""
        trans, intrans, comp = False, False, False
        
        try:
            for synset in wn.synsets(verb, pos='v'):
                frame_ids = set(synset.frame_ids())
                
                if frame_ids & WN_INTRANSITIVE_FRAMES:
                    intrans = True
                if frame_ids & WN_TRANSITIVE_FRAMES:
                    trans = True
                if frame_ids & WN_COMPLEMENTIZER_FRAMES:
                    comp = True
        except:
            pass
        
        return {'trans': trans, 'intrans': intrans, 'comp': comp}


    def classify_verb_wiktionary(verb):
        """Classification based on Wiktionary tags - only trans/intrans (no comp)"""
        trans, intrans = False, False
        
        if verb not in wiktionary_verbs:
            return {'trans': trans, 'intrans': intrans, 'comp': False}
        
        entry = wiktionary_verbs[verb]
        for sense in entry.get('senses', []):
            tags = [t.lower() for t in sense.get('tags', [])]
            
            if 'transitive' in tags:
                trans = True
            if 'intransitive' in tags:
                intrans = True
        
        # NO comp detection from Wiktionary - too error-prone
        return {'trans': trans, 'intrans': intrans, 'comp': False}


    def classify_verb_combined(verb):
        """Combine VerbNet, WordNet, and Wiktionary - try both spellings"""
        british = normalize_to_british(verb)
        variants = set([verb, british])
        
        source_vn = {'trans': False, 'intrans': False, 'comp': False}
        source_wn = {'trans': False, 'intrans': False, 'comp': False}
        source_wikt = {'trans': False, 'intrans': False, 'comp': False}
        
        for variant in variants:
            vn_cls = classify_verb_verbnet(variant)
            wn_cls = classify_verb_wordnet(variant)
            wikt_cls = classify_verb_wiktionary(variant)
            
            for key in ['trans', 'intrans', 'comp']:
                source_vn[key] = source_vn[key] or vn_cls[key]
                source_wn[key] = source_wn[key] or wn_cls[key]
            
            # Wiktionary only for trans/intrans
            source_wikt['trans'] = source_wikt['trans'] or wikt_cls['trans']
            source_wikt['intrans'] = source_wikt['intrans'] or wikt_cls['intrans']
        
        return {
            'trans': source_vn['trans'] or source_wn['trans'] or source_wikt['trans'],
            'intrans': source_vn['intrans'] or source_wn['intrans'] or source_wikt['intrans'],
            'comp': source_vn['comp'] or source_wn['comp'],  # NO Wiktionary for comp
            'source_vn': source_vn,
            'source_wn': source_wn,
            'source_wikt': source_wikt,
        }


    # ============================================================
    # EXTRACT VERBS FROM WORDFREQ
    # ============================================================

    common_words = top_n_list('en', 1000000)

    transitive_verbs = []
    intransitive_verbs = []
    complementizer_verbs = []

    trans_from_vn = []
    trans_from_wn = []
    trans_from_wikt = []
    intrans_from_vn = []
    intrans_from_wn = []
    intrans_from_wikt = []
    comp_from_vn = []
    comp_from_wn = []

    seen_lemmas = set()
    seen_normalized = set()
    TARGET = 10000

    processed = 0
    for word in common_words:
        if (len(transitive_verbs) >= TARGET and 
            len(intransitive_verbs) >= TARGET and 
            len(complementizer_verbs) >= TARGET):
            break
        
        doc = nlp(word)
        for token in doc:
            lemma = token.lemma_
            lemma_british = normalize_to_british(lemma)
            
            if lemma in seen_lemmas or lemma_british in seen_normalized:
                break
                
            if token.pos_ == 'VERB' and lemma not in AUXILIARIES and len(lemma) > 1:
                seen_lemmas.add(lemma)
                seen_normalized.add(lemma_british)
                
                cls = classify_verb_combined(lemma)
                store_lemma = lemma_british  # Store British spelling
                
                # Transitive
                if cls['trans'] and len(transitive_verbs) < TARGET:
                    if store_lemma not in transitive_verbs:
                        transitive_verbs.append(store_lemma)
                        if cls['source_vn']['trans'] and len(trans_from_vn) < 30:
                            trans_from_vn.append(store_lemma)
                        elif cls['source_wn']['trans'] and not cls['source_vn']['trans'] and len(trans_from_wn) < 30:
                            trans_from_wn.append(store_lemma)
                        elif cls['source_wikt']['trans'] and not cls['source_vn']['trans'] and not cls['source_wn']['trans'] and len(trans_from_wikt) < 30:
                            trans_from_wikt.append(store_lemma)
                
                # Intransitive
                if cls['intrans'] and len(intransitive_verbs) < TARGET:
                    if store_lemma not in intransitive_verbs:
                        intransitive_verbs.append(store_lemma)
                        if cls['source_vn']['intrans'] and len(intrans_from_vn) < 30:
                            intrans_from_vn.append(store_lemma)
                        elif cls['source_wn']['intrans'] and not cls['source_vn']['intrans'] and len(intrans_from_wn) < 30:
                            intrans_from_wn.append(store_lemma)
                        elif cls['source_wikt']['intrans'] and not cls['source_vn']['intrans'] and not cls['source_wn']['intrans'] and len(intrans_from_wikt) < 30:
                            intrans_from_wikt.append(store_lemma)
                
                # Complementizer (only VerbNet + WordNet)
                if cls['comp'] and len(complementizer_verbs) < TARGET:
                    if store_lemma not in complementizer_verbs:
                        complementizer_verbs.append(store_lemma)
                        if cls['source_vn']['comp'] and len(comp_from_vn) < 30:
                            comp_from_vn.append(store_lemma)
                        elif cls['source_wn']['comp'] and not cls['source_vn']['comp'] and len(comp_from_wn) < 30:
                            comp_from_wn.append(store_lemma)
                
                break
        
        processed += 1

    # ============================================================
    # COMPUTE UNIQUE VERBS
    # ============================================================

    trans_set = set(transitive_verbs)
    intrans_set = set(intransitive_verbs)
    comp_set = set(complementizer_verbs)

    only_transitive = trans_set - intrans_set - comp_set
    only_intransitive = intrans_set - trans_set - comp_set
    only_complementizer = comp_set - trans_set - intrans_set

    trans_and_intrans = trans_set & intrans_set - comp_set
    trans_and_comp = trans_set & comp_set - intrans_set
    intrans_and_comp = intrans_set & comp_set - trans_set
    all_three = trans_set & intrans_set & comp_set

    # ============================================================
    # RESULTS
    # ============================================================

    print("RESULTS")
    print(f"Transitive verbs:     {len(transitive_verbs)}")
    print(f"Intransitive verbs:   {len(intransitive_verbs)}")
    print(f"Complementizer verbs: {len(complementizer_verbs)}")

    print("UNIQUE VERBS (exclusive to one category)")
    print(f"Only transitive:     {len(only_transitive)}")
    print(f"Only intransitive:   {len(only_intransitive)}")
    print(f"Only complementizer: {len(only_complementizer)}")

    # ============================================================
    # SAVE TO FILES
    # ============================================================

    output_dir = str(DATA_ROOT / "vocab_for_wuggy") + '/'

    with open(f'{output_dir}verbs_transitive.txt', 'w', encoding='utf-8') as f:
        for verb in transitive_verbs:
            f.write(f"{verb}\n")

    with open(f'{output_dir}verbs_intransitive.txt', 'w', encoding='utf-8') as f:
        for verb in intransitive_verbs:
            f.write(f"{verb}\n")

    with open(f'{output_dir}verbs_complementizer.txt', 'w', encoding='utf-8') as f:
        for verb in complementizer_verbs:
            f.write(f"{verb}\n")

    with open(f'{output_dir}verbs_only_transitive.txt', 'w', encoding='utf-8') as f:
        for verb in sorted(only_transitive):
            f.write(f"{verb}\n")

    with open(f'{output_dir}verbs_only_intransitive.txt', 'w', encoding='utf-8') as f:
        for verb in sorted(only_intransitive):
            f.write(f"{verb}\n")

    with open(f'{output_dir}verbs_only_complementizer.txt', 'w', encoding='utf-8') as f:
        for verb in sorted(only_complementizer):
            f.write(f"{verb}\n")

    # Save statistics
    with open(f'{output_dir}verb_classification_stats.txt', 'w', encoding='utf-8') as f:
        f.write("VERB CLASSIFICATION STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write("Sources:\n")
        f.write("  - Transitive/Intransitive: VerbNet + WordNet + Wiktionary\n")
        f.write("  - Complementizer: VerbNet + WordNet only (Wiktionary too error-prone)\n")
        f.write("Spelling: Normalised to British English (uwotm8)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("TOTAL COUNTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Transitive verbs:     {len(transitive_verbs)}\n")
        f.write(f"Intransitive verbs:   {len(intransitive_verbs)}\n")
        f.write(f"Complementizer verbs: {len(complementizer_verbs)}\n\n")
        
        f.write("UNIQUE VERBS (exclusive to one category)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Only transitive:     {len(only_transitive)}\n")
        f.write(f"Only intransitive:   {len(only_intransitive)}\n")
        f.write(f"Only complementizer: {len(only_complementizer)}\n\n")
        
        f.write("OVERLAPS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Trans + Intrans (not comp):  {len(trans_and_intrans)}\n")
        f.write(f"Trans + Comp (not intrans):  {len(trans_and_comp)}\n")
        f.write(f"Intrans + Comp (not trans):  {len(intrans_and_comp)}\n")
        f.write(f"All three categories:        {len(all_three)}\n\n")
        
        f.write("EXAMPLES BY SOURCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"\nTransitive from VerbNet: {trans_from_vn}\n")
        f.write(f"Transitive from WordNet only: {trans_from_wn}\n")
        f.write(f"Transitive from Wiktionary only: {trans_from_wikt}\n")
        f.write(f"\nIntransitive from VerbNet: {intrans_from_vn}\n")
        f.write(f"Intransitive from WordNet only: {intrans_from_wn}\n")
        f.write(f"Intransitive from Wiktionary only: {intrans_from_wikt}\n")
        f.write(f"\nComplementizer from VerbNet: {comp_from_vn}\n")
        f.write(f"Complementizer from WordNet only: {comp_from_wn}\n\n")
        
        f.write("UNIQUE VERB EXAMPLES (first 50 each)\n")
        f.write("-" * 40 + "\n")
        f.write(f"\nOnly transitive:\n{sorted(only_transitive)[:50]}\n")
        f.write(f"\nOnly intransitive:\n{sorted(only_intransitive)[:50]}\n")
        f.write(f"\nOnly complementizer:\n{sorted(only_complementizer)[:50]}\n")

    print("FILES SAVED")
    print(f"  1. {output_dir}verbs_transitive.txt ({len(transitive_verbs)} verbs)")
    print(f"  2. {output_dir}verbs_intransitive.txt ({len(intransitive_verbs)} verbs)")
    print(f"  3. {output_dir}verbs_complementizer.txt ({len(complementizer_verbs)} verbs)")
    print(f"  4. {output_dir}verbs_only_transitive.txt ({len(only_transitive)} verbs)")
    print(f"  5. {output_dir}verbs_only_intransitive.txt ({len(only_intransitive)} verbs)")
    print(f"  6. {output_dir}verbs_only_complementizer.txt ({len(only_complementizer)} verbs)")
    print(f"  7. {output_dir}verb_classification_stats.txt")

    if len(transitive_verbs) < TARGET:
        print(f"\nWarning: Warning: Only found {len(transitive_verbs)} transitive verbs (target: {TARGET})")
    if len(intransitive_verbs) < TARGET:
        print(f"Warning: Warning: Only found {len(intransitive_verbs)} intransitive verbs (target: {TARGET})")
    if len(complementizer_verbs) < TARGET:
        print(f"Warning: Warning: Only found {len(complementizer_verbs)} complementizer verbs (target: {TARGET})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", nargs="+", choices=["noun", "adj", "verb"],
                    default=["noun", "adj", "verb"], help="which lists to build")
    ap.add_argument("--out-dir",
                    default=str(DATA_ROOT / "vocab_for_wuggy"),
                    help="output dir for noun/adj lists (verbs use their own path)")
    args = ap.parse_args()
    for pos in args.pos:
        if pos == "verb":
            generate_verbs()
        else:
            generate_pos_list(pos, args.out_dir)


if __name__ == "__main__":
    main()
