#!/usr/bin/env python3
"""
Per-argument surprisal analysis across word orders.

Reads pre-computed BPE-token surprisal JSON files (5MB and 10MB models),
maps tokens back to words via the SP model, annotates with syntactic roles
(Subj / Obj / V) from the bracketed sample files, and aggregates by word order.

Research question: Is the subject harder to predict in SVO/SOV langs than in
OSV/OVS/VSO/VOS, and similarly for other arguments?

Inputs (all relative to $WORD_ORDER_DATA)
------
  Surprisal JSONs:
    word_order_models_{5MB,10MB}_bpe15000_wordboundary_ep20_results/
      gpt2-surprisals/{grammar}/{split}.tst.wordboundary.surprisal.json
  SP models (split 0 only, used for all splits):
    word_order_models_{5MB,10MB}_bpe15000_wordboundary_ep20/
      gpt2-checkpoints/{grammar}/0/bpe15000_wordboundary.model
  Test sentences:
    word_order_data/permuted_splits/{grammar}/{split}.tst
  Bracketed sample trees (for role annotation):
    word_order_data/permuted_samples/sample_{grammar}.txt

Output (relative to $WORD_ORDER_DATA)
------
  word_order_models_5MB_bpe15000_wordboundary_ep20_results/
    gpt2-results/visualisations/argument_surprisal/
"""

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

DATASETS = {
    '5MB': {
        'surprisals': DATA_ROOT / "word_order_models_5MB_bpe15000_wordboundary_ep20_results/gpt2-surprisals",
        'models':     DATA_ROOT / "word_order_models_5MB_bpe15000_wordboundary_ep20/gpt2-checkpoints",
    },
    '10MB': {
        'surprisals': DATA_ROOT / "word_order_models_10MB_bpe15000_wordboundary_ep20_results/gpt2-surprisals",
        'models':     DATA_ROOT / "word_order_models_10MB_bpe15000_wordboundary_ep20/gpt2-checkpoints",
    },
}
SPLITS_DIR  = DATA_ROOT / "word_order_data/permuted_splits"
SAMPLES_DIR = DATA_ROOT / "word_order_data/permuted_samples"
OUT_DIR     = DATA_ROOT / "word_order_models_5MB_bpe15000_wordboundary_ep20_results/gpt2-results/visualisations/argument_surprisal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_ORDERS = ['SVO', 'SOV', 'VSO', 'VOS', 'OVS', 'OSV']
N_SPLITS    = 10
CHUNK_SIZE  = 512

# ── Role parsing (from bracketed sample trees) ────────────────────────────────

def _detailed_role(immediate_lhs: str, phrase_role: str, in_relative: bool) -> str:
    s = immediate_lhs.strip()
    if s == 'Subj':   return 'Subj_marker'
    if s == 'Obj':    return 'Obj_marker'
    if s == 'Rel':    return 'Rel_marker'
    if s == 'Comp':   return 'Comp_marker'
    if s == 'CC':     return 'CC_marker'
    if s in ('Noun_S', 'Noun_P'):
        suffix = 'S' if s == 'Noun_S' else 'P'
        pre = 'Rel_' if in_relative else ''
        if phrase_role == 'Subj': return f'{pre}Subj_Noun_{suffix}'
        if phrase_role == 'Obj':  return f'{pre}Obj_Noun_{suffix}'
        return f'Rel_Noun_{suffix}' if in_relative else s
    if s in ('Pronoun_S', 'Pronoun_P'):
        suffix = 'S' if s == 'Pronoun_S' else 'P'
        pre = 'Rel_' if in_relative else ''
        if phrase_role == 'Subj': return f'{pre}Subj_Pronoun_{suffix}'
        if phrase_role == 'Obj':  return f'{pre}Obj_Pronoun_{suffix}'
        return f'Rel_Pronoun_{suffix}' if in_relative else s
    if s.startswith('IVerb_') or s.startswith('TVerb_'):
        return ('Rel_V_' if in_relative else 'V_') + s
    if s.startswith('Verb_Comp'):
        return ('Rel_V_Comp_' if in_relative else 'V_Comp_') + s.replace('Verb_Comp_', '')
    if s == 'Adj':  return 'Adj'
    if s == 'Prep': return 'Prep'
    return s or 'Other'


def parse_bracketed_tree(line: str):
    """Return (words, roles) for a bracketed tree string."""
    line = line.strip()
    if not line.startswith('(') or not line.endswith(')'):
        return None, None
    words, roles = [], []

    def phrase_role_of(sym):
        if not sym: return None
        if sym.startswith('NP_Subj'): return 'Subj'
        if sym.startswith('NP_Obj'):  return 'Obj'
        if sym.startswith('IVerb_') or sym.startswith('TVerb_') or sym.startswith('Verb_Comp'): return 'V'
        return None

    def in_rel(sym):
        return bool(sym and sym.startswith('VP_Rel'))

    def parse(s, i, phrase_role, in_relative):
        if i >= len(s) or s[i] == ')': return i
        j = i
        while j < len(s) and s[j] not in ' (\n': j += 1
        sym = s[i:j].strip()
        pr = phrase_role_of(sym)
        if pr is not None: phrase_role = pr
        rel = in_rel(sym) or in_relative
        while j < len(s) and s[j] in ' \n': j += 1
        if j >= len(s): return j
        if s[j] == '(':
            while j < len(s) and s[j] == '(':
                j = parse(s, j + 1, phrase_role, rel)
                while j < len(s) and s[j] in ' \n': j += 1
            if j < len(s) and s[j] == ')': j += 1
            return j
        k = j
        while k < len(s) and s[k] != ')': k += 1
        word = s[j:k].strip()
        words.append(word)
        roles.append(_detailed_role(sym, phrase_role, in_relative, ))
        return k + 1

    i = 1
    while i < len(line) and line[i] in ' \n': i += 1
    if i < len(line): parse(line, i, None, False)
    return (words, roles) if words else (None, None)


def coarse_role(role: str) -> str:
    if 'marker' in role:                                          return 'marker'
    if 'Subj_Noun' in role or 'Subj_Pronoun' in role:
        return 'Rel_Subj' if role.startswith('Rel_') else 'Subj'
    if 'Obj_Noun' in role or 'Obj_Pronoun' in role:
        return 'Rel_Obj' if role.startswith('Rel_') else 'Obj'
    if role.startswith('Rel_V_'):                                 return 'Rel_V'
    if role.startswith(('V_', 'V_Comp')):                        return 'V'
    if role == 'Adj':   return 'Adj'
    if role == 'Prep':  return 'Prep'
    return 'other'


def load_sample_roles(grammar: str) -> dict:
    """Load sample file → {normalized_yield: [roles_per_word]}."""
    path = SAMPLES_DIR / f'sample_{grammar}.txt'
    if not path.exists():
        return {}
    mapping = {}
    buf = ''
    with open(path, encoding='utf-8') as f:
        for line in f:
            buf += line
            if buf.count('(') == buf.count(')') and buf.strip():
                w, r = parse_bracketed_tree(buf.strip())
                if w and r and len(w) == len(r):
                    key = ' '.join(w)
                    mapping[key] = r
                buf = ''
    if buf.strip():
        w, r = parse_bracketed_tree(buf.strip())
        if w and r and len(w) == len(r):
            mapping[' '.join(w)] = r
    return mapping


# ── Token → word mapping via SP model ─────────────────────────────────────────

def build_token_span(sentences: list, sp) -> tuple:
    """
    Tokenize sentences with EOS between them.
    Returns:
      all_ids    : list of int token ids
      span       : list of (sent_idx, word_idx) per position; word_idx=-1 for EOS
      word_bounds: list of (sent_idx, word_idx, g_start, g_end_excl)
    """
    eos_id = sp.eos_id()
    all_ids, span = [], []
    word_bounds = []  # (si, wi, g_start, g_end)

    for si, sent in enumerate(sentences):
        words = sent.strip().split()
        for wi, w in enumerate(words):
            ids = sp.Encode(w)
            if not ids:
                ids = [sp.unk_id()]
            g_start = len(all_ids)
            for tid in ids:
                all_ids.append(tid)
                span.append((si, wi))
            word_bounds.append((si, wi, g_start, len(all_ids)))
        all_ids.append(eos_id)
        span.append((si, -1))

    return all_ids, span, word_bounds


def get_surprisals_from_json(json_data: dict, n_tokens: int) -> list:
    """
    Reconstruct per-position surprisal from JSON chunks.
    Returns list of length n_tokens: float or nan.
    surprisals_flat[g] = surprisal of token g (nan if first in chunk or unavailable).
    """
    chunks = json_data['surprisals']
    result = [math.nan] * n_tokens

    for c_idx, chunk in enumerate(chunks):
        chunk_start = c_idx * CHUNK_SIZE
        # chunk[i] = surprisal of token at global position chunk_start + i + 1
        for i, val in enumerate(chunk):
            g = chunk_start + i + 1
            if g < n_tokens:
                result[g] = float(val)

    return result


def words_to_surprisals(word_bounds, token_surprisals):
    """
    Aggregate per-token surprisals to per-word by summing subword token bits.
    Returns dict (si, wi) -> surprisal (nan if any token in word is nan / first in chunk).
    """
    out = {}
    for si, wi, g_start, g_end in word_bounds:
        bits = []
        for g in range(g_start, g_end):
            v = token_surprisals[g]
            if math.isfinite(v):
                bits.append(v)
        if bits:
            out[(si, wi)] = sum(bits)
        # if all nans (e.g. first token of chunk), skip
    return out


# ── Process one grammar/dataset ───────────────────────────────────────────────

def process_grammar(grammar: str, dataset_name: str, sp, roles_map: dict,
                    surprisals_dir: Path) -> list:
    """
    Process all 10 splits for a grammar.
    Returns list of (role_group, surprisal_bits).
    """
    wo = grammar[:3]
    records = []

    for split in range(N_SPLITS):
        json_path = surprisals_dir / grammar / f'{split}.tst.wordboundary.surprisal.json'
        tst_path  = SPLITS_DIR / grammar / f'{split}.tst'
        if not json_path.exists() or not tst_path.exists():
            continue

        with open(tst_path, encoding='utf-8') as f:
            sentences = [l.strip() for l in f if l.strip()]
        with open(json_path, encoding='utf-8') as f:
            jdata = json.load(f)

        all_ids, span, word_bounds = build_token_span(sentences, sp)
        token_surprisals = get_surprisals_from_json(jdata, len(all_ids))
        word_surp = words_to_surprisals(word_bounds, token_surprisals)

        # Match sentences to roles
        for si, sent in enumerate(sentences):
            key = sent.strip()
            # strip trailing " ." if present
            if key.endswith(' .'):
                key = key[:-2]
            roles = roles_map.get(key)
            if roles is None:
                continue
            words = sent.strip().split()
            for wi, (w, role) in enumerate(zip(words, roles)):
                bits = word_surp.get((si, wi))
                if bits is None or not math.isfinite(bits):
                    continue
                rg = coarse_role(role)
                records.append({'word_order': wo, 'role': rg, 'bits': bits})

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

all_records = {}  # dataset_name -> list of dicts

for dataset_name, paths in DATASETS.items():
    records = []
    import sentencepiece as spm

    for wo in WORD_ORDERS:
        # Collect all grammars for this word order
        grammars = sorted(p.name for p in paths['surprisals'].iterdir()
                          if p.is_dir() and p.name.startswith(wo + '_'))
        if not grammars:
            print(f'  No grammars found for {wo}')
            continue

        for grammar in grammars:
            sp_path = paths['models'] / grammar / '0' / 'bpe15000_wordboundary.model'
            if not sp_path.exists():
                continue
            sp = spm.SentencePieceProcessor()
            sp.Load(str(sp_path))

            roles_map = load_sample_roles(grammar)
            if not roles_map:
                print(f'  WARNING: no sample roles for {grammar}')
                continue

            gr = process_grammar(grammar, dataset_name, sp, roles_map, paths['surprisals'])
            records.extend(gr)
            sys.stdout.write(f'  {grammar}: {len(gr)} token records\n')
            sys.stdout.flush()

    all_records[dataset_name] = records
    print(f'  Total records for {dataset_name}: {len(records)}')


# ── Aggregate ─────────────────────────────────────────────────────────────────

FOCUS_ROLES = ['Subj', 'Obj', 'V', 'marker', 'Adj', 'Prep']
ROLE_LABELS  = {'Subj': 'Subject', 'Obj': 'Object', 'V': 'Main Verb',
                'marker': 'Case Marker', 'Adj': 'Adjective', 'Prep': 'Adposition'}


def aggregate(records):
    """Returns DataFrame with columns: word_order, role, mean, std, n."""
    rows = []
    from collections import defaultdict
    by = defaultdict(list)
    for r in records:
        by[(r['word_order'], r['role'])].append(r['bits'])
    for (wo, role), vals in by.items():
        rows.append({'word_order': wo, 'role': role,
                     'mean': np.mean(vals), 'std': np.std(vals),
                     'se': np.std(vals) / math.sqrt(len(vals)),
                     'n': len(vals)})
    return pd.DataFrame(rows)


agg_5mb  = aggregate(all_records['5MB'])
agg_10mb = aggregate(all_records['10MB'])

agg_5mb.to_csv(OUT_DIR  / 'role_surprisal_by_word_order_5MB.csv',  index=False)
agg_10mb.to_csv(OUT_DIR / 'role_surprisal_by_word_order_10MB.csv', index=False)
print('\nSaved CSVs')


# ── Plot 1: Subj / Obj / V surprisal by word order (5MB vs 10MB rows) ─────────

WO_ORDER = ['SVO', 'SOV', 'VSO', 'VOS', 'OVS', 'OSV']
S_BEFORE = {'SVO', 'SOV', 'VSO'}

fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharey='row')

for row_idx, role in enumerate(['Subj', 'Obj', 'V']):
    for col_idx, (dname, agg_df) in enumerate([('5MB', agg_5mb), ('10MB', agg_10mb)]):
        ax = axes[row_idx][col_idx]
        sub = agg_df[agg_df['role'] == role].set_index('word_order').reindex(WO_ORDER)
        means = sub['mean'].values
        ses   = sub['se'].values
        colors = ['#2E86AB' if wo in S_BEFORE else '#E63946' for wo in WO_ORDER]
        x = np.arange(len(WO_ORDER))
        ax.bar(x, means, color=colors, edgecolor='black', linewidth=1.1,
               alpha=0.85, yerr=ses, capsize=5,
               error_kw={'linewidth': 1.3, 'ecolor': 'gray'})
        for i, (m, se) in enumerate(zip(means, ses)):
            if np.isfinite(m):
                ax.text(i, m + se + 0.05, f'{m:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'{ROLE_LABELS[role]} surprisal – {dname}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(WO_ORDER, fontsize=10)
        ax.set_ylabel('Mean surprisal (bits)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)

legend_elements = [mpatches.Patch(facecolor='#2E86AB', edgecolor='black', label='S before O (SVO/SOV/VSO)'),
                   mpatches.Patch(facecolor='#E63946', edgecolor='black', label='S after O (VOS/OVS/OSV)')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Per-argument surprisal by word order (5MB vs 10MB GPT-2 models)',
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig(OUT_DIR / 'argument_surprisal_by_word_order.png', dpi=150, bbox_inches='tight')
plt.savefig(OUT_DIR / 'argument_surprisal_by_word_order.svg', bbox_inches='tight')
plt.close()
print('Saved argument_surprisal_by_word_order.png')


# ── Plot 2: Heatmap of role surprisal by word order ───────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (dname, agg_df) in zip(axes, [('5MB', agg_5mb), ('10MB', agg_10mb)]):
    pivot = agg_df[agg_df['role'].isin(FOCUS_ROLES)].pivot(
        index='role', columns='word_order', values='mean')
    pivot = pivot.reindex(index=FOCUS_ROLES, columns=WO_ORDER)
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(WO_ORDER)))
    ax.set_xticklabels(WO_ORDER, fontsize=11)
    ax.set_yticks(range(len(FOCUS_ROLES)))
    ax.set_yticklabels([ROLE_LABELS.get(r, r) for r in FOCUS_ROLES], fontsize=11)
    ax.set_title(f'Mean surprisal by role × word order – {dname}', fontsize=12)
    plt.colorbar(im, ax=ax, label='bits')
    # Annotate cells
    for i in range(len(FOCUS_ROLES)):
        for j in range(len(WO_ORDER)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if v > pivot.values[np.isfinite(pivot.values)].mean() else 'black')

fig.suptitle('Surprisal heatmap: syntactic role × word order', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'role_x_word_order_heatmap.png', dpi=150, bbox_inches='tight')
plt.savefig(OUT_DIR / 'role_x_word_order_heatmap.svg', bbox_inches='tight')
plt.close()
print('Saved role_x_word_order_heatmap.png')


# ── Plot 3: Subj vs Obj gap by word order ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (dname, agg_df) in zip(axes, [('5MB', agg_5mb), ('10MB', agg_10mb)]):
    subj = agg_df[agg_df['role'] == 'Subj'].set_index('word_order').reindex(WO_ORDER)
    obj  = agg_df[agg_df['role'] == 'Obj'].set_index('word_order').reindex(WO_ORDER)
    x = np.arange(len(WO_ORDER))
    w = 0.35
    ax.bar(x - w/2, subj['mean'], w, label='Subject', color='#4ECDC4',
           edgecolor='black', linewidth=1.1, alpha=0.85,
           yerr=subj['se'], capsize=5, error_kw={'ecolor': 'gray', 'linewidth': 1.3})
    ax.bar(x + w/2, obj['mean'], w, label='Object', color='#FF6B6B',
           edgecolor='black', linewidth=1.1, alpha=0.85,
           yerr=obj['se'], capsize=5, error_kw={'ecolor': 'gray', 'linewidth': 1.3})
    ax.set_xticks(x)
    ax.set_xticklabels(WO_ORDER, fontsize=11)
    ax.set_ylabel('Mean surprisal (bits)', fontsize=11)
    ax.set_title(f'Subject vs Object surprisal – {dname}', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

fig.suptitle('Subject vs Object surprisal across word orders', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'subj_vs_obj_by_word_order.png', dpi=150, bbox_inches='tight')
plt.savefig(OUT_DIR / 'subj_vs_obj_by_word_order.svg', bbox_inches='tight')
plt.close()
print('Saved subj_vs_obj_by_word_order.png')


# ── Plot 4: Effect sizes — does S position predict S surprisal? ───────────────
# Compare: "S appears early" (VSO) vs "S appears late" (OVS) word orders
# For each role: bar of mean across S-early vs S-late orders

S_EARLY  = ['VSO', 'SVO']   # S in position 1 or 2
S_MIDDLE = ['SOV', 'VOS']   # S in position 2 or 3
S_LATE   = ['OVS', 'OSV']   # S in position 3

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (dname, agg_df) in zip(axes, [('5MB', agg_5mb), ('10MB', agg_10mb)]):
    groups = {'S early\n(VSO,SVO)': S_EARLY, 'S mid\n(SOV,VOS)': S_MIDDLE, 'S late\n(OVS,OSV)': S_LATE}
    roles_to_plot = ['Subj', 'Obj', 'V']
    role_colors = {'Subj': '#4ECDC4', 'Obj': '#FF6B6B', 'V': '#FFD93D'}
    x = np.arange(len(groups))
    n_roles = len(roles_to_plot)
    w = 0.25

    for ri, role in enumerate(roles_to_plot):
        means, ses = [], []
        for group_wos in groups.values():
            sub = agg_df[(agg_df['role'] == role) & (agg_df['word_order'].isin(group_wos))]
            if sub.empty:
                means.append(np.nan); ses.append(np.nan)
            else:
                means.append(sub['mean'].mean())
                ses.append(sub['se'].mean())
        offset = (ri - n_roles / 2 + 0.5) * w
        ax.bar(x + offset, means, w, label=ROLE_LABELS[role],
               color=role_colors[role], edgecolor='black', linewidth=1.1,
               alpha=0.85, yerr=ses, capsize=4, error_kw={'ecolor': 'gray'})

    ax.set_xticks(x)
    ax.set_xticklabels(list(groups.keys()), fontsize=11)
    ax.set_ylabel('Mean surprisal (bits)', fontsize=11)
    ax.set_title(f'Role surprisal by subject position group – {dname}', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

fig.suptitle('Does subject position predict argument surprisal?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'surprisal_by_subject_position_group.png', dpi=150, bbox_inches='tight')
plt.savefig(OUT_DIR / 'surprisal_by_subject_position_group.svg', bbox_inches='tight')
plt.close()
print('Saved surprisal_by_subject_position_group.png')


# ── Report ────────────────────────────────────────────────────────────────────

report = []
def R(line=''): report.append(line)

R('# Argument Surprisal by Word Order')
R()
R('## Research question')
R('Does the syntactic role of a token (Subject, Object, Verb) interact with')
R('word order in terms of predictability (surprisal)?')
R('Specifically: is the subject harder to predict in SVO/SOV languages than')
R('in OSV/OVS/VSO/VOS, and similarly for other arguments?')
R()
R('## Method')
R('- BPE-token surprisal from pre-computed JSON files (5MB / 10MB GPT-2 models)')
R('- Tokens mapped back to words via SentencePiece model; word surprisal = sum of subword bits')
R('- Syntactic roles (Subj/Obj/V/marker/Adj/Prep) from bracketed sample parse trees')
R('- Aggregated across all 32 switch configurations per word order and all 10 test splits')
R()

for dname, agg_df in [('5MB', agg_5mb), ('10MB', agg_10mb)]:
    R(f'## Results — {dname}')
    R()
    R('| Role | ' + ' | '.join(WO_ORDER) + ' |')
    R('|------| ' + ' | '.join(['---'] * len(WO_ORDER)) + ' |')
    for role in FOCUS_ROLES:
        sub = agg_df[agg_df['role'] == role].set_index('word_order').reindex(WO_ORDER)
        cells = []
        for wo in WO_ORDER:
            row = sub.loc[wo] if wo in sub.index else None
            if row is not None and np.isfinite(row['mean']):
                cells.append(f"{row['mean']:.2f}±{row['se']:.2f}")
            else:
                cells.append('—')
        R(f'| {ROLE_LABELS.get(role, role)} | ' + ' | '.join(cells) + ' |')
    R()

    # Key findings
    R('### Key findings')
    subj = agg_df[agg_df['role'] == 'Subj'].set_index('word_order').reindex(WO_ORDER)
    obj  = agg_df[agg_df['role'] == 'Obj'].set_index('word_order').reindex(WO_ORDER)
    v    = agg_df[agg_df['role'] == 'V'].set_index('word_order').reindex(WO_ORDER)

    # Subject surprisal: is it lower in S-early orders?
    s_early_subj = subj.loc[['VSO', 'SVO'], 'mean'].mean() if all(w in subj.index for w in ['VSO', 'SVO']) else np.nan
    s_late_subj  = subj.loc[['OVS', 'OSV'], 'mean'].mean() if all(w in subj.index for w in ['OVS', 'OSV']) else np.nan
    R(f'- Subject surprisal: S-early orders (VSO/SVO) avg = {s_early_subj:.3f} bits, '
      f'S-late orders (OVS/OSV) avg = {s_late_subj:.3f} bits.')
    if np.isfinite(s_early_subj) and np.isfinite(s_late_subj):
        direction = 'LOWER' if s_early_subj < s_late_subj else 'HIGHER'
        R(f'  → Subject is {direction} surprisal when it appears early in the sentence.')

    # Object surprisal: is it lower in O-early orders?
    o_early_obj = obj.loc[['OVS', 'OSV'], 'mean'].mean() if all(w in obj.index for w in ['OVS', 'OSV']) else np.nan
    o_late_obj  = obj.loc[['SVO', 'SOV'], 'mean'].mean() if all(w in obj.index for w in ['SVO', 'SOV']) else np.nan
    R(f'- Object surprisal: O-early orders (OVS/OSV) avg = {o_early_obj:.3f} bits, '
      f'O-late orders (SVO/SOV) avg = {o_late_obj:.3f} bits.')
    if np.isfinite(o_early_obj) and np.isfinite(o_late_obj):
        direction = 'LOWER' if o_early_obj < o_late_obj else 'HIGHER'
        R(f'  → Object is {direction} surprisal when it appears early.')

    # Best/worst word order for subject
    best_wo  = subj['mean'].idxmin()
    worst_wo = subj['mean'].idxmax()
    R(f'- Subject easiest to predict in: {best_wo} ({subj.loc[best_wo, "mean"]:.3f} bits)')
    R(f'- Subject hardest to predict in: {worst_wo} ({subj.loc[worst_wo, "mean"]:.3f} bits)')
    R()

R('## Plots saved')
R(f'- argument_surprisal_by_word_order.png — Subj/Obj/V surprisal per word order (5MB vs 10MB)')
R(f'- role_x_word_order_heatmap.png — heatmap of all roles × word orders')
R(f'- subj_vs_obj_by_word_order.png — grouped bars, Subj vs Obj per word order')
R(f'- surprisal_by_subject_position_group.png — role surprisal grouped by S position')

(OUT_DIR / 'argument_surprisal_report.md').write_text('\n'.join(report))
print('\nReport saved to argument_surprisal_report.md')
print(f'\nAll outputs in {OUT_DIR}')
