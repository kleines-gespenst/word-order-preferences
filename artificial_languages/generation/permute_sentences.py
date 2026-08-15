import re
import os
from pathlib import Path
from itertools import product

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

# Define reordering templates for main clause types
WORD_ORDERS = {
    "SOV": ["NP_Subj", "NP_Obj", "Verb"],
    "SVO": ["NP_Subj", "Verb", "NP_Obj"],
    "OSV": ["NP_Obj", "NP_Subj", "Verb"],
    "VSO": ["Verb", "NP_Subj", "NP_Obj"],
    "VOS": ["Verb", "NP_Obj", "NP_Subj"],
    "OVS": ["NP_Obj", "Verb", "NP_Subj"]
}

BASE_DIR_DATA = DATA_ROOT / "word_order_data"
INPUT_FILE = BASE_DIR_DATA / "sample_base_grammar.txt"
OUTPUT_DIR = BASE_DIR_DATA / "permuted_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_PLAIN = BASE_DIR_DATA / "permuted_samples_plain"
OUTPUT_DIR_PLAIN.mkdir(parents=True, exist_ok=True)



def parse_tree(string):
    """Parse Lisp-style bracketed tree into nested list structure"""
    tokens = re.findall(r'\(|\)|[^\s()]+', string)
    stack = []
    current = []
    for token in tokens:
        if token == '(':
            stack.append(current)
            current = []
        elif token == ')':
            parent = stack.pop()
            parent.append(current)
            current = parent
        else:
            current.append(token)
    return current[0]  # Return root


def tree_to_string(tree):
    """Convert nested list structure back to bracketed string"""
    if isinstance(tree, str):
        return tree
    return '(' + ' '.join(tree_to_string(child) for child in tree) + ')'


def get_label(node):
    """Return label without numeric prefix, e.g., 2VP_Comp_P → VP_Comp_P."""
    if isinstance(node, list) and node:
        return re.sub(r'^\d+', '', node[0])
    return None

def raw_label(node):
    """Return the full label (including numeric prefix) of a tree node."""
    return node[0] if isinstance(node, list) and node else None


def ensure_case_terminals(tree):
    """
    Ensure that (Subj) and (Obj) nodes have their terminals.
    If they only have a label, add the terminal.
    """
    if not isinstance(tree, list):
        return tree
    
    label = get_label(tree)
    
    # If this is a (Subj) or (Obj) node with only the label (len==1), add terminal
    if label in ["Subj", "Obj"] and len(tree) == 1:
        terminal = label.lower()
        return [tree[0], terminal]
    
    # Recurse into all children
    return [ensure_case_terminals(child) for child in tree]


def reorder_clause(clause, order):
    """Reorder NP_Subj, NP_Obj, Verb/VP_Comp/IVerb in a 1S clause"""
    label = clause[0]
    if not re.match(r'[0-9]*S', label):
        return clause

    np_subj, np_obj, verb = None, None, None
    others = []

    for child in clause[1:]:
        if isinstance(child, list) and len(child) > 0:
            clabel = get_label(child)
            if clabel and clabel.startswith("NP_Subj"):
                np_subj = child
            elif clabel == "NP_Obj":
                np_obj = child
            elif clabel and clabel.startswith(("TVerb", "IVerb", "VP_Comp")):
                verb = child
            else:
                others.append(child)

    ordered = []
    for part in WORD_ORDERS[order]:
        if part == "NP_Subj" and np_subj:
            ordered.append(np_subj)
        elif part == "NP_Obj" and np_obj:
            ordered.append(np_obj)
        elif part == "Verb" and verb:
            ordered.append(verb)

    return [label] + ordered + others


def apply_switches(tree, s2=False, s3=False, s4=False, s5=False, s6=False):
    """Apply switches 2–6 recursively to tree structure."""
    if not isinstance(tree, list):
        return tree

    label = get_label(tree)
    rlabel = raw_label(tree)

    # === Switch 2: VP_Comp_X → Verb_Comp_X S_Comp ===
    if s2 and label and label.startswith("VP_Comp"):
        s_comp, v_comp = None, None
        for child in tree[1:]:
            clabel = get_label(child)
            if clabel == "S_Comp":
                s_comp = child
            elif clabel and clabel.startswith("Verb_Comp"):
                v_comp = child
        if s_comp and v_comp:
            reordered = [tree[0], v_comp, s_comp]
            return [apply_switches(x, s2, s3, s4, s5, s6) for x in reordered]

    # === Switch 3: S_Comp → Comp S ===
    if s3 and label == "S_Comp":
        s_node, comp = None, None
        for child in tree[1:]:
            clabel = get_label(child)
            if clabel == "Comp":
                comp = child
            elif clabel and (clabel.startswith("1S") or clabel.startswith("S")):
                s_node = child
        if s_node and comp:
            reordered = [tree[0], comp, s_node]
            return [apply_switches(x, s2, s3, s4, s5, s6) for x in reordered]

    # === Switch 4: Flatten NP_S → (PP NP_S) → NP_S + Prep ===
    if s4 and rlabel in {"4NP_S", "4NP_P"} and len(tree) == 3:
        first_child = tree[1]
        second_child = tree[2]

        if raw_label(first_child) == "4PP" and len(first_child) == 3:
            inner_np = first_child[1]
            prep = first_child[2]
            new_tree = [rlabel, second_child, prep, inner_np]

            return [apply_switches(x, s2, s3, s4, s5, s6) for x in new_tree]

    # === Switch 5: NP → Noun Adj ===
    if s5 and label and label.startswith("NP_"):
        noun, adj = None, None
        for child in tree[1:]:
            clabel = get_label(child)
            if clabel and clabel.startswith("Noun_"):
                noun = child
            elif clabel == "Adj":
                adj = child
        if noun and adj:
            reordered = [tree[0], noun, adj]
            return [apply_switches(x, s2, s3, s4, s5, s6) for x in reordered]

    # === Switch 6: NP relative clause reordering ===
    if label and label.startswith(("NP_Obj", "NP_Subj")):
        rel_np, case_marker = None, None
        for child in tree[1:]:
            child_rlabel = raw_label(child)
            clabel = get_label(child)
            if child_rlabel in {"6NP_S", "6NP_P"}:
                rel_np = child
            elif clabel in {"Obj", "Subj"}:
                case_marker = child
        
        if rel_np and case_marker and len(rel_np) == 4:
            vp_rel, rel, noun = None, None, None
            for child in rel_np[1:]:
                clabel = get_label(child)
                if clabel and clabel.startswith("VP_Rel"):
                    vp_rel = child
                elif clabel == "Rel":
                    rel = child
                elif clabel and clabel.startswith("Noun_"):
                    noun = child
            
            if vp_rel and rel and noun:
                if s6:
                    # Switch 6 ON: Noun Obj/Subj Rel VP_Rel (case marker moves inside)
                    new_rel_np = [rel_np[0], noun, case_marker, rel, vp_rel]
                    reordered = [tree[0], new_rel_np]
                else:
                    # Switch 6 OFF: VP_Rel Rel Noun, case marker stays outside
                    new_rel_np = [rel_np[0], vp_rel, rel, noun]
                    reordered = [tree[0], new_rel_np, case_marker]
                
                return [apply_switches(x, s2, s3, s4, s5, s6) for x in reordered]

    # Recurse into all children
    return [apply_switches(child, s2, s3, s4, s5, s6) for child in tree]


def recursively_transform(tree, order, s2=False, s3=False, s4=False, s5=False, s6=False):
    """Apply base word order and switches recursively."""
    if isinstance(tree, list):
        if len(tree) > 0 and re.match(r'[0-9]*S', tree[0]):
            tree = reorder_clause(tree, order)
        tree = apply_switches(tree, s2, s3, s4, s5, s6)
        return [recursively_transform(child, order, s2, s3, s4, s5, s6) for child in tree]
    return tree


def transform_sentences(sentences, order, s2=False, s3=False, s4=False, s5=False, s6=False):
    """Transform all sentences in file."""
    transformed = []
    for sent in sentences:
        tree = parse_tree(sent)
        # Ensure case terminals are present (in case some got lost)
        tree = ensure_case_terminals(tree)
        new_tree = recursively_transform(tree, order, s2, s3, s4, s5, s6)
        # Ensure case terminals again after all transformations
        new_tree = ensure_case_terminals(new_tree)
        transformed.append(tree_to_string(new_tree))
    return transformed


def flatten_sentence(tree):
    """Flatten a parsed tree to word tokens, skipping node labels."""
    if isinstance(tree, str):
        return [tree]
    words = []
    # skip the first element (node label), recurse into children
    for child in tree[1:]:
        words.extend(flatten_sentence(child))
    return words

def strip_parentheses_and_labels(bracketed_str):
    """Convert a bracketed tree string to plain tokens (remove parentheses/labels)."""
    try:
        tree = parse_tree(bracketed_str)
        words = flatten_sentence(tree)
        sentence = " ".join(words) + " ."
        return sentence
    except Exception:
        # Fallback: extract non-label tokens
        toks = re.findall(r'[^\s()]+', bracketed_str)
        kept = []
        for t in toks:
            # Keep lowercase terminals (including 'subj', 'obj')
            if t[0].islower():
                kept.append(t)
            # Drop labels (uppercase start or contains underscore or starts with digit)
            elif t[0].isupper() or '_' in t or t[0].isdigit():
                continue
            else:
                kept.append(t)
        sentence = " ".join(kept) + " ."
        return sentence


def main():
    with open(INPUT_FILE, "r") as f:
        sentences = [line.strip() for line in f if line.strip()]

    for order in WORD_ORDERS:
        for s2, s3, s4, s5, s6 in product([False, True], repeat=5):
            suffix = f"{order}_s2{s2}_s3{s3}_s4{s4}_s5{s5}_s6{s6}"
            transformed = transform_sentences(sentences, order, s2, s3, s4, s5, s6)

            # Original write (with labels and parentheses)
            out_path = OUTPUT_DIR / f"sample_{suffix}.txt"
            with open(out_path, "w") as f_out:
                for line in transformed:
                    f_out.write(line + "\n")
            print(f"Wrote: {out_path}")

            # Plain version (without labels and parentheses)
            plain_transformed = [strip_parentheses_and_labels(line) for line in transformed]
            out_path_plain = OUTPUT_DIR_PLAIN / f"sample_{suffix}.txt"
            with open(out_path_plain, "w") as f_plain:
                for line in plain_transformed:
                    f_plain.write(line + "\n")
            print(f"Wrote: {out_path_plain}")


if __name__ == "__main__":
    main()