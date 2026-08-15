#!/usr/bin/env python3
"""
Cross-configuration surprisal analysis.

Loads per-token surprisal results from all 192 grammar configurations
(6 word orders × 32 switch combos) and produces:
  1. Overall surprisal by word order
  2. Per-role surprisal by word order (which role is hardest in which order?)
  3. Per-role-group surprisal by word order
  4. Effect of each binary switch (s2–s6) on surprisal
  5. Best/worst switch configurations overall and per word order
  6. Interaction: word order × switch effects
  7. Qualitative feature analysis across configs (clause type, rel/comp, etc.)
  8. Summary tables (CSV) and plots

Reads from: $WORD_ORDER_DATA/surprisal_by_category/<grammar>/0/
Writes to:  $WORD_ORDER_DATA/surprisal_cross_config/
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))


# ---------------------------------------------------------------------------
# Parsing grammar names
# ---------------------------------------------------------------------------

WORD_ORDERS = ["SVO", "SOV", "VSO", "VOS", "OSV", "OVS"]
SWITCH_NAMES = ["s2", "s3", "s4", "s5", "s6"]

def parse_grammar_name(name: str):
    """Parse e.g. 'OSV_s2False_s3True_s4False_s5False_s6True' into
    (word_order, {s2: False, s3: True, s4: False, s5: False, s6: True}, switch_bits_str)."""
    m = re.match(
        r"^(SVO|SOV|VSO|VOS|OSV|OVS)_s2(True|False)_s3(True|False)_s4(True|False)_s5(True|False)_s6(True|False)$",
        name,
    )
    if not m:
        return None, None, None
    wo = m.group(1)
    switches = {}
    bits = []
    for i, sname in enumerate(SWITCH_NAMES, start=2):
        val = m.group(i) == "True"
        switches[sname] = val
        bits.append("1" if val else "0")
    return wo, switches, "".join(bits)


# ---------------------------------------------------------------------------
# Loading CSV files
# ---------------------------------------------------------------------------

def load_category_csv(path: str):
    """Load per_token_surprisal_by_category.csv -> list of dicts."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["n_tokens"] = int(row["n_tokens"])
            row["mean_surprisal_bits"] = float(row["mean_surprisal_bits"])
            row["std_surprisal_bits"] = float(row["std_surprisal_bits"])
            rows.append(row)
    return rows


def load_qualitative_csv(path: str):
    """Load qualitative_analysis.csv -> list of dicts."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["n"] = int(row["n"])
            for k in ("mean_bits", "std_bits", "median_bits", "p25_bits", "p75_bits"):
                row[k] = float(row[k])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_mean(means, counts):
    total_n = sum(counts)
    if total_n == 0:
        return float("nan")
    return sum(m * n for m, n in zip(means, counts)) / total_n


def fmt(x, decimals=4):
    if x != x:  # nan
        return "nan"
    return f"{x:.{decimals}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-configuration surprisal analysis")
    parser.add_argument("--base_dir", default=str(DATA_ROOT / "surprisal_by_category"),
                        help="Base directory with per-grammar results")
    parser.add_argument("--split", default="0", help="Split to analyse (default: 0)")
    parser.add_argument("--output_dir", default=str(DATA_ROOT / "surprisal_cross_config"),
                        help="Output directory for cross-config analysis")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load all data
    # ------------------------------------------------------------------
    all_data = []  # list of dicts: {grammar, word_order, switches, bits_str, cat_rows, qual_rows}
    grammars = sorted(d for d in os.listdir(base) if (base / d / args.split).is_dir())
    skipped = 0

    for gname in grammars:
        wo, switches, bits_str = parse_grammar_name(gname)
        if wo is None:
            skipped += 1
            continue
        cat_path = base / gname / args.split / "per_token_surprisal_by_category.csv"
        qual_path = base / gname / args.split / "qualitative_analysis.csv"
        if not cat_path.exists():
            skipped += 1
            continue
        cat_rows = load_category_csv(str(cat_path))
        qual_rows = load_qualitative_csv(str(qual_path)) if qual_path.exists() else []
        all_data.append({
            "grammar": gname,
            "word_order": wo,
            "switches": switches,
            "bits_str": bits_str,
            "cat_rows": cat_rows,
            "qual_rows": qual_rows,
        })

    print(f"Loaded {len(all_data)} grammar configurations (skipped {skipped})")
    if not all_data:
        print("No data found. Exiting.")
        return 1

    # Pre-compute per-grammar overall mean surprisal (weighted by token count, excluding Punctuation)
    for d in all_data:
        content_rows = [r for r in d["cat_rows"] if r["category"] != "Punctuation"]
        d["overall_mean"] = weighted_mean(
            [r["mean_surprisal_bits"] for r in content_rows],
            [r["n_tokens"] for r in content_rows],
        )
        d["overall_n"] = sum(r["n_tokens"] for r in content_rows)

    # ------------------------------------------------------------------
    # 2. Overall surprisal by word order
    # ------------------------------------------------------------------
    print("OVERALL SURPRISAL BY WORD ORDER (excl. punctuation)")

    wo_data = defaultdict(list)
    for d in all_data:
        wo_data[d["word_order"]].append(d["overall_mean"])

    wo_summary = []
    for wo in WORD_ORDERS:
        vals = wo_data[wo]
        if vals:
            wo_summary.append({
                "word_order": wo,
                "n_configs": len(vals),
                "mean": np.mean(vals),
                "std": np.std(vals),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })
            print(f"  {wo:4s}  n={len(vals):3d}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"median={np.median(vals):.4f}  range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")

    with open(out_dir / "overall_by_word_order.csv", "w") as f:
        f.write("word_order,n_configs,mean_bits,std_bits,median_bits,min_bits,max_bits\n")
        for r in wo_summary:
            f.write(f"{r['word_order']},{r['n_configs']},{fmt(r['mean'])},{fmt(r['std'])},"
                    f"{fmt(r['median'])},{fmt(r['min'])},{fmt(r['max'])}\n")
    print(f" Saved: {out_dir / 'overall_by_word_order.csv'}")

    # ------------------------------------------------------------------
    # 3. Per-role mean surprisal by word order
    # ------------------------------------------------------------------
    print("PER-ROLE SURPRISAL BY WORD ORDER")

    # Collect: role -> word_order -> list of (mean, n) from each config
    role_wo = defaultdict(lambda: defaultdict(list))
    all_roles = set()
    for d in all_data:
        for r in d["cat_rows"]:
            if r["category"] == "Punctuation":
                continue
            role_wo[r["category"]][d["word_order"]].append((r["mean_surprisal_bits"], r["n_tokens"]))
            all_roles.add(r["category"])

    role_wo_rows = []
    for role in sorted(all_roles):
        for wo in WORD_ORDERS:
            pairs = role_wo[role][wo]
            if not pairs:
                continue
            means = [p[0] for p in pairs]
            counts = [p[1] for p in pairs]
            wm = weighted_mean(means, counts)
            role_wo_rows.append({
                "role": role, "word_order": wo,
                "n_configs": len(pairs),
                "weighted_mean": wm,
                "unweighted_mean": np.mean(means),
                "std_across_configs": np.std(means),
                "total_tokens": sum(counts),
            })

    with open(out_dir / "role_by_word_order.csv", "w") as f:
        f.write("role,word_order,n_configs,weighted_mean_bits,unweighted_mean_bits,"
                "std_across_configs,total_tokens\n")
        for r in role_wo_rows:
            f.write(f"{r['role']},{r['word_order']},{r['n_configs']},"
                    f"{fmt(r['weighted_mean'])},{fmt(r['unweighted_mean'])},"
                    f"{fmt(r['std_across_configs'])},{r['total_tokens']}\n")
    print(f" Saved: {out_dir / 'role_by_word_order.csv'}")

    # Print top roles with biggest word order effect (max - min across orders)
    role_range = {}
    for role in sorted(all_roles):
        wmeans = {}
        for wo in WORD_ORDERS:
            pairs = role_wo[role][wo]
            if pairs:
                wmeans[wo] = weighted_mean([p[0] for p in pairs], [p[1] for p in pairs])
        if len(wmeans) >= 2:
            role_range[role] = max(wmeans.values()) - min(wmeans.values())
    top_varied = sorted(role_range.items(), key=lambda x: -x[1])[:15]
    print("\n  Roles with largest word order effect (max-min across 6 orders):")
    for role, spread in top_varied:
        wmeans = {}
        for wo in WORD_ORDERS:
            pairs = role_wo[role][wo]
            if pairs:
                wmeans[wo] = weighted_mean([p[0] for p in pairs], [p[1] for p in pairs])
        best_wo = min(wmeans, key=wmeans.get)
        worst_wo = max(wmeans, key=wmeans.get)
        print(f"    {role:30s}  spread={spread:.4f}  best={best_wo}({wmeans[best_wo]:.3f})  "
              f"worst={worst_wo}({wmeans[worst_wo]:.3f})")

    # ------------------------------------------------------------------
    # 4. Role group by word order
    # ------------------------------------------------------------------
    print("ROLE GROUP SURPRISAL BY WORD ORDER")

    rg_wo = defaultdict(lambda: defaultdict(list))
    for d in all_data:
        for qr in d["qual_rows"]:
            if qr["analysis_group"] == "by_role_group":
                rg_wo[qr["key"]][d["word_order"]].append((qr["mean_bits"], qr["n"]))

    rg_order = ["marker", "subj_content", "obj_content", "rel_content", "verb", "adj", "prep", "other"]
    rg_wo_rows = []
    for rg in rg_order:
        vals_per_wo = {}
        for wo in WORD_ORDERS:
            pairs = rg_wo[rg][wo]
            if pairs:
                wm = weighted_mean([p[0] for p in pairs], [p[1] for p in pairs])
                vals_per_wo[wo] = wm
                rg_wo_rows.append({"role_group": rg, "word_order": wo,
                                   "weighted_mean": wm, "n_configs": len(pairs)})
        if vals_per_wo:
            parts = "  ".join(f"{wo}={vals_per_wo.get(wo, float('nan')):.3f}" for wo in WORD_ORDERS)
            print(f"  {rg:15s}  {parts}")

    with open(out_dir / "role_group_by_word_order.csv", "w") as f:
        f.write("role_group,word_order,weighted_mean_bits,n_configs\n")
        for r in rg_wo_rows:
            f.write(f"{r['role_group']},{r['word_order']},{fmt(r['weighted_mean'])},{r['n_configs']}\n")
    print(f" Saved: {out_dir / 'role_group_by_word_order.csv'}")

    # ------------------------------------------------------------------
    # 5. Effect of each binary switch (s2-s6)
    # ------------------------------------------------------------------
    print("EFFECT OF EACH BINARY SWITCH ON OVERALL SURPRISAL")

    switch_effects = []
    for sname in SWITCH_NAMES:
        on_vals = [d["overall_mean"] for d in all_data if d["switches"][sname]]
        off_vals = [d["overall_mean"] for d in all_data if not d["switches"][sname]]
        diff = np.mean(on_vals) - np.mean(off_vals)
        switch_effects.append({
            "switch": sname,
            "on_mean": np.mean(on_vals), "on_std": np.std(on_vals), "on_n": len(on_vals),
            "off_mean": np.mean(off_vals), "off_std": np.std(off_vals), "off_n": len(off_vals),
            "diff_on_minus_off": diff,
        })
        direction = "harder" if diff > 0 else "easier"
        print(f"  {sname}: ON={np.mean(on_vals):.4f}±{np.std(on_vals):.4f} (n={len(on_vals)})  "
              f"OFF={np.mean(off_vals):.4f}±{np.std(off_vals):.4f} (n={len(off_vals)})  "
              f"diff={diff:+.4f} ({direction})")

    # Per word order × switch effect
    print("\n  Switch effects broken down by word order:")
    switch_wo_effects = []
    for sname in SWITCH_NAMES:
        for wo in WORD_ORDERS:
            on_vals = [d["overall_mean"] for d in all_data
                       if d["switches"][sname] and d["word_order"] == wo]
            off_vals = [d["overall_mean"] for d in all_data
                        if not d["switches"][sname] and d["word_order"] == wo]
            if on_vals and off_vals:
                diff = np.mean(on_vals) - np.mean(off_vals)
                switch_wo_effects.append({
                    "switch": sname, "word_order": wo,
                    "on_mean": np.mean(on_vals), "off_mean": np.mean(off_vals),
                    "diff": diff,
                })

    with open(out_dir / "switch_effects.csv", "w") as f:
        f.write("switch,on_mean,on_std,on_n,off_mean,off_std,off_n,diff_on_minus_off\n")
        for r in switch_effects:
            f.write(f"{r['switch']},{fmt(r['on_mean'])},{fmt(r['on_std'])},{r['on_n']},"
                    f"{fmt(r['off_mean'])},{fmt(r['off_std'])},{r['off_n']},"
                    f"{fmt(r['diff_on_minus_off'])}\n")

    with open(out_dir / "switch_effects_by_word_order.csv", "w") as f:
        f.write("switch,word_order,on_mean,off_mean,diff_on_minus_off\n")
        for r in switch_wo_effects:
            f.write(f"{r['switch']},{r['word_order']},{fmt(r['on_mean'])},{fmt(r['off_mean'])},"
                    f"{fmt(r['diff'])}\n")
    print(f" Saved: {out_dir / 'switch_effects.csv'}")
    print(f" Saved: {out_dir / 'switch_effects_by_word_order.csv'}")

    # ------------------------------------------------------------------
    # 6. Best/worst switch configurations
    # ------------------------------------------------------------------
    print("BEST AND WORST SWITCH CONFIGURATIONS (by overall mean surprisal)")

    # Per switch config (across all word orders)
    bits_data = defaultdict(list)
    for d in all_data:
        bits_data[d["bits_str"]].append(d["overall_mean"])

    bits_summary = []
    for bits_str, vals in bits_data.items():
        bits_summary.append({
            "bits": bits_str,
            "mean": np.mean(vals),
            "std": np.std(vals),
            "n": len(vals),
        })
    bits_summary.sort(key=lambda x: x["mean"])

    print("\n  Top 5 easiest switch configs (lowest mean surprisal):")
    for r in bits_summary[:5]:
        switches_str = ", ".join(f"{SWITCH_NAMES[i]}={'T' if r['bits'][i]=='1' else 'F'}"
                                 for i in range(5))
        print(f"    {r['bits']}  ({switches_str})  mean={r['mean']:.4f}±{r['std']:.4f}")

    print("\n  Top 5 hardest switch configs (highest mean surprisal):")
    for r in bits_summary[-5:]:
        switches_str = ", ".join(f"{SWITCH_NAMES[i]}={'T' if r['bits'][i]=='1' else 'F'}"
                                 for i in range(5))
        print(f"    {r['bits']}  ({switches_str})  mean={r['mean']:.4f}±{r['std']:.4f}")

    with open(out_dir / "switch_config_ranking.csv", "w") as f:
        f.write("bits_s2s3s4s5s6,s2,s3,s4,s5,s6,mean_bits,std_bits,n_configs\n")
        for r in bits_summary:
            svals = ",".join("True" if r["bits"][i] == "1" else "False" for i in range(5))
            f.write(f"{r['bits']},{svals},{fmt(r['mean'])},{fmt(r['std'])},{r['n']}\n")
    print(f" Saved: {out_dir / 'switch_config_ranking.csv'}")

    # Per word order: best/worst config
    print("\n  Best/worst switch config PER word order:")
    wo_best_worst_rows = []
    for wo in WORD_ORDERS:
        wo_configs = [(d["bits_str"], d["overall_mean"]) for d in all_data if d["word_order"] == wo]
        if not wo_configs:
            continue
        wo_configs.sort(key=lambda x: x[1])
        best_bits, best_mean = wo_configs[0]
        worst_bits, worst_mean = wo_configs[-1]
        spread = worst_mean - best_mean
        wo_best_worst_rows.append({
            "word_order": wo,
            "best_bits": best_bits, "best_mean": best_mean,
            "worst_bits": worst_bits, "worst_mean": worst_mean,
            "spread": spread,
        })
        print(f"  {wo:4s}  best={best_bits}({best_mean:.4f})  worst={worst_bits}({worst_mean:.4f})  "
              f"spread={spread:.4f}")

    with open(out_dir / "best_worst_per_word_order.csv", "w") as f:
        f.write("word_order,best_bits,best_mean_bits,worst_bits,worst_mean_bits,spread\n")
        for r in wo_best_worst_rows:
            f.write(f"{r['word_order']},{r['best_bits']},{fmt(r['best_mean'])},"
                    f"{r['worst_bits']},{fmt(r['worst_mean'])},{fmt(r['spread'])}\n")
    print(f" Saved: {out_dir / 'best_worst_per_word_order.csv'}")

    # ------------------------------------------------------------------
    # 7. Full grammar ranking (all 192)
    # ------------------------------------------------------------------
    print("FULL GRAMMAR RANKING (top 10 easiest, top 10 hardest)")

    grammar_ranking = sorted(all_data, key=lambda d: d["overall_mean"])

    print("\n  Top 10 easiest grammars:")
    for i, d in enumerate(grammar_ranking[:10], 1):
        print(f"    {i:2d}. {d['grammar']:55s}  mean={d['overall_mean']:.4f}")

    print("\n  Top 10 hardest grammars:")
    for i, d in enumerate(grammar_ranking[-10:], 1):
        rank = len(grammar_ranking) - 10 + i
        print(f"    {rank:3d}. {d['grammar']:55s}  mean={d['overall_mean']:.4f}")

    with open(out_dir / "full_grammar_ranking.csv", "w") as f:
        f.write("rank,grammar,word_order,bits_s2s3s4s5s6,overall_mean_bits\n")
        for rank, d in enumerate(grammar_ranking, 1):
            f.write(f"{rank},{d['grammar']},{d['word_order']},{d['bits_str']},"
                    f"{fmt(d['overall_mean'])}\n")
    print(f" Saved: {out_dir / 'full_grammar_ranking.csv'}")

    # ------------------------------------------------------------------
    # 8. Qualitative feature analysis across configs
    # ------------------------------------------------------------------
    print("QUALITATIVE FEATURES BY WORD ORDER")

    feature_groups = [
        "by_clause_type", "by_has_rel_clause", "by_has_comp_clause",
        "noun_singular_vs_plural", "verb_past_vs_present",
        "verb_intransitive_vs_transitive", "verb_rel_vs_main_clause",
        "noun_rel_vs_main_clause",
    ]

    qual_wo = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d in all_data:
        for qr in d["qual_rows"]:
            if qr["analysis_group"] in feature_groups:
                qual_wo[qr["analysis_group"]][qr["key"]][d["word_order"]].append(
                    (qr["mean_bits"], qr["n"]))

    qual_cross_rows = []
    for ag in feature_groups:
        for key in sorted(qual_wo[ag].keys()):
            for wo in WORD_ORDERS:
                pairs = qual_wo[ag][key][wo]
                if pairs:
                    wm = weighted_mean([p[0] for p in pairs], [p[1] for p in pairs])
                    qual_cross_rows.append({
                        "analysis_group": ag, "key": key, "word_order": wo,
                        "weighted_mean": wm, "n_configs": len(pairs),
                    })
            vals_str = "  ".join(
                f"{wo}={weighted_mean([p[0] for p in qual_wo[ag][key][wo]], [p[1] for p in qual_wo[ag][key][wo]]):.3f}"
                if qual_wo[ag][key][wo] else f"{wo}=  -  "
                for wo in WORD_ORDERS
            )
            print(f"    {key:30s}  {vals_str}")

    with open(out_dir / "qualitative_by_word_order.csv", "w") as f:
        f.write("analysis_group,key,word_order,weighted_mean_bits,n_configs\n")
        for r in qual_cross_rows:
            f.write(f"{r['analysis_group']},{r['key']},{r['word_order']},"
                    f"{fmt(r['weighted_mean'])},{r['n_configs']}\n")
    print(f"\n Saved: {out_dir / 'qualitative_by_word_order.csv'}")

    # ------------------------------------------------------------------
    # 9. Switch effect on each role group (which roles are affected by which switch?)
    # ------------------------------------------------------------------
    print("SWITCH EFFECTS ON ROLE GROUPS")

    switch_rg_rows = []
    for sname in SWITCH_NAMES:
        for rg in rg_order:
            on_means = []
            off_means = []
            for d in all_data:
                for qr in d["qual_rows"]:
                    if qr["analysis_group"] == "by_role_group" and qr["key"] == rg:
                        if d["switches"][sname]:
                            on_means.append(qr["mean_bits"])
                        else:
                            off_means.append(qr["mean_bits"])
            if on_means and off_means:
                diff = np.mean(on_means) - np.mean(off_means)
                switch_rg_rows.append({
                    "switch": sname, "role_group": rg,
                    "on_mean": np.mean(on_means), "off_mean": np.mean(off_means),
                    "diff": diff,
                })

    with open(out_dir / "switch_effects_on_role_groups.csv", "w") as f:
        f.write("switch,role_group,on_mean_bits,off_mean_bits,diff_on_minus_off\n")
        for r in switch_rg_rows:
            f.write(f"{r['switch']},{r['role_group']},{fmt(r['on_mean'])},"
                    f"{fmt(r['off_mean'])},{fmt(r['diff'])}\n")

    print("  Switch × role group effects (diff = ON - OFF, positive = harder):")
    for sname in SWITCH_NAMES:
        parts = []
        for r in switch_rg_rows:
            if r["switch"] == sname:
                parts.append(f"{r['role_group']}={r['diff']:+.3f}")
        print(f"    {sname}: {', '.join(parts)}")
    print(f" Saved: {out_dir / 'switch_effects_on_role_groups.csv'}")

    # ------------------------------------------------------------------
    # 10. Interpretation / automatic findings
    # ------------------------------------------------------------------
    print("AUTOMATIC FINDINGS SUMMARY")

    findings = []

    # Finding: easiest/hardest word order
    if wo_summary:
        easiest = min(wo_summary, key=lambda x: x["mean"])
        hardest = max(wo_summary, key=lambda x: x["mean"])
        findings.append(
            f"Easiest word order: {easiest['word_order']} (mean={easiest['mean']:.4f}), "
            f"Hardest: {hardest['word_order']} (mean={hardest['mean']:.4f}), "
            f"difference={hardest['mean'] - easiest['mean']:.4f} bits"
        )

    # Finding: most impactful switch
    if switch_effects:
        biggest_switch = max(switch_effects, key=lambda x: abs(x["diff_on_minus_off"]))
        direction = "increases" if biggest_switch["diff_on_minus_off"] > 0 else "decreases"
        findings.append(
            f"Most impactful switch: {biggest_switch['switch']} "
            f"({direction} surprisal by {abs(biggest_switch['diff_on_minus_off']):.4f} bits)"
        )

    # Finding: which role is most affected by word order
    if role_range:
        most_varied_role = max(role_range, key=role_range.get)
        findings.append(
            f"Role most affected by word order: {most_varied_role} "
            f"(spread={role_range[most_varied_role]:.4f} bits across 6 orders)"
        )

    # Finding: hardest role overall
    overall_role_mean = {}
    for role in all_roles:
        all_pairs = []
        for wo in WORD_ORDERS:
            all_pairs.extend(role_wo[role][wo])
        if all_pairs:
            overall_role_mean[role] = weighted_mean([p[0] for p in all_pairs], [p[1] for p in all_pairs])
    content_roles = {r: m for r, m in overall_role_mean.items()
                     if "marker" not in r and r != "Punctuation"}
    if content_roles:
        hardest_role = max(content_roles, key=content_roles.get)
        easiest_role = min(content_roles, key=content_roles.get)
        findings.append(
            f"Hardest content role overall: {hardest_role} ({content_roles[hardest_role]:.4f}), "
            f"Easiest: {easiest_role} ({content_roles[easiest_role]:.4f})"
        )

    # Finding: is subject or object harder per word order?
    for wo in WORD_ORDERS:
        subj_pairs = []
        obj_pairs = []
        for role in all_roles:
            for p in role_wo[role][wo]:
                if "Subj_Noun" in role or "Subj_Pronoun" in role:
                    subj_pairs.append(p)
                elif "Obj_Noun" in role or "Obj_Pronoun" in role:
                    obj_pairs.append(p)
        if subj_pairs and obj_pairs:
            subj_mean = weighted_mean([p[0] for p in subj_pairs], [p[1] for p in subj_pairs])
            obj_mean = weighted_mean([p[0] for p in obj_pairs], [p[1] for p in obj_pairs])
            harder = "Subject" if subj_mean > obj_mean else "Object"
            findings.append(
                f"{wo}: {harder} is harder (Subj={subj_mean:.3f}, Obj={obj_mean:.3f}, "
                f"diff={abs(subj_mean - obj_mean):.3f})"
            )

    # Finding: relative clause effect per word order
    for wo in WORD_ORDERS:
        rel_pairs = []
        main_pairs = []
        for d in all_data:
            if d["word_order"] != wo:
                continue
            for qr in d["qual_rows"]:
                if qr["analysis_group"] == "verb_rel_vs_main_clause":
                    if qr["key"] == "rel_clause_verb":
                        rel_pairs.append((qr["mean_bits"], qr["n"]))
                    elif qr["key"] == "main_clause_verb":
                        main_pairs.append((qr["mean_bits"], qr["n"]))
        if rel_pairs and main_pairs:
            rel_mean = weighted_mean([p[0] for p in rel_pairs], [p[1] for p in rel_pairs])
            main_mean = weighted_mean([p[0] for p in main_pairs], [p[1] for p in main_pairs])
            diff = rel_mean - main_mean
            findings.append(
                f"{wo}: Rel clause verbs {'harder' if diff > 0 else 'easier'} than main "
                f"(rel={rel_mean:.3f}, main={main_mean:.3f}, diff={diff:+.3f})"
            )

    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")

    findings_path = out_dir / "findings_summary.txt"
    with open(findings_path, "w") as f:
        for i, finding in enumerate(findings, 1):
            f.write(f"{i}. {finding}\n")
    print(f"\n Saved: {findings_path}")

    # ------------------------------------------------------------------
    # 11. Plots
    # ------------------------------------------------------------------
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def _save(fig, name):
                p = out_dir / name
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f" Plot: {p}")

            # --- Plot 1: Overall surprisal by word order (box plot) ---
            fig, ax = plt.subplots(figsize=(8, 5))
            box_data = [wo_data[wo] for wo in WORD_ORDERS]
            bp = ax.boxplot(box_data, labels=WORD_ORDERS, patch_artist=True)
            colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel("Mean surprisal (bits)")
            ax.set_title("Overall surprisal by word order (across 32 switch configs)")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            _save(fig, "plot_overall_by_word_order.png")

            # --- Plot 2: Role group heatmap (word order × role group) ---
            fig, ax = plt.subplots(figsize=(10, 5))
            rg_matrix = []
            rg_labels_used = []
            for rg in rg_order:
                row = []
                has_data = False
                for wo in WORD_ORDERS:
                    pairs = rg_wo[rg][wo]
                    if pairs:
                        wm = weighted_mean([p[0] for p in pairs], [p[1] for p in pairs])
                        row.append(wm)
                        has_data = True
                    else:
                        row.append(float("nan"))
                if has_data:
                    rg_matrix.append(row)
                    rg_labels_used.append(rg)
            if rg_matrix:
                arr = np.array(rg_matrix)
                im = ax.imshow(arr, aspect="auto", cmap="YlOrRd")
                ax.set_xticks(range(len(WORD_ORDERS)))
                ax.set_xticklabels(WORD_ORDERS, fontsize=11)
                ax.set_yticks(range(len(rg_labels_used)))
                ax.set_yticklabels(rg_labels_used, fontsize=10)
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        if np.isfinite(arr[i, j]):
                            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                                    fontsize=9, color="black" if arr[i, j] < np.nanmax(arr) * 0.7 else "white")
                ax.set_title("Mean surprisal: role group × word order")
                fig.colorbar(im, ax=ax, label="Mean surprisal (bits)")
                plt.tight_layout()
                _save(fig, "plot_role_group_x_word_order_heatmap.png")

            # --- Plot 3: Switch effects bar chart ---
            fig, ax = plt.subplots(figsize=(7, 5))
            sw_names = [r["switch"] for r in switch_effects]
            sw_diffs = [r["diff_on_minus_off"] for r in switch_effects]
            bar_colors = ["#c44e52" if d > 0 else "#55a868" for d in sw_diffs]
            ax.bar(sw_names, sw_diffs, color=bar_colors, alpha=0.8, edgecolor="black")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Δ surprisal (ON − OFF, bits)")
            ax.set_title("Effect of each binary switch on overall surprisal")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            _save(fig, "plot_switch_effects.png")

            # --- Plot 4: Switch config ranking (bar chart, all 32) ---
            fig, ax = plt.subplots(figsize=(14, 5))
            bits_labels = [r["bits"] for r in bits_summary]
            bits_means = [r["mean"] for r in bits_summary]
            bits_stds = [r["std"] for r in bits_summary]
            ax.bar(range(len(bits_labels)), bits_means, yerr=bits_stds, capsize=2,
                   color="steelblue", alpha=0.8)
            ax.set_xticks(range(len(bits_labels)))
            ax.set_xticklabels(bits_labels, rotation=90, fontsize=7)
            ax.set_xlabel("Switch configuration (s2 s3 s4 s5 s6)")
            ax.set_ylabel("Mean surprisal (bits)")
            ax.set_title("All 32 switch configurations ranked by mean surprisal")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            _save(fig, "plot_switch_config_ranking.png")

            # --- Plot 5: Subject vs Object hardness by word order ---
            fig, ax = plt.subplots(figsize=(8, 5))
            subj_means_wo = []
            obj_means_wo = []
            for wo in WORD_ORDERS:
                subj_pairs = []
                obj_pairs = []
                for role in all_roles:
                    for p in role_wo[role][wo]:
                        if "Subj_Noun" in role or "Subj_Pronoun" in role:
                            subj_pairs.append(p)
                        elif "Obj_Noun" in role or "Obj_Pronoun" in role:
                            obj_pairs.append(p)
                subj_means_wo.append(weighted_mean([p[0] for p in subj_pairs], [p[1] for p in subj_pairs]) if subj_pairs else 0)
                obj_means_wo.append(weighted_mean([p[0] for p in obj_pairs], [p[1] for p in obj_pairs]) if obj_pairs else 0)
            x = np.arange(len(WORD_ORDERS))
            w = 0.35
            ax.bar(x - w / 2, subj_means_wo, w, label="Subject", color="#4c72b0", alpha=0.8)
            ax.bar(x + w / 2, obj_means_wo, w, label="Object", color="#dd8452", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(WORD_ORDERS)
            ax.set_ylabel("Mean surprisal (bits)")
            ax.set_title("Subject vs Object surprisal by word order")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            _save(fig, "plot_subj_vs_obj_by_word_order.png")

            # --- Plot 5b: S / V / O comparison across base word orders ---
            # S and O are aggregated from detailed subject/object content roles.
            # V comes from qualitative role-group analysis ("verb"), which is robust
            # across detailed role naming variants.
            s_means_wo, o_means_wo, v_means_wo = [], [], []
            for wo in WORD_ORDERS:
                # Subject/Object from role categories
                subj_pairs = []
                obj_pairs = []
                for role in all_roles:
                    for p in role_wo[role][wo]:
                        if "Subj_Noun" in role or "Subj_Pronoun" in role:
                            subj_pairs.append(p)
                        elif "Obj_Noun" in role or "Obj_Pronoun" in role:
                            obj_pairs.append(p)
                s_mean = weighted_mean([p[0] for p in subj_pairs], [p[1] for p in subj_pairs]) if subj_pairs else float("nan")
                o_mean = weighted_mean([p[0] for p in obj_pairs], [p[1] for p in obj_pairs]) if obj_pairs else float("nan")

                # Verb from qualitative role groups
                verb_pairs = []
                for d in all_data:
                    if d["word_order"] != wo:
                        continue
                    for qr in d["qual_rows"]:
                        if qr["analysis_group"] == "by_role_group" and qr["key"] == "verb":
                            verb_pairs.append((qr["mean_bits"], qr["n"]))
                v_mean = weighted_mean([p[0] for p in verb_pairs], [p[1] for p in verb_pairs]) if verb_pairs else float("nan")

                s_means_wo.append(s_mean)
                o_means_wo.append(o_mean)
                v_means_wo.append(v_mean)

            # Save the underlying table for direct comparison with FLORES analysis
            with open(out_dir / "svo_roles_by_word_order.csv", "w") as f:
                f.write("symbol,word_order,mean_bits\n")
                for i, wo in enumerate(WORD_ORDERS):
                    if s_means_wo[i] == s_means_wo[i]:
                        f.write(f"S,{wo},{fmt(s_means_wo[i])}\n")
                    if v_means_wo[i] == v_means_wo[i]:
                        f.write(f"V,{wo},{fmt(v_means_wo[i])}\n")
                    if o_means_wo[i] == o_means_wo[i]:
                        f.write(f"O,{wo},{fmt(o_means_wo[i])}\n")

            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.plot(WORD_ORDERS, s_means_wo, marker="o", linewidth=2, color="#1f77b4", label="S (subject)")
            ax.plot(WORD_ORDERS, v_means_wo, marker="o", linewidth=2, color="#2ca02c", label="V (verb)")
            ax.plot(WORD_ORDERS, o_means_wo, marker="o", linewidth=2, color="#d62728", label="O (object)")
            ax.set_ylabel("Mean surprisal (bits)")
            ax.set_xlabel("Base word order")
            ax.set_title("S / V / O surprisal across base word orders")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            _save(fig, "plot_svo_comparison_by_word_order.png")

            # --- Plot 6: Full grammar ranking (scatter: word order colored) ---
            fig, ax = plt.subplots(figsize=(14, 5))
            wo_colors = {wo: c for wo, c in zip(WORD_ORDERS, colors)}
            for wo in WORD_ORDERS:
                indices = [i for i, d in enumerate(grammar_ranking) if d["word_order"] == wo]
                vals = [grammar_ranking[i]["overall_mean"] for i in indices]
                ax.scatter(indices, vals, c=wo_colors[wo], label=wo, s=20, alpha=0.7)
            ax.set_xlabel("Rank (1 = easiest)")
            ax.set_ylabel("Mean surprisal (bits)")
            ax.set_title("All 192 grammars ranked by mean surprisal")
            ax.legend(title="Word order", loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            _save(fig, "plot_full_grammar_ranking.png")

            # --- Plot 7: Switch effects per word order (grouped bar) ---
            fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
            for idx, sname in enumerate(SWITCH_NAMES):
                ax = axes[idx]
                diffs = []
                for wo in WORD_ORDERS:
                    match = [r for r in switch_wo_effects if r["switch"] == sname and r["word_order"] == wo]
                    diffs.append(match[0]["diff"] if match else 0)
                bar_colors = ["#c44e52" if d > 0 else "#55a868" for d in diffs]
                ax.bar(WORD_ORDERS, diffs, color=bar_colors, alpha=0.8)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_title(sname)
                ax.grid(True, alpha=0.3, axis="y")
                if idx == 0:
                    ax.set_ylabel("Δ surprisal (bits)")
            fig.suptitle("Switch effects by word order", fontsize=13, y=1.02)
            plt.tight_layout()
            _save(fig, "plot_switch_effects_by_word_order.png")

            # --- Plot 8: Marker surprisal by word order ---
            marker_roles = ["Subj_marker", "Obj_marker", "Rel_marker", "Comp_marker"]
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(WORD_ORDERS))
            w = 0.18
            marker_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
            for mi, mrole in enumerate(marker_roles):
                means_per_wo = []
                for wo in WORD_ORDERS:
                    pairs = role_wo[mrole][wo]
                    if pairs:
                        means_per_wo.append(weighted_mean([p[0] for p in pairs], [p[1] for p in pairs]))
                    else:
                        means_per_wo.append(0)
                ax.bar(x + mi * w - 1.5 * w, means_per_wo, w, label=mrole,
                       color=marker_colors[mi], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(WORD_ORDERS)
            ax.set_ylabel("Mean surprisal (bits)")
            ax.set_title("Marker surprisal by word order")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            _save(fig, "plot_marker_surprisal_by_word_order.png")

            # --- Plot 9: Verb surprisal (main vs rel) by word order ---
            fig, ax = plt.subplots(figsize=(8, 5))
            main_v_wo = []
            rel_v_wo = []
            for wo in WORD_ORDERS:
                mp, rp = [], []
                for d in all_data:
                    if d["word_order"] != wo:
                        continue
                    for qr in d["qual_rows"]:
                        if qr["analysis_group"] == "verb_rel_vs_main_clause":
                            if qr["key"] == "main_clause_verb":
                                mp.append((qr["mean_bits"], qr["n"]))
                            elif qr["key"] == "rel_clause_verb":
                                rp.append((qr["mean_bits"], qr["n"]))
                main_v_wo.append(weighted_mean([p[0] for p in mp], [p[1] for p in mp]) if mp else 0)
                rel_v_wo.append(weighted_mean([p[0] for p in rp], [p[1] for p in rp]) if rp else 0)
            x = np.arange(len(WORD_ORDERS))
            w = 0.35
            ax.bar(x - w / 2, main_v_wo, w, label="Main clause verb", color="#6baed6", alpha=0.8)
            ax.bar(x + w / 2, rel_v_wo, w, label="Rel clause verb", color="#fd8d3c", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(WORD_ORDERS)
            ax.set_ylabel("Mean surprisal (bits)")
            ax.set_title("Main vs relative clause verb surprisal by word order")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            _save(fig, "plot_main_vs_rel_verb_by_word_order.png")

            print(f"\n All plots saved to {out_dir}")
        except Exception as e:
            print(f"Warning: Plotting failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"All outputs saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
