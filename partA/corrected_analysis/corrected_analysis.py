#!/usr/bin/env python3
"""
corrected_analysis.py -- Phase 3 / A3: corrected multilingual tokenizer
analysis on the Phase 2 UDHR corpus (31 aligned units x 4 languages).

Tokenizers (both real, pure-Python implementations built from raw vocab
files -- see tokenizers_impl.py and methodology.md for why packages
aren't used):
  - gpt2   : byte-level BPE, from vocab.json + merges.txt
  - mbert  : WordPiece, from bert-base-multilingual-cased vocab.txt

Denominators (official, per review): tokens/whitespace-word, tokens/UTF-8
byte, and tokens/unit (raw count, no division). A tokens/grapheme-cluster
metric was explored during development but is EXCLUDED from official
results: the only grapheme segmentation implementable in this sandbox
(tokenizers_impl.grapheme_clusters_gb9) is a partial GB9-only
approximation, not a real Unicode UAX #29 implementation, and the
requirement is explicit that grapheme-cluster denominators must use a
real implementation. See tokenizers_impl.py for that function, kept for
reference/future use only -- it is not called anywhere in this script.

Every number in raw_results.csv comes from actually running these
tokenizers on the actual corpus text -- nothing here is estimated.
"""
import csv
import sys
sys.path.insert(0, '.')
from tokenizers_impl import GPT2Tokenizer, BertWordPieceTokenizer
# NOTE: grapheme_clusters_gb9 deliberately NOT imported -- it's a partial
# GB9-only approximation, not a real UAX #29 implementation, and is
# excluded from the official A3 denominators per review. See
# tokenizers_impl.py for the function itself (unused, kept for reference).

import os
HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_DIR = os.path.join(HERE, "vocab_files")
CORPUS_DIR = os.path.join(HERE, "..", "corpus", "processed")
OUT_DIR = HERE
LANGS = ["eng", "hin", "tam", "kan"]

gpt2 = GPT2Tokenizer(f"{VOCAB_DIR}/vocab.json", f"{VOCAB_DIR}/merges.txt")
mbert = BertWordPieceTokenizer(f"{VOCAB_DIR}/vocab.txt")
TOKENIZERS = {"gpt2": gpt2.encode, "mbert": mbert.encode}

manifest = []
with open(f"{CORPUS_DIR}/manifest.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        line_no, unit_id = line.rstrip("\n").split("\t")
        manifest.append((int(line_no), unit_id))

lines = {}
for lang in LANGS:
    with open(f"{CORPUS_DIR}/{lang}.txt", encoding="utf-8") as f:
        lines[lang] = [l.rstrip("\n") for l in f]
    assert len(lines[lang]) == len(manifest), f"{lang}: line/manifest mismatch"

rows = []
for idx, (line_no, unit_id) in enumerate(manifest):
    for lang in LANGS:
        text = lines[lang][idx]
        words = len(text.split())
        byte_len = len(text.encode("utf-8"))
        codepoints = len(text)  # kept for reference only; not an official denominator
        for tok_name, encode_fn in TOKENIZERS.items():
            n_tok = len(encode_fn(text))
            rows.append({
                "unit_id": unit_id,
                "lang": lang,
                "tokenizer": tok_name,
                "tokens": n_tok,
                "whitespace_words": words,
                "utf8_bytes": byte_len,
                "codepoints": codepoints,
                "tok_per_word": n_tok / words,
                "tok_per_byte": n_tok / byte_len,
            })

raw_path = f"{OUT_DIR}/raw_results.csv"
fieldnames = ["unit_id", "lang", "tokenizer", "tokens", "whitespace_words",
              "utf8_bytes", "codepoints",
              "tok_per_word", "tok_per_byte"]
with open(raw_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {raw_path}: {len(rows)} rows "
      f"({len(manifest)} units x {len(LANGS)} langs x {len(TOKENIZERS)} tokenizers)")

# ---------------- summary.csv ----------------
import statistics as stats

def agg(values):
    return {
        "mean": stats.mean(values),
        "median": stats.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": stats.stdev(values) if len(values) > 1 else 0.0,
    }

summary_rows = []
metrics = ["tok_per_word", "tok_per_byte", "tokens"]  # official A3 denominators + raw count
by_key = {}
for r in rows:
    by_key.setdefault((r["lang"], r["tokenizer"]), []).append(r)

raw_means = {}  # (lang, tok) -> {metric: unrounded mean}, used for ratios to avoid rounding-before-dividing
for (lang, tok), group in by_key.items():
    row = {"lang": lang, "tokenizer": tok, "n_units": len(group)}
    raw_means[(lang, tok)] = {}
    for m in metrics:
        vals = [g[m] for g in group]
        a = agg(vals)
        raw_means[(lang, tok)][m] = a["mean"]
        for stat_name, v in a.items():
            row[f"{m}_{stat_name}"] = round(v, 4) if isinstance(v, float) else v
    summary_rows.append(row)

# add eng-relative ratios computed from UNROUNDED means (rounded only for display)
for row in summary_rows:
    lang, tok = row["lang"], row["tokenizer"]
    for m in metrics:
        base = raw_means[("eng", tok)][m]
        row[f"{m}_ratio_vs_eng"] = round(raw_means[(lang, tok)][m] / base, 4)

summary_path = f"{OUT_DIR}/summary.csv"
summary_fields = ["lang", "tokenizer", "n_units"]
for m in metrics:
    for stat_name in ["mean", "median", "min", "max", "stdev"]:
        summary_fields.append(f"{m}_{stat_name}")
    summary_fields.append(f"{m}_ratio_vs_eng")
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=summary_fields)
    w.writeheader()
    for row in sorted(summary_rows, key=lambda r: (r["tokenizer"], r["lang"])):
        w.writerow(row)
print(f"Wrote {summary_path}: {len(summary_rows)} rows (4 langs x {len(TOKENIZERS)} tokenizers)")

# ---------------- console report ----------------
print("\n=== tok_per_word mean, by tokenizer, with ratio vs eng ===")
for tok in TOKENIZERS:
    print(f"-- {tok} --")
    for lang in LANGS:
        row = next(r for r in summary_rows if r["lang"] == lang and r["tokenizer"] == tok)
        print(f"  {lang}: mean={row['tok_per_word_mean']:.3f}  "
              f"ratio_vs_eng={row['tok_per_word_ratio_vs_eng']:.3f}x")
