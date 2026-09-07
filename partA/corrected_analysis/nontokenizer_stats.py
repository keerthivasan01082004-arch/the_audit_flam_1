#!/usr/bin/env python3
"""
nontokenizer_stats.py -- per-unit surface statistics that do NOT require
a tokenizer: whitespace-word count, UTF-8 byte count, Unicode code-point
(character) count. Computed directly from the Phase 2 corpus files.
These are real denominators (word, byte) that A3 will divide token
counts by once a tokenizer is available -- and are useful evidence
on their own about raw text expansion across languages, independent
of any tokenizer.
"""
import csv

LANGS = ["eng", "hin", "tam", "kan"]
BASE = "/mnt/user-data/uploads"

manifest = []
with open(f"{BASE}/manifest.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        line_no, unit_id = line.rstrip("\n").split("\t")
        manifest.append((int(line_no), unit_id))

lines = {}
for lang in LANGS:
    with open(f"{BASE}/{lang}.txt", encoding="utf-8") as f:
        lines[lang] = [l.rstrip("\n") for l in f]
    assert len(lines[lang]) == len(manifest), f"{lang}: line count mismatch"

rows = []
for idx, (line_no, unit_id) in enumerate(manifest):
    row = {"unit_id": unit_id}
    for lang in LANGS:
        text = lines[lang][idx]
        row[f"{lang}_whitespace_words"] = len(text.split())
        row[f"{lang}_utf8_bytes"] = len(text.encode("utf-8"))
        row[f"{lang}_codepoints"] = len(text)
    rows.append(row)

out_path = "/home/claude/work/partA/corrected_analysis/nontokenizer_surface_stats.csv"
fieldnames = ["unit_id"] + [f"{l}_{m}" for l in LANGS for m in
                            ("whitespace_words", "utf8_bytes", "codepoints")]
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

# aggregate
print(f"{'lang':<6}{'total_words':>13}{'total_bytes':>13}{'total_cp':>10}"
      f"{'bytes/word':>12}{'cp/word':>10}")
for lang in LANGS:
    tw = sum(r[f"{lang}_whitespace_words"] for r in rows)
    tb = sum(r[f"{lang}_utf8_bytes"] for r in rows)
    tc = sum(r[f"{lang}_codepoints"] for r in rows)
    print(f"{lang:<6}{tw:>13}{tb:>13}{tc:>10}{tb/tw:>12.2f}{tc/tw:>10.2f}")

print(f"\nWrote {out_path} ({len(rows)} rows)")
