#!/usr/bin/env python3
"""Structural verification of the three tokenizer vocab files.
No tokenization happens here -- this only checks the files are what
they claim to be, before any experiment touches them."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_DIR = os.path.join(HERE, "vocab_files")

report = []

# --- GPT-2 vocab.json ---
v = json.load(open(f'{VOCAB_DIR}/vocab.json', encoding='utf-8'))
report.append(f"vocab.json entries: {len(v)} (expected 50257 for GPT-2)")
report.append(f"vocab.json has <|endoftext|>: {'<|endoftext|>' in v} (id={v.get('<|endoftext|>')})")
gspace = sum(1 for k in v if k.startswith("\u0120"))
report.append(f"vocab.json tokens with byte-level space marker U+0120: {gspace} (nonzero expected)")

# --- GPT-2 merges.txt ---
with open(f'{VOCAB_DIR}/merges.txt', encoding='utf-8') as f:
    lines = [l.rstrip("\n") for l in f]
report.append(f"merges.txt first line: {lines[0]!r} (expected version header)")
report.append(f"merges.txt merge rule count: {len(lines)-1} (expected 50000)")

# --- mBERT vocab.txt ---
with open(f'{VOCAB_DIR}/vocab.txt', encoding='utf-8') as f:
    bert_vocab = [l.rstrip("\n") for l in f]
report.append(f"vocab.txt entries: {len(bert_vocab)} (expected 119547 for bert-base-multilingual-cased)")
specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
report.append(f"vocab.txt special tokens present: {all(s in bert_vocab for s in specials)}")

def script_count(vocab, lo, hi):
    return sum(1 for tok in vocab if any(lo <= ord(c) <= hi for c in tok))

deva = script_count(bert_vocab, 0x0900, 0x097F)
tam = script_count(bert_vocab, 0x0B80, 0x0BFF)
kan = script_count(bert_vocab, 0x0C80, 0x0CFF)
latin = script_count(bert_vocab, 0x0041, 0x007A)
report.append(f"vocab.txt tokens containing Devanagari: {deva}")
report.append(f"vocab.txt tokens containing Tamil: {tam}")
report.append(f"vocab.txt tokens containing Kannada: {kan}")
report.append(f"vocab.txt tokens containing basic Latin: {latin}")

for line in report:
    print(line)

with open(os.path.join(HERE, 'vocab_verification.txt'), 'w', encoding='utf-8') as f:
    f.write("\n".join(report) + "\n")
