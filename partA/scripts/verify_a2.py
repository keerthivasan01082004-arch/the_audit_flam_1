"""
Independent re-derivation of fertility.py's numbers using the offline
GPT-2 BPE reconstruction (gpt2_bpe_offline.py), so the A2 evidence can be
checked without relying on network-fetched tiktoken/HF tokenizers.

Mirrors fertility.py's analyze() logic exactly (including the lower() and
split(" ") calls under test) so isolating one line change means only that
behavior differs from the baseline.
"""

import unicodedata
from gpt2_bpe_offline import GPT2BPE

enc = GPT2BPE("merges.txt")


def read_lines(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def analyze(lines, *, do_lower, split_fn):
    fert, tpc = [], []
    for line in lines:
        if do_lower:
            line = line.lower()
        tokens = enc.encode(line)
        words = split_fn(line)
        chars = len(line)
        fert.append(len(tokens) / len(words))
        tpc.append(len(tokens) / chars)
    n = len(fert)
    return sum(fert) / n, sum(tpc) / n


eng = read_lines("corpus_sample/eng_sample.txt")
hin = read_lines("corpus_sample/hin_sample.txt")

print("=== baseline: exact fertility.py behavior (lower=True, split(' ')) ===")
for name, lines in [("eng", eng), ("hin", hin)]:
    f, t = analyze(lines, do_lower=True, split_fn=lambda s: s.split(" "))
    print(f"{name}: fertility={f:.4f}  tok/char={t:.4f}")

print("\n=== bug #1 isolated: split(' ') vs split() (lower=True both) ===")
for name, lines in [("eng", eng), ("hin", hin)]:
    f_bug, _ = analyze(lines, do_lower=True, split_fn=lambda s: s.split(" "))
    f_fix, _ = analyze(lines, do_lower=True, split_fn=lambda s: s.split())
    delta = (f_fix - f_bug) / f_bug * 100
    print(f"{name}: buggy={f_bug:.4f}  fixed={f_fix:.4f}  delta={delta:+.2f}%")

print("\n=== bug #2 isolated: lower() vs no lower() (split(' ') both) ===")
for name, lines in [("eng", eng), ("hin", hin)]:
    f_with, _ = analyze(lines, do_lower=True, split_fn=lambda s: s.split(" "))
    f_without, _ = analyze(lines, do_lower=False, split_fn=lambda s: s.split(" "))
    delta = (f_with - f_without) / f_without * 100
    print(f"{name}: with_lower={f_with:.4f}  without={f_without:.4f}  delta={delta:+.2f}%")

print("\n=== both bugs fixed: eng/hin ratio, buggy vs fixed ===")
f_eng_bug, _ = analyze(eng, do_lower=True, split_fn=lambda s: s.split(" "))
f_hin_bug, _ = analyze(hin, do_lower=True, split_fn=lambda s: s.split(" "))
f_eng_fix, _ = analyze(eng, do_lower=False, split_fn=lambda s: s.split())
f_hin_fix, _ = analyze(hin, do_lower=False, split_fn=lambda s: s.split())
print(f"buggy ratio (report's 5.89x claim): {f_hin_bug/f_eng_bug:.2f}x  (eng={f_eng_bug:.4f}, hin={f_hin_bug:.4f})")
print(f"both-fixed ratio:                   {f_hin_fix/f_eng_fix:.2f}x  (eng={f_eng_fix:.4f}, hin={f_hin_fix:.4f})")

print("\n=== double-space line check ===")
for name, path in [("eng", "corpus_sample/eng_sample.txt"),
                     ("hin", "corpus_sample/hin_sample.txt")]:
    lines = read_lines(path)
    for i, l in enumerate(lines):
        if "  " in l:
            print(f"{name}[{i}]: split(' ')->{len(l.split(' '))} words, split()->{len(l.split())} words")
