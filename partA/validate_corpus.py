#!/usr/bin/env python3
"""
validate_corpus.py — Phase 2, Part A1 validation

Runs the checks required before the corpus can be used in A3:
  1. Expected language files exist
  2. Same number of lines (aligned units) across all four languages
  3. No empty lines in any file
  4. No duplicate lines within a language file (would indicate a
     copy/merge bug in prepare_corpus.py, not a property of the source)
  5. manifest.tsv line count matches the corpus line count, and line i
     of every language file corresponds to the same unit_id
  6. Unicode is preserved (each file decodes cleanly as UTF-8, and
     round-trips through encode/decode unchanged)

Prints PASS/FAIL for each check and exits non-zero on any failure.
"""

import sys
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "processed"
LANGS = ["eng", "hin", "tam", "kan"]

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main():
    # 1. files exist
    paths = {}
    for lang in LANGS:
        p = CORPUS_DIR / f"{lang}.txt"
        check(f"file exists: {p.name}", p.exists())
        paths[lang] = p
    manifest_path = CORPUS_DIR / "manifest.tsv"
    check("manifest.tsv exists", manifest_path.exists())

    if failures:
        print("\nStopping early: required files missing.")
        sys.exit(1)

    # load with strict utf-8 decoding
    lines = {}
    for lang in LANGS:
        raw_bytes = paths[lang].read_bytes()
        try:
            text = raw_bytes.decode("utf-8")
            reencoded = text.encode("utf-8")
            check(f"utf-8 clean round-trip: {lang}.txt", reencoded == raw_bytes)
        except UnicodeDecodeError as e:
            check(f"utf-8 clean round-trip: {lang}.txt", False, str(e))
            text = ""
        file_lines = text.split("\n")
        if file_lines and file_lines[-1] == "":
            file_lines = file_lines[:-1]  # drop trailing newline artifact
        lines[lang] = file_lines

    # 2. same line count across languages
    counts = {lang: len(lines[lang]) for lang in LANGS}
    same_count = len(set(counts.values())) == 1
    check("same number of aligned lines across all 4 languages", same_count, detail=str(counts))

    # 3. no empty lines
    for lang in LANGS:
        empty_idx = [i + 1 for i, l in enumerate(lines[lang]) if not l.strip()]
        check(f"no empty lines: {lang}.txt", len(empty_idx) == 0, detail=f"empty at lines {empty_idx}" if empty_idx else "")

    # 4. no duplicate lines within a language (would indicate a real bug)
    for lang in LANGS:
        seen = {}
        dups = []
        for i, l in enumerate(lines[lang], start=1):
            if l in seen:
                dups.append((seen[l], i))
            else:
                seen[l] = i
        check(f"no duplicate lines: {lang}.txt", len(dups) == 0, detail=f"duplicate pairs {dups}" if dups else "")

    # 5. manifest alignment
    manifest_lines = manifest_path.read_text(encoding="utf-8").strip().split("\n")[1:]  # skip header
    check("manifest line count matches corpus line count",
          len(manifest_lines) == counts.get(LANGS[0], -1),
          detail=f"manifest={len(manifest_lines)} corpus={counts.get(LANGS[0])}")

    expected_unit_ids = ["preamble"] + [f"article_{i:02d}" for i in range(1, 31)]
    manifest_unit_ids = [row.split("\t")[1] for row in manifest_lines]
    check("manifest unit_id sequence matches expected preamble+article_01..30 order",
          manifest_unit_ids == expected_unit_ids)

    print()
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"RESULT: all checks PASSED. {counts[LANGS[0]]} aligned units x {len(LANGS)} languages.")


if __name__ == "__main__":
    main()
