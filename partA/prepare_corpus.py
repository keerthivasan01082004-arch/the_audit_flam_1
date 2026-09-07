#!/usr/bin/env python3
"""
prepare_corpus.py — Phase 2, Part A1

Builds the English / Hindi / Tamil / Kannada evaluation corpus used in A3.

SOURCE
------
"UDHR in XML" project (the canonical source behind https://unicode.org/udhr),
maintained at:
    https://github.com/eric-muller/udhr
Underlying translation text is sourced from OHCHR (UN Office of the High
Commissioner for Human Rights) official UDHR translations; see
data/status/status_<lang>.xml in that repository for the per-language
correction history and named contributors.

Commit pinned for reproducibility: 588b3f4b2d0467aff54842a4b926551b69d5a66a

WHY THIS SOURCE AND NOT FLORES-200
-----------------------------------
FLORES-200 / FLORES+ was investigated first (see NOTEBOOK.md, Phase 2 - A1).
It is distributed only via HuggingFace (huggingface.co) and a Meta blob
store (dl.fbaipublicfiles.com); both were confirmed unreachable from this
environment's network egress (HTTP 403, x-deny-reason: host_not_allowed).
The GitHub repos that once mirrored it (facebookresearch/flores,
openlanguagedata/flores) contain no committed sentence data for FLORES-200 —
only download pointers to those same blocked hosts. This is documented
with exact evidence in NOTEBOOK.md rather than asserted here.

USAGE
-----
    # Using the raw XML files already vendored in ../corpus/raw/
    python3 prepare_corpus.py --source-dir ../corpus/raw --out-dir ../corpus/processed

    # Or, with normal (unrestricted) internet access, to re-fetch from
    # scratch at the pinned commit instead of using the vendored copies:
    python3 prepare_corpus.py --fetch --out-dir ../corpus/processed
"""

import argparse
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"u": "http://efele.net/udhr"}
PINNED_COMMIT = "588b3f4b2d0467aff54842a4b926551b69d5a66a"
REPO_URL = "https://github.com/eric-muller/udhr.git"

LANGS = {
    "eng": "udhr_eng.xml",
    "hin": "udhr_hin.xml",
    "tam": "udhr_tam.xml",
    "kan": "udhr_kan.xml",
}

N_ARTICLES = 30  # UDHR has exactly 30 articles in every language, by definition


def fetch_source(dest: Path) -> Path:
    """Clone the UDHR-in-XML repo and check out the pinned commit."""
    subprocess.run(["git", "clone", REPO_URL, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", PINNED_COMMIT], check=True)
    return dest / "data" / "udhr"


def extract_units(xml_path: Path) -> dict:
    """Return {'preamble': text, 'article_01': text, ..., 'article_30': text}
    for one language file. Each value is that unit's paragraphs joined with
    a single space (documented under Preprocessing in corpus/README.md)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    units = {}

    preamble = root.find("u:preamble", NS)
    if preamble is None:
        raise ValueError(f"{xml_path}: no <preamble> element found")
    paras = preamble.findall(".//u:para", NS)
    units["preamble"] = " ".join(p.text.strip() for p in paras if p.text and p.text.strip())

    articles = root.findall("u:article", NS)
    found_numbers = set()
    for art in articles:
        num = art.get("number")
        found_numbers.add(int(num))
        paras = art.findall(".//u:para", NS)
        text = " ".join(p.text.strip() for p in paras if p.text and p.text.strip())
        units[f"article_{int(num):02d}"] = text

    expected = set(range(1, N_ARTICLES + 1))
    if found_numbers != expected:
        missing = sorted(expected - found_numbers)
        extra = sorted(found_numbers - expected)
        raise ValueError(f"{xml_path}: article numbers mismatch. missing={missing} extra={extra}")

    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=None,
                     help="Directory already containing udhr_<lang>.xml files")
    ap.add_argument("--fetch", action="store_true",
                     help="Clone the source repo at the pinned commit instead of using --source-dir")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.fetch:
        tmp = Path(tempfile.mkdtemp())
        source_dir = fetch_source(tmp)
    elif args.source_dir:
        source_dir = args.source_dir
    else:
        print("ERROR: pass --source-dir or --fetch", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_lang_units = {}
    for lang, fname in LANGS.items():
        xml_path = source_dir / fname
        if not xml_path.exists():
            print(f"ERROR: expected file not found: {xml_path}", file=sys.stderr)
            sys.exit(1)
        per_lang_units[lang] = extract_units(xml_path)

    # unit ids in a fixed, deterministic order: preamble, article_01 .. article_30
    unit_ids = ["preamble"] + [f"article_{i:02d}" for i in range(1, N_ARTICLES + 1)]

    # sanity check before writing anything: every language must have every unit,
    # and no unit may be empty for any language
    for lang, units in per_lang_units.items():
        for uid in unit_ids:
            if uid not in units:
                print(f"ERROR: {lang} missing unit {uid}", file=sys.stderr)
                sys.exit(1)
            if not units[uid].strip():
                print(f"ERROR: {lang} unit {uid} is empty after preprocessing", file=sys.stderr)
                sys.exit(1)

    # write one file per language, one line per unit, in unit_ids order
    for lang in LANGS:
        out_path = args.out_dir / f"{lang}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for uid in unit_ids:
                f.write(per_lang_units[lang][uid].replace("\n", " ").strip() + "\n")
        print(f"wrote {out_path} ({len(unit_ids)} lines)")

    # manifest mapping line number -> unit id, for traceability
    manifest_path = args.out_dir / "manifest.tsv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("line_number\tunit_id\n")
        for i, uid in enumerate(unit_ids, start=1):
            f.write(f"{i}\t{uid}\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
