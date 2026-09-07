# NOTEBOOK

> Note on this file: Phase 1 (repo setup, A2 script-bug audit, Part B
> capacity analysis) was completed in a prior session and its findings were
> reported in full there (evidence table matching every claim to a command
> and result). This sandbox instance does not have Phase 1's script/output
> files physically present, and per instruction ("Do NOT redo Phase 1"),
> they were not regenerated here. This notebook starts fresh at Phase 2 —
> A1; ask if you'd like Phase 1's files rebuilt into this same repo.

## Phase 2 — A1 Multilingual Corpus

### 1. Initial hypothesis

Phase 1 found a large English/Hindi tokenizer-fertility gap on a 10-sentence
smoke-test sample, and showed the gap shrinks drastically under a
different tokenizer (gpt2 → o200k_base: 5.89x → 1.32x). A1's job is to get
real English + Hindi + two Dravidian languages (Tamil, Kannada) text so A3
can check whether the same pattern holds outside that toy sample — not to
re-derive Phase 1's numbers.

### 2. Corpus options investigated, in order tried

**Attempt 1 — FLORES-200 / FLORES+ (assignment's stated first choice).**

```
git clone --depth 1 https://github.com/facebookresearch/flores.git
git clone --depth 1 https://github.com/openlanguagedata/flores.git
```
Both cloned successfully (GitHub itself is reachable), but neither repo
contains sentence data:
- `openlanguagedata/flores` (current FLORES+ maintainer) is a single
  README pointing to `https://huggingface.co/datasets/openlanguagedata/flores_plus`.
- `facebookresearch/flores`'s `flores200/README.md` points to a tinyurl
  redirect and to HuggingFace for the actual data; the repo's only
  committed data archive is `previous_releases/floresv1/data/flores_test_sets.tgz`,
  which is FLoRes v1 (2019) and covers Nepali/Sinhala/Khmer/Pashto↔English
  only — not our language set.

Tested reachability of the actual download endpoints directly:
```
curl -sSL -o /dev/null -w "%{http_code}" https://tinyurl.com/flores200dataset
  -> 403
curl -sSL -o /dev/null -w "%{http_code}" https://huggingface.co/datasets/facebook/flores
  -> 403
curl -sSL -o /dev/null -w "%{http_code}" https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz
  -> 403
```
Response header on all three: `x-deny-reason: host_not_allowed`. This is a
network egress restriction in this sandbox, not a "file doesn't exist"
error — confirmed dead end for FLORES-200/FLORES+ **from this environment**.
(The user has normal internet access and could still supply these files
directly; that option was not needed once Attempt 3 succeeded below, but
remains open if a sentence-level corpus is preferred over the
article-level one actually used.)

**Attempt 2 — NLTK's bundled `udhr` corpus.**

```
curl -sSL -o udhr.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr.zip
  -> 200, 1,170,177 bytes
unzip udhr.zip
```
This succeeded and contains `English-Latin1`, `Hindi-UTF8`, `Tamil-UTF8`,
`Kannada-UTF8` (UDHR translations). Inspecting file sizes, all five
UDHR files examined are ~9,999–10,000 bytes exactly — a red flag. Decoding
each and finding where clean decoding stops:

| File | Size | Decodes cleanly through |
|---|---|---|
| English-Latin1 | 10,000 B | Article 29 of 30 |
| Hindi-UTF8 | 10,000 B | mid-Article 7 |
| Tamil-UTF8 | 9,999 B | mid-Article 2 (decode error at the cut point) |
| Kannada-UTF8 | 9,999 B | mid-Article 8 |

This is a fixed ~10 KB truncation cap baked into this specific packaging
of the UDHR-in-Unicode data (non-Latin scripts use more UTF-8 bytes per
character, so the same byte cap keeps far less actual content). Overlap
across all four languages where every one has a *complete* article would
be Articles 1 only (Tamil cuts off inside Article 2) — far too little to
build a corpus from. Rejected.

**Attempt 3 — Full upstream source of the same UDHR data.**

A unicode.org mailing list thread (found via web search) pointed to
`https://github.com/eric-muller/udhr` as the canonical source repo behind
the NLTK package (and the official unicode.org/udhr site), and separately
noted OHCHR's own PDF for Hindi is a scanned image with no extractable
text — ruling out going straight to OHCHR's PDFs as a fix.

```
git clone --depth 1 https://github.com/eric-muller/udhr.git
```
Succeeded. `data/udhr/udhr_eng.xml`, `udhr_hin.xml`, `udhr_tam.xml`,
`udhr_kan.xml` exist and are 16–43 KB each (not capped at 10 KB). Parsed
each with `xml.etree.ElementTree` and confirmed via
`root.findall('u:article')` that all four contain exactly articles
1 through 30 (no gaps, no extras) and one `<preamble>` each. No paragraph
inside any article/preamble has nested child elements, so plain-text
extraction is complete and lossless. This became the corpus source.
Commit pinned: `588b3f4b2d0467aff54842a4b926551b69d5a66a`.

### 3. Why this corpus was chosen

It is the only option found that is (a) actually reachable from this
environment, (b) verifiably complete (not silently truncated) in all four
required languages, and (c) has documented provenance back to OHCHR for
each language (correction history with named contributors, in
`data/status/status_<lang>.xml` of the same repo). It is not FLORES, and
its limitations (small size, single formal-register document, non-uniform
translation provenance, article-level not sentence-level alignment) are
documented in full in `partA/corpus/README.md`.

### 4. Preparation

```
python3 partA/scripts/prepare_corpus.py --source-dir partA/corpus/raw --out-dir partA/corpus/processed
```
Output:
```
wrote partA/corpus/processed/eng.txt (31 lines)
wrote partA/corpus/processed/hin.txt (31 lines)
wrote partA/corpus/processed/tam.txt (31 lines)
wrote partA/corpus/processed/kan.txt (31 lines)
wrote partA/corpus/processed/manifest.tsv
```
31 = 1 preamble + 30 articles, identical for every language (checked
structurally before writing — the script hard-fails if any language is
missing a unit or an article number doesn't run exactly 1–30).

### 5. Actual corpus size

31 aligned units × 4 languages. Per-language character/word counts:

| Language | Units | Total chars | Total whitespace words |
|---|---|---|---|
| eng | 31 | 10,239 | 1,681 |
| hin | 31 | 10,381 | 1,943 |
| tam | 31 | 12,714 | 1,141 |
| kan | 31 | 10,069 | 1,015 |

### 6. Validation results

```
python3 partA/experiments/validate_corpus.py
```
All 18 checks PASSED (file existence ×4, UTF-8 clean round-trip ×4, equal
line counts across languages, no empty lines ×4, no duplicate lines ×4,
manifest line-count match, manifest unit-id ordering match). Full output
saved at `partA/experiments/validation_output.txt`.

### 7. Preprocessing decisions

See `partA/corpus/README.md`, Preprocessing section — paragraph-join with
a single space, newline collapsing, and whitespace stripping only. No
lowercasing, no Unicode normalization, no tokenization, deliberately (this
corpus should reach A3 close to raw, unlike Phase 1's script which baked
`.lower()` into fertility measurement).

### 8. Limitations

See `partA/corpus/README.md`, Limitations section in full. Short version:
small (31 units vs. FLORES-200's ~1,012), single formal-legal-document
domain, non-uniform translation provenance across languages, and
article-level (not sentence-level) alignment.

### 9. What this corpus can and cannot establish

**Can:** whether Phase 1's English/Hindi fertility gap (and its
tokenizer-dependence) replicates on real, independently-sourced,
human-translated text, and whether Tamil/Kannada show a similar or
different pattern.

**Cannot:** the magnitude of the gap on production-like conversational
text, sentence-level fertility variance, or generalization beyond formal
legal prose.

---

## Phase 5 — repo audit and repair (this session)

**Hypothesis:** the repo as it stood (multiple prior AI sessions across
Phases 1–4, one flat "final" zip plus a stale nested zip inside it) is
actually complete and consistent.

**What I ran:** inventoried every file across both zips, diffed the
duplicate memo drafts, and cross-checked every numeric claim in
`findings.md` and `partB/analysis.md` against the raw starter-kit files.

**Result — it wasn't consistent:**
- `AI_USAGE.md` didn't exist anywhere. Required deliverable, 0% done.
- `partB/calculations_final.py` was 0 bytes, and `audit_experiments.py`
  (cited in `findings.md` as "the runnable evidence") didn't exist in
  either zip. The A2 and B1–B3 write-ups had real, correct numbers with
  no runnable code behind them anymore — a bad place to be for a defense
  built around "run this input I'm about to paste."
- Two files disagreed on the headline B1 answer: `EARLIER_ATTEMPT_phase4_do_not_use.md`
  (formerly `final_phase_4_roconcila.md`) says 46 concurrent sequences;
  `partB/analysis.md` says 25. Root cause: the Phase 4 calc never
  subtracted model-weights memory (~8.4 GB) from the usable pool before
  dividing by KV-cache bytes/token — it only subtracted the non-KV
  overhead. `analysis.md`'s 25 is correct and is what the corrected
  `partB/calculations.py` (rebuilt this session) reproduces.
- `memo-1.md` / `fianl_memeo_1.md` are the *Part C* memo, not another A4
  draft — the filenames made that easy to miss.
- One real zip-inside-a-zip: `phase_3zipfinalfinal.zip` held the actual
  `partA/corpus/` and `partA/corrected_analysis/` files, never extracted
  into the final structure.

**Independent re-verification done this session (see `partA/scripts/`):**
reconstructed the GPT-2 byte-level BPE tokenizer offline from the local
`merges.txt` (no network available in this environment) and re-ran the
exact `fertility.py` algorithm against it. Reproduced the report's
baseline (eng 1.2652≈1.27, hin 7.4485≈7.45) and `findings.md`'s claimed
deltas almost exactly (split-fix: +1.41%/+2.01% match exactly; lower-fix:
+2.92% vs. the ~2.88% previously reported — trivial rounding, not a real
discrepancy). This closes the "no runnable evidence" gap for A2.

**Revision:** kept `analysis.md` (25 sequences) as the answer, rebuilt
`partB/calculations.py` from `model_spec.md` + `bench_log.csv` to
reproduce it plus the B3 goodput numbers, and kept the old Phase 4 file
around (renamed, not deleted) as the documented dead end rather than
scrubbing it — the wrong-then-corrected arc is real signal, not noise.

**Next:** verify the `mBERT` numbers in `partA/corrected_analysis/` the
same way this session verified the `gpt2`/`o200k_base` ones (not done
here — no network access to fetch mBERT's tokenizer in this environment).
Also: `AI_USAGE.md` now exists as a first draft — needs a human pass to
add anything from Phases 1–4 that this session couldn't see.
