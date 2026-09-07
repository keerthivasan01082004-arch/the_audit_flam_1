# A3 — Corrected Multilingual Analysis: Methodology & Status

**Status: COMPLETED.** A3 was executed end-to-end using the supplied
GPT-2 `vocab.json` + `merges.txt` and `bert-base-multilingual-cased`
`vocab.txt`, via pure-Python reimplementations of byte-level BPE and
WordPiece (`tokenizers_impl.py`). Real, measured results for all 31
aligned units x 4 languages x 2 tokenizers are in `raw_results.csv` and
`summary.csv`; the write-up is in `findings.md`, with the full chronology
(including one post-execution correction) in `NOTEBOOK.md`.

This document is kept in three parts, in the order things actually
happened:
1. **Historical — pre-execution blocker.** The environment genuinely had
   no working tokenizer at the start of this phase (no network, no
   installable packages, no vocab files on disk). Preserved below exactly
   as originally written, because it's a real part of the record, not
   because it describes the current state.
2. **Historical — design drafted while blocked.** The experiment design
   (tokenizers, denominators, formulas) written before any vocab file
   existed in this sandbox, so execution could start immediately once
   unblocked. Preserved as originally written; **the "4 denominators"
   design below includes tokens/grapheme-cluster, which was later dropped
   — see part 3.**
3. **Actual completed experiment**, including a post-execution correction
   from A3 review (grapheme-cluster denominator removed from official
   results). This is the part that reflects what was actually reported.

---

## Part 1 (HISTORICAL — superseded): Pre-execution blocker

*(Everything in this part describes the state of the sandbox before any
tokenizer file was uploaded. It is retained verbatim as the original
record. It does not describe the current status — see Part 3.)*

**Status at the time: BLOCKED on tokenizer availability. No token-count
results in this document are fabricated or estimated — none exist yet.
What follows is the verified blocker, the non-tokenizer groundwork
completed, and the exact design ready to execute the moment a tokenizer
path is unblocked.**

### 1. Environment check (run first, per the assignment's evidence standard)

Command: `python3 partA/corrected_analysis/precheck.py`

Result (MEASURED):
```
=== Package availability ===
tiktoken        MISSING
transformers    MISSING
sentencepiece   MISSING
regex           MISSING

=== Network ===
curl https://huggingface.co -> HTTP 403

=== Local vocab/merges files on disk (excluding uploads) ===
(none found)
```

This container is a fresh instance (same as Phase 2's "fresh container" note
in NOTEBOOK.md) — it does not have the `tiktoken`/GPT-2 files a prior
session in this thread reported using, `pip install` cannot reach PyPI
(HTTP 403, same `host_not_allowed`-style block seen in Phase 2 for FLORES),
and no vocab/merges files exist anywhere on disk outside the uploads folder.

**This is a hard blocker for A3 as specified**, which requires "two real
tokenizer implementations." I am not going to approximate GPT-2's ~50,000
merge rules from memory or simulate tokenizer output — that would be exactly
the kind of fabricated evidence this assignment is designed to catch.

### 2. What I did instead of stopping entirely

Two of the four candidate denominators in the brief — tokens/whitespace-word
and tokens/UTF-8-byte — need a tokenizer for the numerator, but the
denominator side (word counts, byte counts) can be computed from the Phase 2
corpus right now, with no tokenizer involved. I ran that:

Command: `python3 partA/corrected_analysis/nontokenizer_stats.py`
Output: `partA/corrected_analysis/nontokenizer_surface_stats.csv` (31 rows,
per-unit whitespace-word / UTF-8-byte / code-point counts for all 4
languages).

Aggregate (MEASURED, no tokenizer involved):

| lang | total words | total UTF-8 bytes | total code points | bytes/word | codepoints/word |
|---|---|---|---|---|---|
| eng | 1681 | 10,251 | 10,239 | 6.10 | 6.09 |
| hin | 1943 | 27,121 | 10,381 | 13.96 | 5.34 |
| tam | 1141 | 35,532 | 12,714 | 31.14 | 11.14 |
| kan | 1015 | 27,935 | 10,069 | 27.52 | 9.92 |

INTERPRETATION (DERIVED, not tokenizer evidence): Hindi/Tamil/Kannada need
2.3x–5.1x more UTF-8 bytes per whitespace-word than English. This is a pure
UTF-8 encoding fact (Devanagari, Tamil, and Kannada code points sit outside
the single-byte ASCII range, so every character costs 3 bytes in UTF-8) and
tells us nothing about any tokenizer yet — it's the "text representation"
layer (item B in the assignment's A/B/C/D distinction), not tokenizer
behavior (A). I'm surfacing it now because it's the natural pre-registered
hypothesis this analysis should try to falsify: **if the eventual tokenizer
gap tracks this byte-expansion ratio closely, that's evidence for "script
encoding cost," not "tokenizer inefficiency."** If the gap is much larger or
smaller than these ratios, or varies a lot between two tokenizers on the
identical text, that points to tokenizer vocabulary/training effects
instead — which is exactly the Phase 1 gpt2-vs-o200k_base finding (5.89x vs
1.32x on the same text) that this whole phase is supposed to check for
replication.

### 3. What's needed to unblock (two concrete options)

**Option A — upload the vocab files.** Both algorithms below are simple
enough to implement correctly in pure Python (no `pip install` needed) if
given the raw vocabulary/merge files, which are plain text/JSON and small:
- GPT-2 (Tokenizer 1, reproduces the Phase 1/REPORT_v0 baseline): needs
  `encoder.json` (~1MB) and `vocab.bpe` (~446KB), the same two files the
  prior session in this thread said it had.
- A WordPiece Indic-aware candidate (Tokenizer 2): `bert-base-multilingual-cased`'s
  `vocab.txt` (~1MB, plain text, one subword per line). WordPiece's
  greedy-longest-match algorithm is straightforward to reimplement exactly
  and mBERT's vocab includes Devanagari/Tamil/Kannada subwords, satisfying
  the brief's "genuinely multilingual" requirement — though its coverage of
  Indic scripts is known to be comparatively sparse versus Latin scripts,
  which I'll verify directly (vocab.txt lets me count Devanagari/Tamil/
  Kannada entries) rather than assert.
  (SentencePiece-based options like XLM-R need the binary `.model` protobuf,
  which isn't practically parseable without the `sentencepiece` package, so
  I'm not proposing XLM-R unless network access is restored instead.)

**Option B — restore network access for this session**, so `pip install
tiktoken transformers` succeeds and the standard libraries can be used
directly.

Either unblocks A3 within the same run; I don't need both.

*(Resolution: you chose Option A and uploaded `vocab.json`, `merges.txt`,
`vocab.txt`. See Part 3.)*

## Part 2 (HISTORICAL — superseded): Design drafted while blocked

*(Written before any vocab file existed in this sandbox, so execution
could start immediately once unblocked. The 4-denominator design below
includes tokens/grapheme-cluster — dropped after execution; see Part 3
for why and for the actual, official 2-denominator set that was reported.)*

### Tokenizers
1. **GPT-2** (`tiktoken.get_encoding("gpt2")` or a from-scratch BPE encoder
   built from `encoder.json`+`vocab.bpe`) — reproduces the Phase 1/
   REPORT_v0 baseline exactly, on the new corpus.
2. **bert-base-multilingual-cased WordPiece** (or XLM-R if Option B) —
   genuinely multilingual, trained with Devanagari/Tamil/Kannada data in
   its pretraining corpus, unlike GPT-2 which is trained overwhelmingly on
   English/Latin-script web text.

### Denominators (four planned; only two were used in the official results — see Part 3)
| Denominator | What it measures | What it controls for | Serving-cost relevance |
|---|---|---|---|
| tokens / whitespace word | Phase 1's original metric | Nothing about script; whitespace-word boundaries are themselves not comparable across languages with different morphology | Weak — "word" isn't the unit GPUs bill on |
| tokens / UTF-8 byte | tokens per unit of raw storage/transmission size | Removes "word" as a fuzzy cross-lingual unit; byte count is objective and language-agnostic | Weak-to-moderate — bytes correlate with but aren't the same as tokens, which is what's actually billed/computed on |
| tokens / grapheme cluster *(planned, not used — see Part 3)* | tokens per user-perceived character | Corrects for Unicode combining marks (e.g. Devanagari conjuncts) that inflate raw code-point counts without inflating perceived text length | Moderate — closer to "how much text did the user actually write," useful for UX-side framing, not cost |
| tokens / aligned unit (article) | tokens per fixed piece of content | Controls for the amount of *content*, holding meaning roughly constant | Strongest candidate for serving cost, since GPU time and KV-cache footprint scale with token count per request, and "per request" is unit-like |

The grapheme-cluster denominator needs a real UAX #29 segmenter (Python's
`regex` module's `\X`, or PyICU) — `unicodedata` alone only gives code
points, and the brief explicitly forbids calling code points grapheme
clusters. `regex` is also unavailable in this container (see Part 1), so
this denominator was blocked by the same network issue at design time —
and, as it turned out, stayed unavailable even after Option A resolved the
other two tokenizers' blocker. See Part 3 for the resulting decision.

### Formulas
- fertility (tok/word) = `tokens_in_unit / whitespace_words_in_unit`, averaged per language (mean of per-unit ratios, matching Phase 1's method, plus median and range for robustness — Phase 1 only reported the mean)
- tok/byte = `tokens_in_unit / utf8_bytes_in_unit`
- tok/grapheme = `tokens_in_unit / grapheme_clusters_in_unit` *(planned, not used — see Part 3)*
- tok/unit = `tokens_in_unit` directly (no division — this is the raw per-request token count)
- cross-language ratio = `metric[lang] / metric[eng]` for each metric, each tokenizer

### What this design targeted
The six comparison questions in the brief (does the 5.89x/6x claim
survive; tokenizer-dependence; denominator-dependence; do Tamil/Kannada
match Hindi; is "script not tokenizer" supported; what can be inferred
about cost) — answered in `findings.md` using real executed results.

## Part 3: Actual completed experiment

**Tokenizers used, exactly as planned in Part 2:** GPT-2 byte-level BPE
(from the uploaded `vocab.json` + `merges.txt`) and bert-base-multilingual-cased
WordPiece (from the uploaded `vocab.txt`), both implemented in pure Python
in `tokenizers_impl.py` and structurally verified before use
(`verify_vocab_files.py` → `vocab_verification.txt`: 50,257 GPT-2 vocab
entries, 50,000 merges, 119,547 mBERT vocab entries, all matching known
reference sizes).

**Denominators actually reported — two, not four:** tokens/whitespace-word
and tokens/UTF-8-byte. This is a deliberate, post-execution correction
from A3 review: the tokens/grapheme-cluster function that got implemented
(`tokenizers_impl.grapheme_clusters_gb9`) turned out to only cover one
UAX #29 rule (GB9, combining-mark attachment), not the full Unicode
grapheme-cluster specification the brief requires for that denominator.
Rather than report a metric that doesn't meet the stated bar, it was
excluded from `raw_results.csv`/`summary.csv` and the function was
relabeled exploratory/unused in its own module. Tokens/whitespace-word and
tokens/UTF-8-byte on their own already satisfy the assignment's ">=2
denominators" requirement, so nothing about the requirement is
unsatisfied by dropping the third.

**Where the real numbers live:** `raw_results.csv` (248 rows: 31 units x
4 languages x 2 tokenizers) and `summary.csv` (8 rows: 4 languages x 2
tokenizers, with mean/median/min/max/stdev and English-relative ratios).
**Where the write-up lives:** `findings.md` (six findings plus the
recommended "input tokens per request" metric, with its revision log at
the bottom listing every A3-review correction applied). **Where the full
chronology lives, including this correction:** `NOTEBOOK.md`'s "Phase 3
continued — A3 executed" and "Phase 3 — A3 review corrections" sections.

### Reproducibility
```
python3 partA/corrected_analysis/verify_vocab_files.py
python3 partA/corrected_analysis/corrected_analysis.py
```
Inputs: `vocab_files/vocab.json`, `vocab_files/merges.txt`,
`vocab_files/vocab.txt` (packaged in this submission), plus the Phase 2
corpus under `partA/corpus/processed/` (unchanged). No package installs
required — pure Python 3.12 standard library only. Outputs:
`raw_results.csv` (248 rows), `summary.csv` (8 rows),
`vocab_verification.txt`.

### Limitations carried forward
Same corpus limitations as Phase 2 (31 units, single formal-legal domain,
article-level alignment) apply to every number reported. Only two
tokenizers were tested; a SentencePiece-based option (e.g. XLM-R) was not,
because its binary vocab format isn't practically parseable without the
`sentencepiece` package, which isn't installable here. The
grapheme-cluster denominator was not reported, for the reason given above.
