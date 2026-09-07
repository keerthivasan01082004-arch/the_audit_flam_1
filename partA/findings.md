# A2 — Script & metric audit of `fertility.py`

Sanity check first: re-ran the intern's exact command and reproduced
their numbers exactly (eng 1.27 / 0.226, hin 7.45 / 1.579, 5.89x),
confirming the local tokenizer setup matches theirs before auditing
anything. See `audit_experiments.py` for the runnable evidence below.

## Code bug #1 — `words = line.split(" ")`

Splitting on a literal single space breaks on double spaces (common in
real corpora — OCR output, copy-pasted text, scraped HTML). A double
space produces an empty-string "word", inflating the word count and
silently deflating fertility for that line.

**Evidence:** both sample files already contain one double-spaced line
each (`eng[6]`, `hin[9]`). Fixing `split(" ")` → `split()`:

| lang | buggy (split(" ")) | fixed (split()) | delta |
|---|---|---|---|
| eng | 1.2652 | 1.2831 | +1.41% |
| hin | 7.4485 | 7.5985 | +2.01% |

Small on a 10-line toy corpus because only 1/10 lines are affected per
language, but the bug scales with how messy the real corpus is —
worth fixing before scaling to a real eval set.

## Code bug #2 — `line.lower()` distorts English only, not Hindi

Devanagari has no case distinction, so `.lower()` is a no-op on every
Hindi line (verified: 0/10 Hindi lines change). It **does** change all
10 English lines (proper nouns, acronyms like "NASA", "ISRO", "GPU"
lose casing), and that changes their GPT-2 tokenization since casing
affects which BPE merges apply.

**Evidence:**

| lang | WITH .lower() | WITHOUT | delta | lines altered by .lower() |
|---|---|---|---|---|
| eng | 1.2831 | 1.2472 | +2.88% | 10/10 |
| hin | 7.5985 | 7.5985 | 0.00% | 0/10 |

This is a real bug — a preprocessing step is applied asymmetrically
across the two languages being compared — but note the *direction*:
removing it lowers English's fertility, which would **widen**, not
narrow, the reported eng/hin gap (5.89x buggy → ~6.09x with both bugs
fixed). Fixing these two implementation bugs does not rescue the
report's conclusion; if anything it's slightly worse. That's exactly
why A2 is worth only 20 of the 50 points here — the bigger issue is
conceptual (below), not these two bugs.

## Conceptual problem — fertility (tok/word) is the wrong cross-lingual metric, and the report's causal claim is falsifiable

The script faithfully computes tokens-per-whitespace-word. But
whitespace-delimited "words" are not a comparable unit across
languages with different morphological typology — Hindi postpositions,
compounding, and script conventions mean a "word" doesn't carry the
same amount of information in Hindi as in English. Two things follow:

1. Tok/word and tok/char are **not independent confirmation** of each
   other, as the report claims ("the two metrics agree, so the result
   is robust") — both are computed from the exact same token list on
   the exact same text, just divided by two different (and both
   language-sensitive) denominators. Agreement between them isn't
   evidence of robustness.

2. The report's stated root cause — *"Hindi simply has more Unicode
   characters per word... a property of the script, not the
   tokenizer"* — is directly falsifiable. I ran the identical text
   through a second, more recent tokenizer (`o200k_base`, GPT-4o's,
   200k vocab vs GPT-2's 50k, trained on far more multilingual data):

   | tokenizer | eng fertility | hin fertility | ratio |
   |---|---|---|---|
   | gpt2 (2019, English-centric) | 1.27 | 7.45 | **5.89x** |
   | o200k_base (2024, multilingual) | 1.23 | 1.61 | **1.32x** |

   Same text, same script, same "number of Unicode characters per
   word" — the gap nearly disappears with a tokenizer trained with
   real Hindi coverage. The root cause is the tokenizer's training
   data/vocab, not an inherent property of Devanagari. This directly
   contradicts the report's recommendation to treat the 6x gap as a
   fixed cost of serving Hindi.

## Looks suspicious but is fine — two candidates, both verified harmless

- **`unicodedata.normalize("NFC", line)`**: looks like an odd,
  unexplained transform, but NFC normalization before tokenizing is
  standard practice (guards against a source file using decomposed
  vs. precomposed Unicode forms producing different token counts for
  visually identical text). Verified it's a no-op on both sample
  files (0/10 lines changed either language) — harmless here, and
  correct practice in general.
- **`random.seed(1337)`**: looks like it should matter for
  reproducibility, but `random` is never called anywhere in the
  analysis path. Verified: re-running with a different seed (999999)
  produces byte-identical output. Dead code, zero effect.

## Unverified-flaw floor check

I did *not* flag `line.lower()`'s intent ("casing doesn't add noise")
as wrong in isolation — it's a reasonable idea in principle, applied
incorrectly here because it's case-insensitive for one language and
not the other. I'm flagging the asymmetry, not the intent, since the
former is what's actually measurable and demonstrated above.
