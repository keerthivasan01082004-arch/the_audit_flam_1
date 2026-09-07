# AI Usage

This repo was built across several AI-assisted sessions (referred to in
NOTEBOOK.md as Phases 1-5). This file covers what's verifiable from the
current state of the repo and this session's own work. **The Phase 1-4
entries below are inferred from file evidence, not from being present for
those sessions — replace them with your own account before submitting.**

## Where AI helped
- Phases 1-4 (per NOTEBOOK.md and file timestamps): initial A2 script
  audit, A1 corpus construction from UDHR, A3 corrected multi-tokenizer
  analysis, and Part B capacity math were all AI-assisted across separate
  sessions.
- This session: audited the assembled repo for consistency, rebuilt two
  pieces of missing/broken evidence code (below), and reorganized the flat
  file dump into the structure the brief asks for.

## Where AI was wrong, or would have been if untrusted
- An earlier session's Part B capacity calculation (`partB/EARLIER_ATTEMPT_phase4_do_not_use.md`)
  computed **46** concurrent sequences by forgetting to subtract model-weights
  memory from the usable GPU memory pool — only the non-KV overhead was
  subtracted. A later session corrected this to **25** in `partB/analysis.md`,
  but the stale `run_output.txt` from the first calculation was never deleted,
  so the repo briefly contained two contradicting answers to the same
  question with no code to settle it either way (`calculations_final.py`
  had gone empty somewhere in between). This session rebuilt
  `partB/calculations.py` from the raw spec + log to settle it — 25 is
  correct, cross-checked against `bench_log.csv`'s own `kv_cache_util`
  column.
- `findings.md` (A2) cited `audit_experiments.py` as "the runnable evidence"
  for its bug-isolation numbers; that file doesn't exist in the delivered
  zip. The numbers themselves check out (see below) but were, until this
  session, unverifiable claims rather than evidence.
- [Phases 1-4: add your own entries here — e.g. any tokenizer claim, code
  bug, or metric AI got wrong on the first pass, per the brief's own
  example: "Claude wrote 80% of my plotting code and confidently invented
  a bug that didn't exist."]

## What was verified (not just asserted)
- Reconstructed the GPT-2 byte-level BPE tokenizer from scratch offline
  (`partA/scripts/gpt2_bpe_offline.py`, using only the local `merges.txt` —
  no network call), since this environment couldn't reach either
  `tiktoken`'s or HuggingFace's servers to download a tokenizer directly.
  Sanity-checked it against known GPT-2 tokenization behavior before
  trusting it, then used it to independently reproduce:
  - The report's exact baseline (eng 1.2652, hin 7.4485 — matches the
    published 1.27 / 7.45 to rounding).
  - `findings.md`'s claimed bug deltas (split-fix +1.41%/+2.01%: exact
    match; lower-fix +2.92% vs. the reported ~2.88%: rounding, not a real
    discrepancy).
- Rebuilt and ran `partB/calculations.py` against `model_spec.md` and
  `bench_log.csv` directly; it reproduces `analysis.md`'s 25-sequence
  answer and the ~201-250 tok/s goodput figures from raw arithmetic, not
  from re-typing the claimed numbers.
- Did **not** independently verify the `mBERT` tokenizer numbers in
  `partA/corrected_analysis/` — no network access to fetch that tokenizer
  in this environment. Worth a quick local re-run before the defense,
  since that's the one A3 claim in this repo that's currently unverified
  by two independent methods.

## What still needs a human pass
- Fill in the Phases 1-4 "where AI helped / was wrong" entries above from
  memory — this file was reconstructed from evidence, not lived experience.
- Re-run `partA/corrected_analysis/` end to end once more before the
  defense to confirm the mBERT numbers reproduce.
