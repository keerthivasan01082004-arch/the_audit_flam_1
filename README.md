# The Audit — Tokenizer & Serving Stack Review

Submission for FlamAI's AI Team Intern take-home ("The Audit"). Start with
`NOTEBOOK.md` for the full working log (hypothesis → experiment → result →
revision) and `AI_USAGE.md` for an honest breakdown of where AI helped and
where it didn't.

## Structure

- `partA/` — Tokenizer fertility audit
  - `findings.md` — script/metric audit (A2): the code bugs, the conceptual
    flaw, and the one thing that looks suspicious but isn't
  - `corpus/` — the multilingual eval corpus (A1), UDHR-based across
    English, Hindi, Kannada, Tamil; provenance and caveats in `corpus/README.md`
  - `corrected_analysis/` — corrected cross-language comparison (A3):
    GPT-2 and mBERT, two denominators
  - `memo.md` — recommendation memo (A4)
  - `scripts/` — the original `fertility.py`, plus an independent offline
    re-implementation (`gpt2_bpe_offline.py`, `verify_a2.py`) used to verify
    the audit's numbers from scratch

- `partB/` — Capacity reconciliation
  - `analysis.md` — B1–B4 write-up
  - `calculations.py` — reproduces every number in `analysis.md` from
    `model_spec.md` and `bench_log.csv`
  - `EARLIER_ATTEMPT_phase4_do_not_use.md` — kept intentionally as a
    documented dead end (see `NOTEBOOK.md`): an earlier pass got the
    KV-cache capacity number wrong (46 vs. the correct 25) before the
    memory-budget bug was found

- `partC/memo.md` — decision memo for the casual-tone localization problem

## Reproducing the results

```bash
cd partA/scripts && python3 verify_a2.py
cd partB && python3 calculations.py
```

Both scripts regenerate the numbers cited in `findings.md` and `analysis.md`
independently of the write-ups.

## Before the defense

- Re-run both scripts above from a clean clone, cold.
- Know why each flagged bug in `fertility.py` distorts the numbers —
  direction and magnitude, not just "it's wrong."
- Be ready to walk through the B1 46→25 correction as a live counterfactual.
- Not yet done: independently re-verify the mBERT numbers in
  `partA/corrected_analysis/` the same way the GPT-2 numbers were verified.
