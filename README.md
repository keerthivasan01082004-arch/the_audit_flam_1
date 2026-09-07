# The Audit — submission

Start with `NOTEBOOK.md` (chronological log, including the Phase 5 repo
audit) and `AI_USAGE.md`.

## Layout
- `partA/` — tokenizer audit
  - `findings.md` — A2 (script/metric bugs + the conceptual flaw)
  - `memo.md` — A4 (recommendation memo)
  - `corpus/` — A1 (UDHR-based, 4 languages, provenance in `corpus/README.md`)
  - `corrected_analysis/` — A3 (gpt2 + mBERT, two denominators)
  - `scripts/` — the intern's original `fertility.py`, plus this session's
    rebuilt, offline, independently-verified evidence for A2
    (`gpt2_bpe_offline.py`, `verify_a2.py`)
- `partB/` — capacity reconciliation
  - `analysis.md` — the write-up (B1–B4), numbers are correct
  - `calculations.py` — rebuilt this session; run it, it reproduces
    every number in `analysis.md` from `model_spec.md` + `bench_log.csv`
  - `EARLIER_ATTEMPT_phase4_do_not_use.md` — kept deliberately as the
    documented dead end (see NOTEBOOK Phase 5) — the 46-vs-25 discrepancy
    and why 25 is right
- `partC/memo.md` — decision memo, grounded in Part A's actual measured
  tokenizer costs across languages, not hypothetical ones

## Before the defense
Run these two and be ready to modify them live:
```
cd partA/scripts && python3 verify_a2.py
cd partB && python3 calculations.py
```
Also worth doing before the defense, not done in this pass (see
`AI_USAGE.md`): re-verify the mBERT numbers in `partA/corrected_analysis/`
the same independent way the gpt2 numbers were verified here.
