# Part C — Localization Product Decision Memo

**Scope:** Hindi, Kannada, Tamil, Telugu, Bengali, Marathi. Constraints: 1×A100-80GB,
2 weeks eng time at 10h/week (**20 eng-hours total**, ASSUMED), 1 native reviewer
(Hindi+Kannada only) at 10h/week, launch review in 3 weeks, no external API budget.

# Recommendation
**RECOMMENDED: Option (c), prompt-engineering only** (few-shot style exemplars +
a "casualness" rubric baked into the system prompt), with no fine-tuning this cycle.

# Why This Option
- **Time-to-first-result:** a prompting change can be tested same-day; SFT (a) and
  a rewriter model (b) both require a data-generation → filter → train → eval loop
  before a single reviewable output exists (DERIVED from the pipeline steps each
  option requires).
- **Reviewer bandwidth is the real bottleneck, not GPU:** 1 reviewer × 10h/week ×
  2 languages cannot validate quality gains for 6 launch languages regardless of
  which technical approach is chosen (MEASURED from stated constraints). Any option
  that also needs the reviewer to *label training data* (a, b) competes for the same
  scarce hours that are needed for *evaluation*.
- **Reversibility/launch risk:** a prompt change ships and rolls back in minutes;
  a trained artifact adds a release/rollback surface with 3 weeks to launch
  (ASSUMED typical for this org, not measured here).
- **Cost asymmetry across languages is real and untested by (c):** Part A's
  measured mBERT fertility shows Kannada tokenizing ~3x less efficiently than
  Hindi (4.03x vs 1.36x vs English, MEASURED in `partA/corrected_analysis`).
  This is a *tokenizer* finding, not a GPU-dollar figure, but it INFERS that any
  approach adding tokens per response (long few-shot prompts, or a second
  rewriter pass) will cost disproportionately more to serve in Kannada than
  Hindi. Option (c) is not exempt from this: few-shot exemplars in the system
  prompt add input-token overhead of their own. The actual magnitude of that
  overhead depends on request frequency and on whether effective prefix
  caching exists in the serving stack — neither is established by the
  evidence available here (ASSUMED unknown, not measured).

# Alternatives Considered
- **(a) SFT on synthetic pairs — rejected.** Needs synthetic data generation
  (no external API budget means generation must run locally on the one A100,
  competing with any training time), reviewer-labeled quality filtering across
  6 languages, and training/eval — not achievable in 20 eng-hours with
  reviewer coverage for only 2 of 6 languages (DERIVED).
- **(b) ≤1B rewriter model — rejected for this cycle.** Lower engineering lift
  than (a) but still requires a training pass, adds a permanent inference-time
  cost and latency on every response, and — per the tokenizer finding above —
  that added cost is not uniform across languages (INFERRED). Revisit as a
  Phase 2 investment if prompting plateaus.
- **(c) Prompt engineering — selected.** See above.

# Data Plan
No training data required. Needed: a small **style exemplar set** — natural,
casual-register example replies per language, for few-shot conditioning and as
the reviewer's rubric anchor.
- **How much:** RECOMMENDED starting point of 15–25 exemplars per language for
  the Day-1 experiment. This is not a measured or externally validated number
  — it has no basis in the supplied evidence and should be revised if Day-1
  results suggest it's too few or too many.
- **Source:** hand-written/adapted by whoever is available (PM, reviewer,
  existing style guides) — not model-generated, to avoid spending the
  reviewer's scarce hours checking synthetic output for naturalness before
  it's even used.
- Hindi + Kannada exemplars reviewed by the native reviewer; Tamil/Telugu/
  Bengali/Marathi exemplars sourced from other native speakers on the product
  team if available, or flagged as **unverified** if not (RECOMMENDED to note
  this explicitly to leadership rather than silently treating all 6 languages
  as equally validated).

# Experiment Plan (Day-1 Experiment)
- **Input data:** 30 existing production/eval prompts per language (60 for
  Hindi+Kannada if available), reused as-is — no new corpus needed.
- **Baseline:** current system prompt, same 30 prompts, same model.
- **Intervention:** system prompt + few-shot exemplars (casual-register),
  same 30 prompts, same model, temperature held constant.
- **Evaluation method:** reviewer blind A/B rating on Hindi+Kannada only
  (baseline vs. intervention, order randomized, reviewer doesn't know which is
  which); other 4 languages get a lightweight non-blind spot-check by any
  available speaker as a sanity check only, not a quality gate.
- **Success threshold:** ≥65% of Hindi+Kannada pairs preferred for casualness
  by the reviewer, **with no increase** in reviewer-flagged correctness/safety
  regressions vs. baseline (RECOMMENDED thresholds — no prior data exists to
  derive them from).
- **Kill criterion:** if preference rate is ≤50% (no better than chance) after
  one round of prompt iteration, or any correctness/safety regression appears,
  stop and do not scale the prompt change to more languages.

# Reviewer Plan
- **What they review:** blind A/B pairs (baseline vs. new prompt) for Hindi and
  Kannada only — the two languages this reviewer can actually judge.
- **How many samples:** ~30 pairs/language/week (60 pairs/week total across
  Hindi+Kannada) at ~2 min/pair ≈ 2 hours/week of blind A/B review. The
  remaining ~8h/week of the reviewer's 10h/week budget is not idle reserve —
  it is allocated to exemplar validation (Data Plan), re-review of items
  flagged after prompt iterations, and the reviewer's input into the Week-2
  decision gate.
- **Selection:** stratified sample across prompt types already in eval sets
  (not cherry-picked), plus any pairs the on-call team flags as odd.
- **Rubric:** 1–5 casualness score + binary correctness/safety flag per output;
  disagreement with a second-pass self-review (reviewer re-checks own flagged
  items) since a second native reviewer isn't available — this is a real
  limitation, not a solved problem (RECOMMENDED to disclose to leadership).
- **Protecting reviewer time:** hard cap at 10h/week; no ad-hoc requests
  outside the weekly review batch; automated heuristics (banned-word/formality
  word-list checks) used to pre-filter obviously-bad outputs before they reach
  the reviewer.

# Success Metric
Reviewer-rated casualness preference rate (intervention vs. baseline) on
Hindi+Kannada blind A/B, with zero net-new correctness/safety regressions.

**Numerical threshold:** ≥65% preference for the new prompt, 0 regressions,
sustained across two consecutive weekly review batches.

# Kill Criterion
Preference rate ≤50% after one full iteration cycle, OR any correctness/
safety regression flagged by the reviewer, OR the 4 unreviewed languages'
spot-checks surface obvious breakage (e.g., code-switching into English,
broken script rendering) — any one of these kills the prompt-only path for
that language and reverts to baseline for it.

# Day-1 Action
Assemble the Hindi+Kannada exemplar sets (reviewer contributes/validates),
draft the new system prompt, run the 30-prompt baseline-vs-intervention
generation, and hand the blind pairs to the reviewer.

# Two-Week Decision Gate
By end of Week 1: first reviewer batch scored; prompt iterated once if below
threshold. By end of Week 2, leadership decides one of:
1. **Ship** the prompt change for Hindi+Kannada (threshold met) and extend to
   the 4 unreviewed languages with spot-check-only sign-off (explicitly lower
   confidence, RECOMMENDED to flag as such).
2. **Iterate once more** within the remaining time before the Week-3 launch
   review (only if trend is positive but short of threshold).
3. **Kill/hold**: ship no localization change this cycle; scope a Phase 2
   investment in the ≤1B rewriter (b) with a full 2-week GPU/reviewer budget
   of its own, informed by whatever prompting revealed about the failure mode.

# Risks and Limitations
- 4 of 6 launch languages have no native-reviewer validation this cycle —
  a real product risk, not eliminated by any of the three options under these
  constraints (MEASURED constraint: reviewer covers only Hindi+Kannada).
- Thresholds (65%, 50%) and exemplar count (15–25) are RECOMMENDED starting
  points, not derived from prior experiments — they should move if Day-1
  results suggest they're miscalibrated.
- Single reviewer means no inter-annotator agreement check; disagreement
  handling relies on self re-review, which is a known weak substitute for a
  second annotator.
- This memo does not establish GPU-dollar serving costs for any option; the
  tokenizer-fertility numbers cited are token-count evidence from Part A, used
  here only to reason about *relative* exposure across languages, not as a
  cost figure.
