# Part B — Capacity reconciliation

Model: FLM-4B-Instruct, 28 layers, 8 KV heads (GQA), head_dim 128,
fp16 KV cache. GPU: 1× L4, 24 GB, `gpu_memory_utilization=0.92`,
`max_model_len=4096`, ~1.6 GB non-KV overhead.

## B1 — KV-cache bytes/token and max concurrent 4096-token sequences

**Exact arithmetic** (per token, per layer you store K and V, each
`kv_heads × head_dim` values in fp16 = 2 bytes):

```
bytes/token = layers × 2(K,V) × kv_heads × head_dim × 2 bytes(fp16)
            = 28 × 2 × 8 × 128 × 2
            = 114,688 bytes  (= 112 KiB/token, exactly)
```

For one full 4096-token sequence: `114,688 × 4096 = 469,762,048 bytes`
(= 448 MiB, exactly).

**Assumption flagged:** I treat "GB" as decimal (1 GB = 1e9 bytes),
matching how NVIDIA markets the L4's 24 GB and how param counts
naturally convert to bytes. If your team's tooling measures in
binary GiB (2^30) instead, the usable pool shrinks by ~7% and the
final count below would drop by roughly one sequence — this unit
choice is exactly the kind of thing worth stating explicitly rather
than leaving implicit.

```
usable        = 0.92 × 24e9            = 22.08e9 bytes
weights (fp16)= 4.2e9 params × 2 bytes = 8.40e9 bytes
overhead      = ~1.6e9 bytes
remaining_KV  = 22.08e9 − 8.40e9 − 1.6e9 = 12.08e9 bytes

max_seqs = floor(12.08e9 / 469,762,048) = 25
```

**Check against the log:** the 3584-prompt sweep (closest to
max_model_len) shows `kv_cache_util=0.93` and **zero** preemptions at
batch 24, then `kv_cache_util=0.97` with preemptions starting at batch
32. A ceiling of 25 full-length sequences sits exactly between "24
fits cleanly" and "32 doesn't" — the prediction and the empirical
data agree.

## B2 — The long-prompt throughput anomaly

Long-prompt (3584) sweep, `reported_tok_s` by batch:

| batch | reported_tok_s | preempted_seqs | kv_cache_util |
|---|---|---|---|
| 4 | 565.4 | 0 | 0.16 |
| 8 | 902.6 | 0 | 0.31 |
| 16 | 1311.4 | 0 | 0.62 |
| 24 | **1607.4** (peak) | 0 | 0.93 |
| 32 | 1384.0 (↓) | 7 | 0.97 |
| 48 | 1298.5 (↓ further) | 23 | 0.97 |

Throughput rises with batch size only up to 24, then **falls** as
batch size keeps increasing — the opposite of naive "throughput scales
with batch" intuition.

**Mechanism:** I verified `reported_tok_s` is computed as
`num_requests × (prompt_len + gen_len) / wall_clock_s` — it counts
*every* token of the request, prompt included, as if it were produced
at decode speed. (Checked against all 13 rows in the log; matches to
within rounding on every single one.) Once concurrent sequences exceed
the ~25-sequence KV-cache ceiling from B1, the scheduler has to
preempt sequences to free cache space — those sequences' prefill work
gets redone, wall-clock time balloons, and even this inflated metric
can't keep up. `preempted_seqs` (0 → 7 → 23) confirms this tracks
exactly with `kv_cache_util` saturating at 0.97.

**Proposed fix with predicted effect:** cap concurrent long-context
(3584-token) requests at ≤24 (below the KV ceiling) via the serving
scheduler's max-batch setting, or reduce `max_model_len`/`gpu_memory_utilization`
tradeoffs to raise the ceiling. Predicted effect: eliminates
preemption thrashing entirely for this workload (preempted_seqs → 0),
recovering the batch-24 peak of 1607 tok/s (nominal metric) as the
achievable ceiling instead of degrading to 1298 at batch 48.

## B3 — Why "longer prompts give better throughput" and "batch 48 → ~3200 tok/s" are both wrong

Both come from the same root cause identified in B2: `reported_tok_s`
bundles prompt tokens (processed once, in parallel, during prefill —
cheap) together with decode tokens (produced one at a time,
memory-bandwidth-bound — expensive) in the same numerator. A
long-prompt request "generates" far more counted tokens per request
without doing proportionally more of the expensive work, which makes
long prompts look faster than they really are for the thing that
matters: sustained decode capacity. The ~1600 tok/s "best observed"
the report anchors on is literally the batch-24 long-prompt row
(1607.4), and linearly doubling it for batch 48 (`1607×2≈3214≈"~3200"`)
ignores that the *very next rows in the same log* already show
throughput **falling**, not doubling, once you cross the KV ceiling.

**Honest goodput of the batch-24 long-prompt row, two independent
ways:**

1. **Decode tokens over wall clock:** `24 requests × 512 gen tokens /
   61.16s = 12,288 / 61.16 = 200.9 tok/s`.
2. **Batch size over steady-state inter-token latency:** at batch 24,
   one decode step produces one token for all 24 sequences at once, so
   `24 / (itl_ms_p50 / 1000) = 24 / 0.09607 = 249.8 tok/s`.

Both land around **200–250 tok/s** — roughly **1/7th to 1/8th** of the
1607.4 the report used for capacity planning. (The two methods don't
match exactly: a `ttft + itl×(gen_len−1)` model predicts a 49.6s wall
clock vs the actual 61.2s — an ~19% gap I can't fully explain from the
columns given, likely batch-formation/scheduling overhead not captured
by the simple per-token latency model. Flagging this rather than
papering over it.)

**What the report should have said:** the true decode-time capacity of
this setup is closer to ~200–250 tok/s per L4 at this batch/context
size, not ~1600–3200 tok/s, and it degrades further (not linearly
improves) once concurrent 3584+-token requests exceed roughly 24–25.

## B4 — Serving-stack counter to confirm the mechanism

Pull the scheduler's **preemption counter** (e.g. vLLM's
`num_preemptions_total` / KV-cache-eviction counter — the production
analogue of this log's `preempted_seqs` column). Expected value: 0 at
concurrency below the ~25-sequence, 4096-token-equivalent ceiling
derived in B1, then rising sharply above it — which is exactly the 0 →
7 → 23 pattern already visible in this log at batch 24 → 32 → 48, so
production should reproduce the same step function.
