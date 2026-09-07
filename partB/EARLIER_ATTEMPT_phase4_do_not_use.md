# Phase 4 — Part B: Capacity Reconciliation

## Executive conclusion

The model specification implies **114,688 KV-cache bytes per token**, or **448 MiB for one fully allocated 4096-token sequence**. Under the explicit primary assumption that the stated 24 GB GPU capacity is treated as 24 GiB, the usable KV budget is **20.48 GiB**, yielding a theoretical ceiling of **46 concurrent full-length sequences** after the stated 1.6 GiB non-KV overhead. A decimal-GB sensitivity gives **43 sequences**. This is a specification-derived ceiling, not a measured capacity guarantee.

The benchmark begins reporting preemptions at **batch 32**, while batch 24 already reaches **0.93 KV utilization**. Thus, the benchmark is consistent with saturation occurring well before the theoretical full-allocation ceiling, but it does not numerically validate “46 sequences” as an operational capacity. The most defensible interpretation is that allocator/scheduler behavior, block granularity, runtime reservations, or the benchmark's utilization definition leaves less practical headroom than the simple arithmetic suggests. Those causes are **not determinable from supplied evidence**.

The harness's `reported_tok_s` is reconstructed by counting **prompt tokens plus generated tokens**:

```text
reported_tok_s = num_requests × (prompt_len + gen_len) / wall_clock_s
```

For the long-prompt batch-24 row, this is **1,607.33 tokens/s**, matching the supplied **1,607.4** within rounding. Actual generated-token goodput is only **200.92 generated tokens/s**. An ITL-based estimate is **249.82 tokens/s**, but it is not an independent exact goodput measurement because it divides request count by a p50 per-request statistic. The disagreement must remain visible.

The previous claim that batch 48 supports approximately **3,200 tok/s** is not defensible. The supplied batch-48 harness value is **1,298.5 prompt-plus-generated tokens/s**, not 3,200, and generated-token goodput is **162.31 tokens/s**. Batch 48 also has **23 preempted sequences**, **0.97 KV utilization**, **955.4 ms TTFT p50**, and **105.4 s E2E p95**. The single primary leadership metric should be **request completion rate at an explicit latency and error/preemption SLO**, with KV utilization, preemptions, TTFT, ITL, and E2E latency as secondary guardrails. A raw token counter is not sufficient for routing decisions.

## B1 — KV-cache arithmetic

### Hypothesis

The model specification should determine the theoretical KV-cache memory cost per token and, with an explicit usable-memory assumption, the maximum number of concurrent 4096-token sequences.

### Exact command

```bash
python3 partB/calculations.py
```

The script parses the relevant constants from `model_spec.md` and loads `bench_log.csv` from paths relative to the repository root.

### Measured inputs

| Variable | Value | Evidence classification |
|---|---:|---|
| Transformer layers | 28 | Measured/specification |
| KV heads (GQA) | 8 | Measured/specification |
| Head dimension | 128 | Measured/specification |
| KV-cache precision | fp16 | Measured/specification |
| Bytes per element | 2 | Derived from fp16 convention |
| K and V factor | 2 | Formula factor: both K and V are stored |
| Maximum model length | 4096 tokens | Measured/specification |
| GPU memory | 24 GB | Measured/specification |
| GPU memory utilization | 0.92 | Measured/specification |
| Non-KV runtime overhead | approximately 1.6 GB | Assumed by the supplied specification |

### Calculation

The required formula is:

```text
KV bytes/token
= 2 × layers × KV_heads × head_dim × bytes_per_element
= 2 × 28 × 8 × 128 × 2
= 114,688 bytes/token
= 112 KiB/token
```

For a fully allocated 4096-token sequence:

```text
114,688 × 4096
= 469,762,048 bytes
= 448 MiB
```

For the primary capacity arithmetic, the stated 24 GB is treated as 24 GiB. This is an explicit unit assumption because the specification does not resolve the vendor-unit convention. The usable KV budget is:

```text
0.92 × 24 GiB − 1.6 GiB
= 22.08 GiB − 1.6 GiB
= 20.48 GiB
```

Therefore:

```text
floor(20.48 GiB / 448 MiB)
= floor(20.48 × 1024 / 448)
= floor(46.72)
= 46 full 4096-token sequences
```

Using decimal 24 GB only as a sensitivity check produces **43 sequences** after converting the stated 1.6 GB overhead to 1.6 GiB for consistency. The primary result is therefore **46 sequences under the explicit 24-GiB assumption**, with **43** as a unit-convention sensitivity.

### Interpretation

The 46-sequence value is **derived**, not measured. It is the number of full-length allocations that fit in the simplified stated budget. It is not a production-safe routing threshold because the supplied specification does not quantify fragmentation, block metadata, CUDA allocator effects, additional framework reservations, or transient workspace requirements.

### Limitation

The total usable KV memory and the 24 GB unit convention require assumptions. Actual operational capacity is **not determinable from supplied evidence** beyond this arithmetic ceiling and the benchmark observations below.

## B2 — Benchmark reconciliation

### Hypothesis

If the theoretical full-sequence capacity is operationally representative, the long-prompt benchmark should remain non-preemptive until near that capacity. The benchmark should also show whether utilization and latency signal saturation before preemptions.

### Exact command

```bash
python3 partB/calculations.py
```

### Measured long-prompt rows

These are the original rows with `prompt_len = 3584` and `gen_len = 512`.

| Batch | Prompt | Gen | Requests | Wall clock (s) | Reported tok/s | TTFT p50 (ms) | ITL p50 (ms) | E2E p95 (ms) | Preempted | KV util |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 3584 | 512 | 4 | 28.98 | 565.4 | 483.2 | 51.33 | 32,673.3 | 0 | 0.16 |
| 8 | 3584 | 512 | 8 | 36.30 | 902.6 | 519.0 | 62.26 | 39,982.9 | 0 | 0.31 |
| 16 | 3584 | 512 | 16 | 49.97 | 1,311.4 | 498.3 | 77.20 | 54,602.1 | 0 | 0.62 |
| 24 | 3584 | 512 | 24 | 61.16 | 1,607.4 | 500.5 | 96.07 | 69,221.3 | 0 | 0.93 |
| 32 | 3584 | 512 | 32 | 94.71 | 1,384.0 | 636.9 | 101.79 | 97,465.7 | 7 | 0.97 |
| 48 | 3584 | 512 | 48 | 151.41 | 1,298.5 | 955.4 | 100.00 | 105,427.5 | 23 | 0.97 |

### Transition and comparison with theory

The first long-prompt row with preemptions is **batch 32**, with 7 preempted sequences. Batch 24 has no preemptions but is already at **0.93** reported peak KV utilization. The theoretical arithmetic gives 46 full 4096-token sequences, whereas the first observed preemptions occur at 32 requests. The gap is 14 requests, or approximately 30.4% below the simplified 46-sequence ceiling. Relative to the 43-sequence decimal-GB sensitivity, batch 32 is 11 requests, or approximately 25.6%, below that ceiling.

This is not an exact theory-versus-measurement match. It is directionally compatible with the idea that high KV occupancy precedes scheduler stress, but the benchmark does not establish why the operational transition is earlier. The scheduler may allocate blocks for prompt and decode states differently from the simple full-length arithmetic, and the CSV does not expose enough detail to quantify that effect.

### Interpretation

Batch size increases initially improve the harness counter: long-prompt `reported_tok_s` rises from 565.4 at batch 4 to 1,607.4 at batch 24. At the same time, KV utilization rises from 0.16 to 0.93, ITL worsens from 51.33 ms to 96.07 ms, and E2E p95 rises from 32.7 s to 69.2 s. After the transition, reported throughput falls to 1,384.0 at batch 32 and 1,298.5 at batch 48, while preemptions and latency increase. This is saturation behavior, not linear scaling.

### Limitation

There is one row per configuration and no confidence interval or repeated-trial variance. `kv_cache_util` is a peak harness statistic whose exact denominator is not documented beyond the column note. The benchmark does not provide direct memory allocation traces.

## B3 — Throughput claim audit

### Hypothesis

The supplied `reported_tok_s` should be reproducible from the request count, prompt length, generation length, and wall-clock time if it counts all processed prompt and generated tokens.

### Exact command

```bash
python3 partB/calculations.py
```

### Measured and reconstructed result

For every row, the script computes:

```text
num_requests × (prompt_len + gen_len) / wall_clock_s
```

The maximum absolute difference from the CSV's `reported_tok_s` is **0.193242 tok/s**, consistent with display rounding. For long-prompt batch 24:

```text
24 × (3584 + 512) / 61.16
= 98,304 / 61.16
= 1,607.33 tokens/s
```

The CSV reports **1,607.4 tokens/s**. Thus the formula is supported.

### Generated-token goodput

For batch 24, generated-token goodput is:

```text
24 × 512 / 61.16
= 12,288 / 61.16
= 200.92 generated tokens/s
```

This excludes the 86,016 prompt tokens counted by the harness. The two measures answer different questions: the harness counter treats prompt processing as token work, while generated-token goodput measures only output production over the complete run elapsed time.

### ITL-derived estimate

A second estimate can be formed from the p50 ITL column:

```text
num_requests / (itl_ms_p50 / 1000)
= 24 / (96.07 / 1000)
= 249.82 tokens/s
```

This is an **ITL-based aggregate estimate**, not an exact independent generated-token throughput measurement. It uses a median per-sequence inter-token latency and assumes that multiplying its reciprocal by the number of requests is representative of aggregate decode rate. The estimate is **24.3% higher** than the wall-clock generated-token goodput (249.82 versus 200.92). The discrepancy is not repaired or hidden. It can arise because p50 ITL is not a run-wide mean, because wall-clock time includes prompt processing and other scheduling effects, and because prefill/decode overlap and measurement windows are not specified.

### Limitation

The supplied columns do not include per-token timestamps or a decode-only elapsed interval. Therefore, an exact independent generated-token throughput measurement from ITL is **not determinable from supplied evidence**.

## Batch 48 claim audit

### Hypothesis

The prior claim that batch 48 yields approximately 3,200 tok/s should survive reconstruction and should not coincide with saturation signals if it is a defensible capacity recommendation.

### Exact command

```bash
python3 partB/calculations.py
```

### Measured result and calculations

For batch 48:

```text
reported harness throughput
= 48 × (3584 + 512) / 151.41
= 1,298.51 tokens/s
```

The supplied value is **1,298.5**, not approximately 3,200. Generated-token goodput is:

```text
48 × 512 / 151.41
= 162.31 generated tokens/s
```

The ITL-derived estimate is:

```text
48 / (100.00 / 1000)
= 480.00 tokens/s
```

Again, this is only a p50-ITL-based estimate and is not equivalent to exact wall-clock generated-token goodput.

### Saturation signals

Compared with batch 24, batch 48 has lower reported throughput (**1,298.5 vs 1,607.4 tok/s**) and lower generated goodput (**162.31 vs 200.92 generated tok/s**). Preemptions rise from **0 to 23**. KV utilization rises from **0.93 to 0.97**. TTFT p50 rises from **500.5 ms to 955.4 ms**. ITL p50 rises from **96.07 ms to 100.00 ms**. E2E p95 rises from **69,221.3 ms to 105,427.5 ms**. These observations contradict the prior recommendation to assume linear scaling to batch 48.

### Interpretation

The claim **batch 48 → approximately 3,200 tok/s is unsupported and contradicted by the supplied benchmark**. The benchmark shows that batch 48 is beyond the non-preemptive operating region represented by batch 24. It delivers lower measured throughput and worse user-visible latency with substantial preemption.

### Limitation

The benchmark does not test all intermediate batch sizes, does not report repeated trials, and does not state an SLO. It therefore cannot identify the exact optimal operating point for every workload. It does establish that this particular batch-48 run is not evidence for 3,200 tok/s.

## B4 — Serving metric decision

### Hypothesis

Leadership needs one metric that reflects usable serving capacity and supports routing decisions, rather than a counter that can increase while latency and saturation worsen.

### Decision

Use **request completion rate at an explicit user-visible latency SLO, with preempted/error requests excluded from successful completions**, as the single primary serving metric for capacity and routing decisions.

### Evaluation

| Candidate | Assessment |
|---|---|
| Reported tok/s | Reproducible, but combines prompt and generated tokens and can obscure saturation. Not sufficient alone. |
| Generated tokens/sec | More aligned with output production, but the supplied wall-clock measure includes prefill and does not directly express whether requests complete acceptably. |
| Request completion rate at latency SLO | Directly measures usable service capacity, is interpretable for routing, and penalizes overload when requests stall or are preempted. Recommended primary. |
| ITL | User-visible during decode and useful as a guardrail, but p50 does not capture tail behavior or completion. |
| TTFT | Important for responsiveness, but does not capture decode completion. |
| E2E latency | Directly user-visible and essential as an SLO, but a latency distribution alone does not express how many requests complete. |
| Preemptions | Strong saturation signal, but not a complete service outcome. |
| KV utilization | Strong leading indicator, but its exact harness semantics are not fully specified and it is not user-facing. |

### Interpretation

A raw throughput maximum is not a safe routing target. Batch 24 has the highest long-prompt `reported_tok_s` in the supplied set, but it already has 0.93 KV utilization and 69.2 s E2E p95. Batch 32 and batch 48 add preemptions and worsen latency while reducing the harness counter. A completion-rate SLO captures the business-relevant question: how much traffic can the server complete within an acceptable time without overload behavior.

### Secondary guardrails

Use **E2E latency p95**, **TTFT p50/p95**, **ITL**, **preempted sequence count/rate**, and **KV-cache utilization** as secondary guardrails. The supplied CSV has p50 TTFT/ITL and p95 E2E, but no request-level completion status or error count. Consequently, the recommended primary metric cannot be computed from this CSV alone; an instrumented future benchmark should add successful completions, deadline misses, and preemption counts per run.

## Reproducibility and unresolved issues

The exact commands used were:

```bash
cd /home/ubuntu/phase3_A3_final
python3 partB/calculations.py | tee partB/run_output.txt
python3 partB/calculations.py > /tmp/partB_second_run.txt
cmp -s partB/run_output.txt /tmp/partB_second_run.txt
```

The first and second script outputs were identical. The script uses repository-relative paths and is runnable from the repository root. Important unresolved issues are **not determinable from supplied evidence**: actual free GPU memory, allocator/block overhead, the precise definition and denominator of `kv_cache_util`, repeated-run variance, request completion/error status, per-token timestamps, and an explicit serving latency SLO.

## References

[1]: ../model_spec.md "Supplied FLM-4B-Instruct model specification and serving configuration"

[2]: ../bench_log.csv "Supplied benchmark log"

[3]: ../REPORT_v0.md "Previous intern's draft tokenizer and serving findings"
