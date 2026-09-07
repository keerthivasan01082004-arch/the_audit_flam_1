"""
Part B — capacity reconciliation, rebuilt from model_spec.md + bench_log.csv.

This replaces the empty calculations_final.py. It reproduces the numbers
in analysis.md exactly (25 sequences, ~201/~250 tok/s goodput) and shows
why the OLD run_output.txt figure of 46 sequences was wrong: that run
forgot to subtract model-weights memory from the usable pool before
dividing by KV-cache-bytes-per-token.
"""

import csv

# ---- from starter_kit/bench/model_spec.md ----
LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
BYTES_PER_PARAM = 2          # fp16
PARAMS = 4.2e9
GPU_MEM_BYTES = 24e9         # decimal GB, see note below
GPU_MEM_UTIL = 0.92
NON_KV_OVERHEAD_BYTES = 1.6e9
MAX_MODEL_LEN = 4096

print("=== B1: KV-cache bytes/token ===")
kv_bytes_per_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES_PER_PARAM
print(f"kv_bytes_per_token = 2 * {LAYERS} * {KV_HEADS} * {HEAD_DIM} * {BYTES_PER_PARAM}"
      f" = {kv_bytes_per_token} bytes ({kv_bytes_per_token/1024:.1f} KiB)")

print("\n=== B1: max concurrent 4096-token sequences ===")
print("NOTE: decimal GB (1e9) used throughout, matching how the L4's 24GB")
print("and the param count naturally convert to bytes. Binary GiB would")
print("shrink the usable pool ~7% and drop the answer by about one sequence.")

usable = GPU_MEM_UTIL * GPU_MEM_BYTES
weights = PARAMS * BYTES_PER_PARAM
remaining_kv = usable - weights - NON_KV_OVERHEAD_BYTES
print(f"usable        = {GPU_MEM_UTIL} * {GPU_MEM_BYTES:.2e} = {usable:.3e} bytes")
print(f"weights       = {PARAMS:.2e} * {BYTES_PER_PARAM}     = {weights:.3e} bytes")
print(f"overhead      =                        {NON_KV_OVERHEAD_BYTES:.3e} bytes")
print(f"remaining_KV  = {remaining_kv:.3e} bytes")

bytes_per_seq = kv_bytes_per_token * MAX_MODEL_LEN
max_seqs = int(remaining_kv // bytes_per_seq)
print(f"bytes/4096-seq = {bytes_per_seq:,} ({bytes_per_seq/2**20:.1f} MiB)")
print(f"max_seqs = floor({remaining_kv:.3e} / {bytes_per_seq:,}) = {max_seqs}")

print(f"\n*** THIS IS THE NUMBER: {max_seqs} concurrent full-length sequences ***")
print("(old run_output.txt said 46 -- that calc used usable-minus-overhead")
print(" only and never subtracted model-weights memory. Wrong; discard it.)")

print("\n=== cross-check against bench_log.csv (kv_cache_util column) ===")
with open("starter_kit/bench/bench_log.csv", newline="") as f:
    rows = [r for r in csv.DictReader(f) if int(r["prompt_len"]) == 3584]

for r in rows:
    b = int(r["batch_size"])
    util = float(r["kv_cache_util"])
    implied_capacity = b / util if util else float("nan")
    print(f"batch={b:>3}  kv_cache_util={util:.2f}  preempted={r['preempted_seqs']:>2}  "
          f"-> implied full capacity ~= {implied_capacity:.1f} seqs")

print(f"\nAt batch=24 (last row with 0 preemptions), implied capacity ~= "
      f"{24/0.93:.1f} -- matches the computed {max_seqs} far better than 46 would")
print("(46 would predict kv_cache_util at batch=24 of only "
      f"{24/46:.2f}, not the observed 0.93).")

print("\n=== B3: honest goodput of the batch=24, prompt=3584 row ===")
row24 = next(r for r in rows if int(r["batch_size"]) == 24)
num_requests = int(row24["num_requests"])
gen_len = int(row24["gen_len"])
wall_clock_s = float(row24["wall_clock_s"])
itl_ms_p50 = float(row24["itl_ms_p50"])
reported = float(row24["reported_tok_s"])
prompt_len = int(row24["prompt_len"])

method1 = num_requests * gen_len / wall_clock_s
method2 = num_requests / (itl_ms_p50 / 1000)
reconstructed_reported = num_requests * (prompt_len + gen_len) / wall_clock_s

print(f"method 1 (decode tokens / wall clock): {num_requests} * {gen_len} / {wall_clock_s} "
      f"= {method1:.1f} tok/s")
print(f"method 2 (batch / inter-token latency): {num_requests} / ({itl_ms_p50}/1000) "
      f"= {method2:.1f} tok/s")
print(f"reported_tok_s formula check: {num_requests}*({prompt_len}+{gen_len})/{wall_clock_s} "
      f"= {reconstructed_reported:.1f}  (log says {reported}) "
      f"-> {'MATCH' if abs(reconstructed_reported-reported) < 1 else 'MISMATCH'}")
print(f"\nreal goodput ~= {method1:.0f}-{method2:.0f} tok/s, vs reported {reported} tok/s "
      f"-> report's number is {reported/method1:.1f}-{reported/method2:.1f}x inflated")
