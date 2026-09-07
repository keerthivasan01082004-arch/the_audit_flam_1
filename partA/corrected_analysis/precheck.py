#!/usr/bin/env python3
"""
precheck.py -- environment verification for A3, run BEFORE any tokenizer work.
Records whether the tokenizer implementations required by the A3 brief
are actually available in this sandbox. This is a real check, not a claim.
"""
import importlib, subprocess, sys, os

def check_import(name):
    try:
        importlib.import_module(name)
        return True, None
    except ModuleNotFoundError as e:
        return False, str(e)

results = {}
for mod in ["tiktoken", "transformers", "sentencepiece", "regex"]:
    ok, err = check_import(mod)
    results[mod] = (ok, err)

net = subprocess.run(
    ["curl", "-sS", "-m", "5", "-o", "/dev/null", "-w", "%{http_code}",
     "https://huggingface.co"],
    capture_output=True, text=True
)

vocab_search = subprocess.run(
    ["bash", "-c",
     "find / -xdev -iname '*vocab.bpe*' -o -iname '*encoder.json*' "
     "-o -iname '*merges.txt*' -o -iname '*vocab.txt' 2>/dev/null | grep -v uploads"],
    capture_output=True, text=True
)

print("=== Package availability ===")
for mod, (ok, err) in results.items():
    print(f"{mod:15s} {'AVAILABLE' if ok else 'MISSING'}")

print("\n=== Network ===")
print(f"curl https://huggingface.co -> HTTP {net.stdout.strip()}")

print("\n=== Local vocab/merges files on disk (excluding uploads) ===")
found = vocab_search.stdout.strip()
print(found if found else "(none found)")
