"""
Standalone, offline reconstruction of the GPT-2 byte-level BPE tokenizer,
built only from the local `merges.txt` (no network call, no vocab.json
needed for counting purposes -- we only need token *counts*, not ids).

This is the standard GPT-2 tokenizer algorithm (same one used by tiktoken's
"gpt2" and HF's GPT2Tokenizer) -- byte-to-unicode remapping + regex
pre-tokenization + greedy BPE merges in rank order.
"""

import functools
import regex as re


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_pairs(word):
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class GPT2BPE:
    def __init__(self, merges_path):
        self.byte_encoder = bytes_to_unicode()
        with open(merges_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert lines[0].startswith("#version"), "unexpected merges.txt header"
        merges = [tuple(line.split()) for line in lines[1:] if line]
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return [token]
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        self.cache[token] = word
        return list(word)

    def encode(self, text):
        """Return the list of BPE token strings (ids not needed for counting)."""
        tokens = []
        for piece in re.findall(PAT, text):
            piece_bytes = piece.encode("utf-8")
            mapped = "".join(self.byte_encoder[b] for b in piece_bytes)
            tokens.extend(self.bpe(mapped))
        return tokens


if __name__ == "__main__":
    import sys

    enc = GPT2BPE("merges.txt")
    # sanity checks against well-known GPT-2 behavior before trusting it on
    # the real corpus
    checks = [
        ("Hello world", 2),      # "Hello", " world"
        ("The Quarterly Review", 3),  # "The", " Quarterly", " Review"
    ]
    for text, expected in checks:
        toks = enc.encode(text)
        status = "OK" if len(toks) == expected else "MISMATCH"
        print(f"[{status}] {text!r} -> {len(toks)} tokens (expected {expected}): {toks}")
