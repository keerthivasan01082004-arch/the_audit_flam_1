#!/usr/bin/env python3
"""
tokenizers_impl.py -- pure-Python, from-scratch implementations of:
  1. GPT-2 byte-level BPE, built from vocab.json + merges.txt
  2. BERT WordPiece, built from vocab.txt (bert-base-multilingual-cased)

These reimplement the published algorithms (OpenAI's GPT-2 encoder.py;
Google's BERT tokenization.py) exactly, using only the raw vocab/merge
files -- no `tiktoken` or `transformers` package involved, because neither
is installable in this sandbox (see precheck.py). This file is the only
place tokenizer logic lives; corrected_analysis.py just calls it.
"""
import json
import unicodedata

# ============================================================
# 1. GPT-2 byte-level BPE
# ============================================================

def bytes_to_unicode():
    """Standard GPT-2 byte<->unicode mapping (OpenAI gpt-2 repo)."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class GPT2Tokenizer:
    def __init__(self, vocab_path, merges_path):
        with open(vocab_path, encoding="utf-8") as f:
            self.encoder = json.load(f)
        with open(merges_path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        # first line is the "#version" header
        merges = [tuple(l.split()) for l in lines[1:] if l.strip()]
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.cache = {}
        # exact literal contractions checked before generic letter/number/punct scan
        self._contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]

    @staticmethod
    def _is_letter(ch):
        return unicodedata.category(ch).startswith("L")

    @staticmethod
    def _is_number(ch):
        return unicodedata.category(ch).startswith("N")

    def _pretokenize(self, text):
        """Reimplements the semantics of GPT-2's pretokenizer regex:
            's|'t|'re|'ve|'m|'ll|'d
            | ?\\p{L}+ | ?\\p{N}+ | ?[^\\s\\p{L}\\p{N}]+
            | \\s+(?!\\S) | \\s+
        using unicodedata categories directly, since the `regex` package
        (which supports \\p{L}) is not installable here. Standard `re`
        does not support Unicode property classes, hence the manual scan."""
        tokens = []
        i, n = 0, len(text)
        is_letter, is_number = self._is_letter, self._is_number
        while i < n:
            matched = False
            for c in self._contractions:
                if text.startswith(c, i):
                    tokens.append(c)
                    i += len(c)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            if ch == " " and i + 1 < n and is_letter(text[i + 1]):
                j = i + 1
                while j < n and is_letter(text[j]):
                    j += 1
                tokens.append(text[i:j]); i = j; continue
            if is_letter(ch):
                j = i
                while j < n and is_letter(text[j]):
                    j += 1
                tokens.append(text[i:j]); i = j; continue
            if ch == " " and i + 1 < n and is_number(text[i + 1]):
                j = i + 1
                while j < n and is_number(text[j]):
                    j += 1
                tokens.append(text[i:j]); i = j; continue
            if is_number(ch):
                j = i
                while j < n and is_number(text[j]):
                    j += 1
                tokens.append(text[i:j]); i = j; continue

            def is_other(c):
                return not c.isspace() and not is_letter(c) and not is_number(c)

            if ch == " " and i + 1 < n and is_other(text[i + 1]):
                j = i + 1
                while j < n and is_other(text[j]):
                    j += 1
                tokens.append(text[i:j]); i = j; continue
            if is_other(ch):
                j = i
                while j < n and is_other(text[j]):
                    j += 1
                tokens.append(text[i:j]); i = j; continue
            # remaining case: ch is whitespace
            j = i
            while j < n and text[j].isspace():
                j += 1
            if j == n:
                tokens.append(text[i:j])
            elif j - i > 1:
                tokens.append(text[i:j - 1])
                j = j - 1
            else:
                tokens.append(text[i:j])
            i = j
        return tokens

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
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
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
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
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        """Returns list of token strings (subword units)."""
        out = []
        for chunk in self._pretokenize(text):
            byte_str = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            out.extend(self._bpe(byte_str).split(" "))
        return out


# ============================================================
# 2. BERT WordPiece (bert-base-multilingual-cased: do_lower_case=False)
# ============================================================

def _is_punctuation(ch):
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F))


class BertWordPieceTokenizer:
    def __init__(self, vocab_path, unk_token="[UNK]", max_chars=200):
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = [l.rstrip("\n") for l in f]
        self.vocab_set = set(self.vocab)
        self.unk_token = unk_token
        self.max_chars = max_chars

    def _basic_tokenize(self, text):
        # cased model: no lowercasing, no accent stripping
        text = text.strip()
        if not text:
            return []
        # insert spaces around CJK chars (none expected in this corpus, kept for fidelity)
        buf = []
        for ch in text:
            if _is_cjk(ord(ch)):
                buf.append(" "); buf.append(ch); buf.append(" ")
            else:
                buf.append(ch)
        text = "".join(buf)

        orig_tokens = text.split()
        split_tokens = []
        for token in orig_tokens:
            output = []
            start_new = True
            for ch in token:
                if _is_punctuation(ch):
                    output.append([ch])
                    start_new = True
                else:
                    if start_new:
                        output.append([])
                    output[-1].append(ch)
                    start_new = False
            split_tokens.extend("".join(piece) for piece in output)
        return split_tokens

    def _wordpiece(self, token):
        chars = list(token)
        if len(chars) > self.max_chars:
            return [self.unk_token]
        sub_tokens = []
        start = 0
        is_bad = False
        while start < len(chars):
            end = len(chars)
            cur = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab_set:
                    cur = substr
                    break
                end -= 1
            if cur is None:
                is_bad = True
                break
            sub_tokens.append(cur)
            start = end
        return [self.unk_token] if is_bad else sub_tokens

    def encode(self, text):
        out = []
        for tok in self._basic_tokenize(text):
            out.extend(self._wordpiece(tok))
        return out


# ============================================================
# 3. Grapheme-cluster approximation (GB9 "Extend" rule only)
#
# *** EXPLORATORY / UNUSED IN OFFICIAL A3 RESULTS (per review) ***
# This function is NOT imported or called by corrected_analysis.py and
# does NOT contribute to raw_results.csv, summary.csv, findings.md, or
# any A3 finding/recommendation. It implements only one rule (GB9) of the
# Unicode UAX #29 grapheme-cluster spec, not the full algorithm, and the
# A3 requirements are explicit that a grapheme-cluster denominator must
# use a real (i.e. complete/certified) Unicode implementation -- which
# would require the `regex` or `pyicu` package, neither installable in
# this sandbox (see precheck.py). Left here only as a record of what was
# tried and why it was excluded, not as a basis for any conclusion.
# Official A3 denominators are tokens/whitespace-word and tokens/UTF-8-byte
# (see corrected_analysis.py).
# ============================================================

def grapheme_clusters_gb9(text):
    """Partial UAX #29 implementation: merges a base character with any
    immediately-following combining mark (Unicode category Mn/Mc/Me),
    which is rule GB9 ("do not break before Extend"). This correctly
    handles vowel-sign/matra attachment in Devanagari, Tamil, and Kannada
    (the dominant source of code-point inflation in this corpus).
    KNOWN GAP, explicitly not claimed as full UAX #29: does not apply the
    Unicode 15 Indic-Conjunct-Break (InCB) tailoring that clusters
    consonant+virama+consonant conjuncts, and does not implement
    Hangul-syllable, regional-indicator/flag, or emoji-ZWJ rules (none of
    which occur in this Latin/Devanagari/Tamil/Kannada UDHR corpus).
    NOT USED by corrected_analysis.py -- see module-level note above."""
    if not text:
        return []
    clusters = []
    current = text[0]
    for ch in text[1:]:
        if unicodedata.category(ch) in ("Mn", "Mc", "Me"):
            current += ch
        else:
            clusters.append(current)
            current = ch
    clusters.append(current)
    return clusters
