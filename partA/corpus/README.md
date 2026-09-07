# A1 Multilingual Evaluation Corpus

## Source

**"UDHR in XML"** project — the canonical source behind the official
[unicode.org/udhr](https://unicode.org/udhr) mirror.

- Repository: https://github.com/eric-muller/udhr
- Commit pinned for reproducibility: `588b3f4b2d0467aff54842a4b926551b69d5a66a`
- Files used: `data/udhr/udhr_eng.xml`, `udhr_hin.xml`, `udhr_tam.xml`, `udhr_kan.xml`
  (vendored unmodified in `corpus/raw/`)
- Underlying translations are sourced from **OHCHR** (UN Office of the High
  Commissioner for Human Rights). Each language file's correction history
  and named contributors are recorded in that repository under
  `data/status/status_<lang>.xml`. Highlights actually read from those files:
  - Hindi: OHCHR identified as source (2006); further corrections in 2006, 2007, 2014, 2017.
  - Tamil: the India-variant translation (used here) replaced an earlier
    Sri-Lanka-variant translation in 2017; a Sri-Lanka variant remains
    available separately as `udhr_tam_LK.xml` but was **not** used here.
  - Kannada: full version contributed via the Language Observatory (2006);
    corrections in 2015 and 2017.

**FLORES-200/FLORES+ was investigated first and rejected as inaccessible**,
not merely as a preference. Evidence:
- `openlanguagedata/flores` (the current FLORES+ maintainer) contains no
  data files, only a README pointing to
  `https://huggingface.co/datasets/openlanguagedata/flores_plus`.
- `facebookresearch/flores` (legacy FLORES-200) contains no committed
  sentence data either; its README points to a tinyurl redirect and to
  HuggingFace.
- Direct requests confirmed all three download paths are blocked from this
  environment: `huggingface.co` → HTTP 403, `dl.fbaipublicfiles.com` → HTTP 403,
  the tinyurl redirect → HTTP 403, all with response header
  `x-deny-reason: host_not_allowed`.
- The one FLORES-family repo with real committed data, `floresv1/data/flores_test_sets.tgz`,
  covers Nepali/Sinhala/Khmer/Pashto↔English only — none of our four
  required languages.
- A second candidate, the UDHR corpus bundled with **NLTK** (`nltk_data`,
  reachable via `raw.githubusercontent.com`), was also tested and rejected:
  every file in that package is truncated at a fixed ~10,000-byte cap (a
  known legacy artifact of the original UDHR-in-Unicode file format). Because
  non-Latin scripts use more bytes per character in UTF-8, this cap left
  English almost complete (through Article 29 of 30) but cut Hindi off
  mid-Article-7, Kannada mid-Article-8, and Tamil mid-Article-2 — nowhere
  near enough overlap for a 4-way aligned corpus. The `eric-muller/udhr`
  repository used here is the untruncated upstream source for that same
  NLTK package and does not have this problem (files are 16–43 KB, not
  capped at 10 KB).

Full chronology, including the exact commands and HTTP responses, is in
`NOTEBOOK.md`, Phase 2 — A1.

## Languages

| Language | ISO 639-3 | ISO 15924 (script) |
|---|---|---|
| English | eng | Latn |
| Hindi | hin | Deva |
| Tamil (India variant) | tam | Taml |
| Kannada | kan | Knda |

Tamil and Kannada satisfy the "two Dravidian languages" requirement.

## Size

**31 aligned units per language** (1 Preamble + 30 Articles — the UDHR has
exactly 30 articles by definition, so this count is fixed regardless of
source). This is confirmed identical across all four languages by
`experiments/validate_corpus.py`.

| Language | Units | Total characters | Total whitespace-split words | Avg. chars/unit |
|---|---|---|---|---|
| English | 31 | 10,239 | 1,681 | 330.3 |
| Hindi | 31 | 10,381 | 1,943 | 334.9 |
| Tamil | 31 | 12,714 | 1,141 | 410.1 |
| Kannada | 31 | 10,069 | 1,015 | 324.8 |

This is far smaller than FLORES-200's ~1,012 devtest sentences per
language — see Limitations.

## Domain

Formal, declarative legal/diplomatic register: the 1948 UN Universal
Declaration of Human Rights. Each unit is one Preamble clause-block or one
full numbered Article (Articles average well over 100 words; several
combine multiple legal clauses into one unit). This is a single, fixed,
73-year-old document translated by different individuals/organizations per
language at different times (see Source section) — it is not conversational
text, not user-generated text, and not representative of typical LLM input.

## Alignment

Alignment is by **UDHR article number**, not by sentence. Every translation
of the UDHR contains exactly one Preamble and exactly 30 numbered Articles,
so `preamble`, `article_01`, ..., `article_30` are guaranteed to denote the
same legal content across all four languages — this was verified
structurally (all four source XML files were parsed and confirmed to
contain exactly articles 1–30, no more, no fewer) rather than assumed.

This is coarser than FLORES-200's sentence-level alignment. Where an
article contains multiple paragraphs (e.g., Article 2's two clauses, or
Article 11's two ordered sub-clauses), all paragraphs belonging to that
article are concatenated into a single line for that unit — see
Preprocessing. Sub-article-level (paragraph- or sentence-level) alignment
was not attempted because paragraph counts inside a given article are not
always equal across languages (e.g. the Preamble has 10 paragraphs in
English and Hindi, 9 in Tamil, 8 in Kannada — the article/preamble
boundary is the only boundary guaranteed to be structurally identical
across all four files).

## Preprocessing

Performed by `scripts/prepare_corpus.py`, applied identically and only
where justified:

1. Parse each source XML file with `xml.etree.ElementTree`.
2. For each unit (preamble, article 1–30), collect all `<para>` elements
   anywhere inside it (some articles nest paragraphs inside
   `<orderedlist><listitem>`, e.g. Article 11 and Article 13 — confirmed by
   inspection of the raw XML) and join their text with a single space.
3. Collapse any embedded newline inside a paragraph's text to a space.
4. Strip leading/trailing whitespace from the resulting line.
5. Write one line per unit, in a fixed order (preamble, article_01 ...
   article_30), identical across all four language files, plus
   `manifest.tsv` recording which line number corresponds to which unit id.

**Not done, deliberately:** no lowercasing, no Unicode normalization, no
punctuation stripping, no tokenization. `fertility.py`'s preprocessing
choices are not carried over here — this corpus is meant to be measured by
A3 in as close to its raw form as possible; any normalization decisions
belong in A3's tokenizer-comparison script, not baked into the corpus
itself, so that A3 can test the effect of normalization choices rather than
inherit ours silently.

## Limitations

**Size.** 31 aligned units is small — roughly 3% of FLORES-200's per-language
devtest size. Fertility statistics computed on this corpus (mean, variance)
will have much wider uncertainty than statistics computed on FLORES-200, and
single long/short articles can swing the average noticeably. Treat any A3
result from this corpus as indicative, not conclusive, and prefer to
re-run A3 against FLORES-200 or another sentence-level parallel corpus if
the user is able to supply one (see NOTEBOOK.md for exactly what was
attempted and blocked).

**Domain bias.** All 31 units come from a single 1948 legal document. UDHR
prose is formal, sentence-dense, and uses fixed legal formulae ("Everyone
has the right to...", "No one shall be...") that repeat across articles.
This is about as far as possible from typical production LLM traffic
(short conversational turns, code, questions, mixed-register text), and
any tokenizer-fertility gap measured here may not generalize to production
prompts at all — legal register is known in the MT/NLP literature to
tokenize differently (often more predictably, sometimes more verbosely)
than conversational text.

**Translation provenance is not uniform.** Unlike FLORES-200, where all
languages are translated from the same source sentences by a coordinated
professional pipeline at the same time, these four translations were
produced by different translators/organizations at different times (Hindi:
OHCHR-attributed, corrected 2006–2017; Tamil: India-variant substituted in
for a Sri-Lanka variant in 2017; Kannada: contributed via the Language
Observatory in 2006, corrected 2015/2017). Cross-language differences in
tokenizer fertility measured on this corpus could partly reflect
translator style/era rather than purely the script or tokenizer.

**Article-level, not sentence-level, alignment.** A single "unit" here can
be a full paragraph or several sentences concatenated together (see
Alignment). This is coarser than what a tokenizer-fertility comparison
normally uses. It still supports a valid corpus-level fertility
comparison (total tokens / total words, or per-unit token counts), but it
cannot support any analysis that requires knowing exact sentence
boundaries within a unit.

**What this corpus can establish:** whether a real tokenizer-fertility gap
between English/Hindi/Tamil/Kannada persists on independently-sourced,
human-translated, real text outside the original 10-sentence smoke-test
sample — i.e., whether Phase 1's finding is an artifact of that specific
10-sentence sample or something broader.

**What it cannot establish:** the magnitude of the gap on production-like
text, sentence-level fertility variance, or whether the finding holds for
languages/domains beyond formal legal prose.
