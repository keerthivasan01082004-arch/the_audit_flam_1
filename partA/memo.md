# A4 — Audit Recommendation Memo

## Recommendation

The original claim that **Hindi costs approximately 6× more to tokenize than English** was directionally reproducible under GPT-2, but it is not a stable language property.

## Evidence

We measured **31 aligned UDHR units across four languages**—English, Hindi, Tamil, and Kannada—using **two verified tokenizers** (GPT-2 and mBERT) and two denominators (tokens/word and tokens/byte): **4 × 2 × 31 = 248 measured data points**.

For mean tokens per word relative to English, GPT-2 measured Hindi at **8.22 tokens (7.38×)**, Tamil at **31.96 (28.69×)**, and Kannada at **27.13 (24.35×)**. Under mBERT, the corresponding results were Hindi **1.70 (1.36×)**, Tamil **4.62 (3.71×)**, and Kannada **5.02 (4.03×)**. Thus, the original approximately 6× Hindi finding broadly replicates under GPT-2, while the previously untested Tamil and Kannada gaps are substantially larger on that tokenizer.

The ratio is strongly **tokenizer- and metric-dependent**: replacing GPT-2 with mBERT reduces the ratios by roughly 5–8×, while changing the denominator from words to bytes reverses the direction under mBERT, where all three languages fall below English. A multiplier that changes materially with tokenizer and denominator is not a defensible universal language constant. These tokenizer ratios should not be treated as GPU cost figures, and the result is not established as universal across all text.

**Leadership recommendation:** Do not use a fixed language multiplier for routing or capacity planning. Measure token counts with the actual production tokenizer on representative production text for each language; do not assume Kannada or Tamil match Hindi.

**Caveat:** The measurements use one formal/legal corpus, the UDHR, rather than conversational traffic, and cover only two of many possible tokenizers.
