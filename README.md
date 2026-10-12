---
title: Bangla Tokenizer Playground
emoji: 🔤
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Compare Bangla and Banglish token counts in your browser
tags:
  - tokenizer
  - bangla
  - bengali
  - banglish
  - nlp
  - tokenizers
---

# Bangla Tokenizer Playground

Paste Bangla, Banglish, English or code-mixed text and compare exact token
counts across the tokenizers people use for Bangla today. Everything runs in
your browser; the text you paste is never uploaded.

The comparison includes the two [banglatok](https://github.com/iamaber/banglatok)
models, the standard multilingual models, and OpenAI's tiktoken encodings.

| Tokenizer | Vocabulary |
| --- | ---: |
| banglatok `bn-bpe-58k` | 57,925 |
| banglatok `bn-bpe-28k` | 27,760 |
| BanglaBERT (csebuetnlp) | 32,000 |
| Bangla-BERT (sagorsarker) | 101,975 |
| MuRIL (Google) | 197,285 |
| mBERT cased | 119,547 |
| XLM-R / mE5 / BGE-M3 | 250,002 |
| Qwen3 0.6B | 151,669 |
| tiktoken `o200k_base` (GPT-4o) | 200,019 |
| tiktoken `cl100k_base` (GPT-4) | 100,277 |

## What the table shows

- **Tokens** — the number of IDs emitted, without special tokens.
- **Tokens/word** — total IDs divided by the number of whitespace-separated
  words.
- **Bytes/token** — UTF-8 bytes divided by IDs; higher is more compact.
- **Lossless** — whether `decode(encode(text))` returns the same text after NFC
  and whitespace normalization. "no" means the round trip changed the text.
- **Encode (ms)** — the median of three runs, timed one tokenizer at a time
  after every tokenizer has loaded and been warmed up once. Absolute numbers
  depend on your device.

Every tokenizer sees the same NFC/whitespace-normalized input. The banglatok
models were trained on the
[banglatok training corpus](https://huggingface.co/datasets/iamaber/banglatok-training-corpus);
the other tokenizers never saw that split, so this is an unseen-source
comparison for them.

## How it works

The Space is static. Tokenizer files are served from `models/` and loaded with
[transformers.js](https://github.com/huggingface/transformers.js). The tiktoken
encodings run through [gpt-tokenizer](https://github.com/niieani/gpt-tokenizer).

Three upstream models publish only a `vocab.txt`, so `scripts/export_tokenizers.py`
converts them to fast tokenizers once. The mBERT, XLM-R and Qwen3 `tokenizer.json`
files are the exact files published on the Hub.

The banglatok files are the published `tokenizer.json` with one patch: the
tokenizers WASM build used by transformers.js does not honour the
`MergedWithNext` pre-tokenizer behaviour, so it emits each space as a separate
token. `scripts/patch_tokenizers.py` rewrites that split as `\s*\S+`, which the
browser implements correctly and which produces the same pieces as the Python
`tokenizers` library. Without the patch, every multi-word input is overcounted.

The first comparison downloads about 35 MB of tokenizer files. Later visits use
the browser cache.

## Local development

```bash
python -m http.server 8000
```

Open <http://localhost:8000>. To regenerate the bundled tokenizer files:

```bash
uv run --with transformers==4.57.6 --with sentencepiece --with protobuf \
  python scripts/export_tokenizers.py
hf download iamaber/banglatok-bn-bpe-58k tokenizer.json tokenizer_config.json \
  --local-dir models/banglatok-bn-bpe-58k
hf download iamaber/banglatok-bn-bpe-28k tokenizer.json tokenizer_config.json \
  --local-dir models/banglatok-bn-bpe-28k
uv run --with tokenizers python scripts/patch_tokenizers.py
```

Use the original published `tokenizer.json` for the banglatok models: their
vocabulary contains intentional index gaps, and re-saving it can corrupt it.
Then apply the pre-tokenizer patch above.

## Licenses

The application code is MIT; see [LICENSE](LICENSE).

The bundled tokenizer files keep their upstream licenses. The banglatok models
and training corpus are CC BY-SA 4.0. BanglaBERT is CC BY-NC-SA 4.0. See
[NOTICE.md](NOTICE.md) for the full list. The MIT license covers this Space's
own code, not the bundled tokenizer data.
