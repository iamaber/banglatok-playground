---
title: Bangla Tokenizer Playground
emoji: 🔤
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: other
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
- **Token pieces** — the strings emitted by the selected tokenizer. The
  playground generates them on demand and shows up to 240 pieces.

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
converts only those three to fast tokenizers, pinned to the revisions in
[models/PROVENANCE.md](models/PROVENANCE.md). The mBERT, XLM-R and Qwen3
`tokenizer.json` files are the exact files published on the Hub and are copied
verbatim, not re-saved.

The banglatok files are the published `tokenizer.json` with one patch: the
tokenizers WASM build used by transformers.js does not honour the
`MergedWithNext` pre-tokenizer behaviour, so it emits each space as a separate
token. `scripts/patch_tokenizers.py` rewrites that split as `\s*\S+`, which the
browser implements correctly and which produces the same pieces as the Python
`tokenizers` library. Without the patch, every multi-word input is overcounted.

The first comparison downloads about 35 MB of tokenizer files. Later visits use
the browser cache. Input is capped at 20,000 normalized characters so a single
comparison keeps the page responsive.

The browser libraries are loaded from their CDNs at the versions written in the
`app.js` import URLs; `package.json` records the same versions under `cdnPins`
(nothing is installed with npm, and there is no build step);
the maintenance-script versions and Python 3.12 are pinned in
`scripts/requirements.txt` and `.python-version`. A
Content-Security-Policy limits script sources to this origin and the two pinned
CDN origins (`cdn.jsdelivr.net`, `esm.sh`). `fetch`/XHR additionally allows
Hugging Face's `*.hf.co` model-file CDN, which serves large bundled files after
the Space redirects same-origin requests. The policy allows WebAssembly
compilation and disables plugins, frames, form submission and `<base>`. It is
defence in depth, not a guarantee: the two CDN bundles still run with full page
privileges, and the policy does not restrict navigation. The page's own code
never sends pasted text anywhere.

## Local development

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open <http://127.0.0.1:8000>. Binding to loopback keeps the working tree,
including `.git/`, off your network.

To regenerate the bundled tokenizer files, follow the steps in order. Every
download is pinned to the revision recorded in
[models/PROVENANCE.md](models/PROVENANCE.md), and the `hf` CLI comes from the
pinned `huggingface_hub` in `scripts/requirements.txt`.

```bash
# 1. Convert the three models that publish only a vocab.txt.
uv run --with-requirements scripts/requirements.txt \
  python scripts/export_tokenizers.py

# 2. Copy the models that already publish tokenizer.json, verbatim.
hf download google-bert/bert-base-multilingual-cased \
  tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt config.json \
  --revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 \
  --local-dir models/bert-base-multilingual-cased
hf download FacebookAI/xlm-roberta-base \
  tokenizer.json tokenizer_config.json special_tokens_map.json \
  sentencepiece.bpe.model config.json \
  --revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 \
  --local-dir models/xlm-roberta-base
hf download Qwen/Qwen3-0.6B \
  tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json \
  merges.txt vocab.json chat_template.jinja config.json \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir models/qwen3-0.6b

# 3. Download the published banglatok files, then apply the pre-tokenizer patch.
hf download iamaber/banglatok-bn-bpe-58k \
  tokenizer.json tokenizer_config.json special_tokens_map.json \
  --revision 7e59b254d50ad60adec3747206d251abf0310558 \
  --local-dir models/banglatok-bn-bpe-58k
hf download iamaber/banglatok-bn-bpe-28k \
  tokenizer.json tokenizer_config.json special_tokens_map.json \
  --revision 4450dd6184bfdce8078ea428179ab507ab70f117 \
  --local-dir models/banglatok-bn-bpe-28k
# The patch script is standard-library only, so it needs no pinned packages.
uv run python scripts/patch_tokenizers.py

# 4. Verify the bundled files match the recorded hashes.
shasum -a 256 -c models/CHECKSUMS.sha256
```

Use the original published `tokenizer.json` for the banglatok models: their
vocabulary contains intentional index gaps, and re-saving it can corrupt it.
That is why `scripts/export_tokenizers.py` never touches those directories.

## Licenses

The application code is MIT; see [LICENSE](LICENSE).

The Space is marked `other` because the bundle is mixed. The bundled tokenizer
files keep their upstream licenses, recorded per directory in
`models/*/LICENSE`: the banglatok models and training corpus are CC BY-SA 4.0,
BanglaBERT is CC BY-NC-SA 4.0, and the rest are MIT or Apache-2.0. See
[NOTICE.md](NOTICE.md) for the full list. The MIT license covers this Space's
own code, not the bundled tokenizer data.
