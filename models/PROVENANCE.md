# Bundled tokenizer provenance

Every directory under `models/` is a copy of the tokenizer files for the
upstream repository below, pinned to the revision shown. `CHECKSUMS.sha256`
records a SHA-256 for every tracked file in `models/`; verify a checkout with:

```bash
shasum -a 256 -c models/CHECKSUMS.sha256
```

| Directory | Upstream repository | Revision | How it was produced |
| --- | --- | --- | --- |
| `banglatok-bn-bpe-58k` | [iamaber/banglatok-bn-bpe-58k](https://huggingface.co/iamaber/banglatok-bn-bpe-58k) | `7e59b254d50ad60adec3747206d251abf0310558` | Published files from the Hub, then `scripts/patch_tokenizers.py` |
| `banglatok-bn-bpe-28k` | [iamaber/banglatok-bn-bpe-28k](https://huggingface.co/iamaber/banglatok-bn-bpe-28k) | `4450dd6184bfdce8078ea428179ab507ab70f117` | Published files from the Hub, then `scripts/patch_tokenizers.py` |
| `banglabert` | [csebuetnlp/banglabert](https://huggingface.co/csebuetnlp/banglabert) | `9ce791f330578f50da6bc52b54205166fb5d1c8c` | `scripts/export_tokenizers.py` (publishes only `vocab.txt`) |
| `bangla-bert-base` | [sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base) | `875aa80a42ec196c16bd931ae5d85ad949f58b16` | `scripts/export_tokenizers.py` (publishes only `vocab.txt`) |
| `muril-base-cased` | [google/muril-base-cased](https://huggingface.co/google/muril-base-cased) | `afd9f36c7923d54e97903922ff1b260d091d202f` | `scripts/export_tokenizers.py` (publishes only `vocab.txt`) |
| `bert-base-multilingual-cased` | [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) | `3f076fdb1ab68d5b2880cb87a0886f315b8146f8` | Hub files copied verbatim |
| `xlm-roberta-base` | [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | Hub files copied verbatim |
| `qwen3-0.6b` | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | Hub files copied verbatim |

The banglatok revision is the one whose published `tokenizer.json` is the
basis for the patched file, and the export/verbatim revisions are the current
`main` commit at the time the files were bundled. Re-pinning means updating
this table, the constants in `scripts/export_tokenizers.py`, and
`CHECKSUMS.sha256` together.

The tiktoken `o200k_base` and `cl100k_base` encodings are not bundled here;
they ship inside `gpt-tokenizer` at the version pinned in `package.json`.

Licenses for each directory are listed in [../NOTICE.md](../NOTICE.md).
