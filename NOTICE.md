# Third-party notices

This Space bundles tokenizer files so the comparison runs without any network
calls after the first load. The application code is MIT. The tokenizer files
remain under their upstream licenses, and each directory under `models/` also
carries a `LICENSE` notice naming its SPDX identifier:

| Bundled directory | Upstream repository | License |
| --- | --- | --- |
| `models/banglatok-bn-bpe-58k` | [iamaber/banglatok-bn-bpe-58k](https://huggingface.co/iamaber/banglatok-bn-bpe-58k) | CC BY-SA 4.0 |
| `models/banglatok-bn-bpe-28k` | [iamaber/banglatok-bn-bpe-28k](https://huggingface.co/iamaber/banglatok-bn-bpe-28k) | CC BY-SA 4.0 |
| `models/banglabert` | [csebuetnlp/banglabert](https://huggingface.co/csebuetnlp/banglabert) | CC BY-NC-SA 4.0 |
| `models/bangla-bert-base` | [sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base) | MIT |
| `models/muril-base-cased` | [google/muril-base-cased](https://huggingface.co/google/muril-base-cased) | Apache-2.0 |
| `models/bert-base-multilingual-cased` | [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) | Apache-2.0 |
| `models/xlm-roberta-base` | [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | MIT |
| `models/qwen3-0.6b` | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache-2.0 |

The BanglaBERT tokenizer files are a derivative of the CC BY-NC-SA 4.0 model
and are used here for a non-commercial comparison. The banglatok tokenizer files
are distributed under CC BY-SA 4.0 with the model card's attribution.

The tiktoken `o200k_base` and `cl100k_base` encodings come from OpenAI's
[tiktoken](https://github.com/openai/tiktoken) (MIT) and are executed through
[gpt-tokenizer](https://github.com/niieani/gpt-tokenizer) (MIT).
