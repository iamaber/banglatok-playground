"""Export fast tokenizers as local files for the static Space.

The browser uses transformers.js, which can only load fast tokenizers
(``tokenizer.json``). Three of the comparison models publish only a
``vocab.txt``, so this script converts them once and writes the files the
Space serves under ``models/``.

Run from the repository root:

    uv run --with transformers==4.57.6 --with sentencepiece --with protobuf \\
        python scripts/export_tokenizers.py
"""

from pathlib import Path

from transformers import AutoTokenizer

MODELS = {
    "banglatok-bn-bpe-58k": "iamaber/banglatok-bn-bpe-58k",
    "banglatok-bn-bpe-28k": "iamaber/banglatok-bn-bpe-28k",
    "banglabert": "csebuetnlp/banglabert",
    "bangla-bert-base": "sagorsarker/bangla-bert-base",
    "muril-base-cased": "google/muril-base-cased",
    "bert-base-multilingual-cased": "google-bert/bert-base-multilingual-cased",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}


def main():
    output = Path(__file__).resolve().parent.parent / "models"
    for name, repo_id in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(target)
        files = sorted(path.name for path in target.iterdir())
        print(f"{repo_id} -> models/{name}: {type(tokenizer).__name__}")
        print(f"  {', '.join(files)}")


if __name__ == "__main__":
    main()
