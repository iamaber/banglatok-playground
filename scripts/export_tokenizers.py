"""Export fast tokenizers as local files for the static Space.

The browser uses transformers.js, which can only load fast tokenizers
(``tokenizer.json``). Three of the comparison models publish only a
``vocab.txt``, so this script converts them once and writes the files the
Space serves under ``models/``.

The other models are not handled here:

* ``bert-base-multilingual-cased``, ``xlm-roberta-base`` and ``qwen3-0.6b``
  publish a ``tokenizer.json`` already; download the Hub files verbatim.
* The banglatok tokenizers also publish a ``tokenizer.json``; download it and
  run ``scripts/patch_tokenizers.py`` instead. Re-saving it here would corrupt
  the intentional index gaps in its vocabulary.

Every model is pinned to the revision recorded in ``models/PROVENANCE.md``.

Run from the repository root, with the versions pinned in
``scripts/requirements.txt``:

    uv run --with-requirements scripts/requirements.txt \\
        python scripts/export_tokenizers.py
"""

from pathlib import Path

from transformers import AutoTokenizer

MODELS = {
    "banglabert": ("csebuetnlp/banglabert", "9ce791f330578f50da6bc52b54205166fb5d1c8c"),
    "bangla-bert-base": (
        "sagorsarker/bangla-bert-base",
        "875aa80a42ec196c16bd931ae5d85ad949f58b16",
    ),
    "muril-base-cased": (
        "google/muril-base-cased",
        "afd9f36c7923d54e97903922ff1b260d091d202f",
    ),
}


def main():
    output = Path(__file__).resolve().parent.parent / "models"
    for name, (repo_id, revision) in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, revision=revision, use_fast=True
        )
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(target)
        files = sorted(path.name for path in target.iterdir())
        print(f"{repo_id}@{revision[:12]} -> models/{name}: {type(tokenizer).__name__}")
        print(f"  {', '.join(files)}")


if __name__ == "__main__":
    main()
