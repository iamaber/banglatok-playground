"""Make the banglatok tokenizer JSON work in transformers.js.

The banglatok tokenizer uses ``Split`` on a space with ``MergedWithNext``.
The tokenizers WASM build used by transformers.js does not honour that
behaviour: it emits each space as a separate token, which overcounts every
multi-word input. Rewriting the split as a whole-chunk regex produces the same
pieces in both the Python ``tokenizers`` library and the browser.

Run after downloading the published ``tokenizer.json`` files:

    uv run python scripts/patch_tokenizers.py
"""

import json
from pathlib import Path

NAMES = ["banglatok-bn-bpe-58k", "banglatok-bn-bpe-28k"]
PRE_TOKENIZER = {
    "type": "Split",
    "pattern": {"Regex": r"\s*\S+"},
    "behavior": "Isolated",
    "invert": False,
}


def main():
    root = Path(__file__).resolve().parent.parent / "models"
    for name in NAMES:
        path = root / name / "tokenizer.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["pre_tokenizer"] = PRE_TOKENIZER
        path.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"patched {path}")


if __name__ == "__main__":
    main()
