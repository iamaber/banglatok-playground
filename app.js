import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";
import {
  decode as decodeO200k,
  decodeGenerator as decodeO200kPieces,
  encode as encodeO200k,
} from "https://esm.sh/gpt-tokenizer@4.1.2/model/gpt-4o";
import {
  decode as decodeCl100k,
  decodeGenerator as decodeCl100kPieces,
  encode as encodeCl100k,
} from "https://esm.sh/gpt-tokenizer@4.1.2/model/gpt-4";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "./models/";
env.useBrowserCache = true;

const MODELS = [
  {
    key: "banglatok-bn-bpe-58k",
    label: "banglatok bn-bpe-58k",
    vocab: 57925,
    kind: "hf",
    tag: "this project",
  },
  {
    key: "banglatok-bn-bpe-28k",
    label: "banglatok bn-bpe-28k",
    vocab: 27760,
    kind: "hf",
    tag: "this project",
  },
  { key: "banglabert", label: "BanglaBERT (csebuetnlp)", vocab: 32000, kind: "hf" },
  {
    key: "bangla-bert-base",
    label: "Bangla-BERT (sagorsarker)",
    vocab: 101975,
    kind: "hf",
  },
  { key: "muril-base-cased", label: "MuRIL (Google)", vocab: 197285, kind: "hf" },
  {
    key: "bert-base-multilingual-cased",
    label: "mBERT cased",
    vocab: 119547,
    kind: "hf",
  },
  {
    key: "xlm-roberta-base",
    label: "XLM-R / mE5 / BGE-M3",
    vocab: 250002,
    kind: "hf",
  },
  { key: "qwen3-0.6b", label: "Qwen3 0.6B", vocab: 151669, kind: "hf" },
  {
    key: "o200k_base",
    label: "tiktoken o200k_base (GPT-4o)",
    vocab: 200019,
    kind: "tiktoken",
    encode: encodeO200k,
    decode: decodeO200k,
    decodePieces: decodeO200kPieces,
  },
  {
    key: "cl100k_base",
    label: "tiktoken cl100k_base (GPT-4)",
    vocab: 100277,
    kind: "tiktoken",
    encode: encodeCl100k,
    decode: decodeCl100k,
    decodePieces: decodeCl100kPieces,
  },
];

const EXAMPLES = {
  bangla: "আমি বাংলায় গান গাই। আজ আকাশ পরিষ্কার।",
  banglish: "ami banglay gaan gai! aj akash porishkar.",
  mixed: "meetingটা ১০টায়, don't be late 🙂",
  long:
    "বাংলাদেশের প্রাকৃতিক সৌন্দর্য অসাধারণ। The Sundarbans is the largest " +
    "mangrove forest in the world. আমি প্রতিদিন সকালে ধানক্ষেতের পাশ দিয়ে " +
    "হেঁটে যাই, আর পথে দেখা হয় পরিচিত মানুষের সাথে।",
};

const textarea = document.querySelector("#text");
const compareButton = document.querySelector("#compare");
const statusElement = document.querySelector("#status");
const summaryElement = document.querySelector("#summary");
const tableBody = document.querySelector("#results-body");
const errorsElement = document.querySelector("#errors");
const piecesLabel = document.querySelector("#pieces-label");
const piecesElement = document.querySelector("#pieces");

const encoder = new TextEncoder();
const tokenizerCache = new Map();
const rows = new Map();
const exampleButtons = [...document.querySelectorAll("[data-example]")];
const PIECES_LIMIT = 240;
const MAX_INPUT_CHARS = 20000;

let runId = 0;

function setBusy(busy) {
  compareButton.disabled = busy;
  exampleButtons.forEach((button) => {
    button.disabled = busy;
  });
  statusElement.dataset.busy = busy ? "true" : "false";
}

function normalizeText(value) {
  return value.normalize("NFC").replace(/\s+/g, " ").trim();
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function escapeHtml(value) {
  return String(value).replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[character],
  );
}

function getHfTokenizer(key) {
  if (!tokenizerCache.has(key)) {
    const promise = AutoTokenizer.from_pretrained(key).catch((error) => {
      tokenizerCache.delete(key);
      throw error;
    });
    tokenizerCache.set(key, promise);
  }
  return tokenizerCache.get(key);
}

async function loadModel(model) {
  if (model.kind !== "hf") {
    return {};
  }
  return { tokenizer: await getHfTokenizer(model.key) };
}

function warmUp(model, prepared, text) {
  if (model.kind === "hf") {
    prepared.tokenizer.encode(text, { add_special_tokens: false });
  } else {
    model.encode(text);
  }
}

function measure(model, prepared, text) {
  const words = text.split(" ").filter(Boolean).length;
  const bytes = encoder.encode(text).length;
  const row = { ...model, error: null, pieces: [] };

  if (prepared.error) {
    row.error = prepared.error;
    return row;
  }

  try {
    const times = [];
    let ids = [];

    if (model.kind === "hf") {
      const { tokenizer } = prepared;
      for (let run = 0; run < 3; run += 1) {
        const started = performance.now();
        ids = Array.from(tokenizer.encode(text, { add_special_tokens: false }), Number);
        times.push(performance.now() - started);
      }
      row.lossless =
        tokenizer.decode(ids, { skip_special_tokens: true }) === text;
    } else {
      for (let run = 0; run < 3; run += 1) {
        const started = performance.now();
        ids = model.encode(text);
        times.push(performance.now() - started);
      }
      row.lossless = model.decode(ids) === text;
    }

    row.tokens = ids.length;
    row.tokensPerWord = words ? ids.length / words : 0;
    row.bytesPerToken = ids.length ? bytes / ids.length : 0;
    row.ms = median(times);
  } catch (error) {
    row.error = (error && error.message) || String(error);
  }

  return row;
}

function collectPieces(row, prepared, text) {
  try {
    if (row.kind === "hf") {
      return prepared.tokenizer
        .tokenize(text, { add_special_tokens: false })
        .slice(0, PIECES_LIMIT);
    }
    const ids = row.encode(text).slice(0, PIECES_LIMIT + 8);
    return [...row.decodePieces(ids)].slice(0, PIECES_LIMIT);
  } catch (error) {
    return [];
  }
}

function tag(row) {
  return row.tag ? `<span class="tag">${escapeHtml(row.tag)}</span>` : "";
}

function render() {
  const list = [...rows.values()].sort((a, b) => {
    if (a.error && !b.error) return 1;
    if (!a.error && b.error) return -1;
    if (a.error && b.error) return a.label.localeCompare(b.label);
    return a.tokens - b.tokens;
  });
  const valid = list.filter((row) => !row.error);
  const best = valid.length ? Math.min(...valid.map((row) => row.tokens)) : null;

  tableBody.innerHTML = list
    .map((row) => {
      if (row.error) {
        return `<tr>
          <td class="name">${escapeHtml(row.label)}${tag(row)}</td>
          <td class="num">${row.vocab.toLocaleString()}</td>
          <td colspan="5" class="muted">unavailable in this browser</td>
        </tr>`;
      }
      return `<tr class="${row.tokens === best ? "best" : ""}">
        <td class="name">${escapeHtml(row.label)}${tag(row)}</td>
        <td class="num">${row.vocab.toLocaleString()}</td>
        <td class="num">${row.tokens}</td>
        <td class="num">${row.tokensPerWord.toFixed(3)}</td>
        <td class="num">${row.bytesPerToken.toFixed(2)}</td>
        <td>${row.lossless ? '<span class="ok">yes</span>' : '<span class="no">no</span>'}</td>
        <td class="num">${row.ms.toFixed(1)}</td>
      </tr>`;
    })
    .join("");

  const bestRow = valid.find((row) => row.tokens === best);
  summaryElement.textContent = bestRow
    ? `Shortest: ${bestRow.label} — ${bestRow.tokens} tokens ` +
      `(${bestRow.tokensPerWord.toFixed(3)}/word). ` +
      `Lossless: ${valid.filter((row) => row.lossless).length}/${valid.length}.`
    : "No tokenizer results.";

  if (bestRow) {
    const shown = bestRow.pieces.length;
    piecesLabel.textContent =
      `${bestRow.label} · ${bestRow.tokens} tokens` +
      (shown && shown < bestRow.tokens ? ` · first ${shown} pieces` : "");
    piecesElement.textContent = shown
      ? bestRow.pieces.map((piece) => (piece === "" ? "∅" : piece)).join(" · ")
      : "Token pieces are unavailable for this tokenizer.";
  } else {
    piecesLabel.textContent = "No comparison yet.";
    piecesElement.textContent = "";
  }

  const failures = list.filter((row) => row.error);
  if (failures.length) {
    errorsElement.hidden = false;
    errorsElement.textContent = `Some tokenizers could not load: ${failures
      .map((row) => `${row.label} (${row.error.slice(0, 160)})`)
      .join("; ")}`;
  } else {
    errorsElement.hidden = true;
    errorsElement.textContent = "";
  }
}

async function runComparison() {
  if (statusElement.dataset.busy === "true") {
    return;
  }

  const text = normalizeText(textarea.value);
  if (!text) {
    statusElement.textContent = "Enter some text first.";
    return;
  }

  if (text.length > MAX_INPUT_CHARS) {
    statusElement.textContent =
      `Input is ${text.length.toLocaleString()} characters; the limit is ` +
      `${MAX_INPUT_CHARS.toLocaleString()}. Shorten it and compare again.`;
    return;
  }

  const token = (runId += 1);
  rows.clear();
  render();
  setBusy(true);
  statusElement.textContent = `Loading tokenizers… 0/${MODELS.length}`;

  try {
    const prepared = new Map();
    // Load one tokenizer at a time, then warm each up once, so no timing run
    // overlaps another tokenizer's download or first-call compilation.
    for (const [index, model] of MODELS.entries()) {
      if (token !== runId) {
        return;
      }

      let ready = {};
      try {
        ready = await loadModel(model);
        warmUp(model, ready, text);
      } catch (error) {
        ready = { error: (error && error.message) || String(error) };
      }
      prepared.set(model.key, ready);
      statusElement.textContent = `Loading tokenizers… ${index + 1}/${MODELS.length}`;
    }

    if (token !== runId) {
      return;
    }

    statusElement.textContent = "Measuring tokenizers…";
    for (const model of MODELS) {
      if (token !== runId) {
        return;
      }
      rows.set(model.key, measure(model, prepared.get(model.key), text));
    }

    if (token !== runId) {
      return;
    }

    // Only the row the page displays needs token pieces.
    const best = [...rows.values()]
      .filter((row) => !row.error)
      .sort((a, b) => a.tokens - b.tokens)[0];
    if (best) {
      best.pieces = collectPieces(best, prepared.get(best.key), text);
    }

    statusElement.textContent = "Done";
  } finally {
    if (token === runId) {
      setBusy(false);
    }
  }

  render();
}

compareButton.addEventListener("click", runComparison);

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    textarea.value = EXAMPLES[button.dataset.example];
    runComparison();
  });
});

textarea.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    runComparison();
  }
});
