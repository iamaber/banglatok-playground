import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const modelsDir = path.join(root, "models");

const PATCHED_PRE_TOKENIZER = {
  type: "Split",
  pattern: { Regex: "\\s*\\S+" },
  behavior: "Isolated",
  invert: false,
};

async function readJson(relativePath) {
  return JSON.parse(await readFile(path.join(root, relativePath), "utf8"));
}

async function readModelsFromApp() {
  const source = await readFile(path.join(root, "app.js"), "utf8");
  const start = source.indexOf("const MODELS = [");
  const end = source.indexOf("\n];", start);
  assert.ok(start !== -1 && end !== -1, "app.js exposes a MODELS array");

  return [
    ...source
      .slice(start, end)
      .matchAll(
        /key:\s*"([^"]+)"[\s\S]*?vocab:\s*(\d+)[\s\S]*?kind:\s*"(\w+)"/g,
      ),
  ].map((match) => ({
    key: match[1],
    vocab: Number(match[2]),
    kind: match[3],
  }));
}

test("every HF model in app.js resolves to a parseable bundled tokenizer", async () => {
  const models = await readModelsFromApp();
  assert.equal(models.length, 10, "all ten compared tokenizers are declared");

  const hfModels = models.filter((model) => model.kind === "hf");
  assert.equal(hfModels.length, 8);

  for (const model of hfModels) {
    const tokenizer = await readJson(`models/${model.key}/tokenizer.json`);
    assert.ok(tokenizer.model, `${model.key} has a model section`);
    assert.ok(model.vocab > 0, `${model.key} declares a vocabulary size`);
  }

  const tiktoken = models
    .filter((model) => model.kind === "tiktoken")
    .map((model) => model.key)
    .sort();
  assert.deepEqual(tiktoken, ["cl100k_base", "o200k_base"]);
});

test("MuRIL's declared vocabulary matches the bundled files", async () => {
  const models = await readModelsFromApp();
  const muril = models.find((model) => model.key === "muril-base-cased");
  const config = await readJson("models/muril-base-cased/config.json");
  const tokenizer = await readJson("models/muril-base-cased/tokenizer.json");
  const vocab = await readFile(
    path.join(modelsDir, "muril-base-cased/vocab.txt"),
    "utf8",
  );

  assert.equal(muril.vocab, 197285);
  assert.equal(config.vocab_size, 197285);
  assert.equal(Object.keys(tokenizer.model.vocab).length, 197285);
  assert.equal(vocab.split("\n").filter(Boolean).length, 197285);
});

test("the banglatok tokenizers carry the pre-tokenizer patch", async () => {
  for (const key of ["banglatok-bn-bpe-58k", "banglatok-bn-bpe-28k"]) {
    const tokenizer = await readJson(`models/${key}/tokenizer.json`);
    assert.deepEqual(tokenizer.pre_tokenizer, PATCHED_PRE_TOKENIZER, key);
  }
});

test("every tracked model file matches models/CHECKSUMS.sha256", async () => {
  const manifest = await readFile(
    path.join(modelsDir, "CHECKSUMS.sha256"),
    "utf8",
  );
  const entries = manifest
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [expected, relativePath] = line.split(/\s+/, 2);
      return { expected, relativePath };
    });

  assert.ok(entries.length > 0, "the manifest lists files");

  for (const { expected, relativePath } of entries) {
    const digest = createHash("sha256")
      .update(await readFile(path.join(root, relativePath)))
      .digest("hex");
    assert.equal(digest, expected, relativePath);
  }
});
