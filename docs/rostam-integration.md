# Rostam integration

rembed is the pure-Go local embedder behind [Rostam](https://docs.rostamlabs.com/).
When Rostam is configured with a local model, it embeds text **in-process** via
rembed — no cloud embedding API, no ONNX Runtime, no build tag. It's compiled
into every Rostam binary and image.

## How it's wired

Rostam wraps `rembed.Embedder` behind its internal `semcache.Embedder`
interface. Selecting a local model calls `rembed.Load` and validates that the
loaded model's dimension matches Rostam's catalog entry. Because rembed is pure
Go, this added no cgo or native dependency to Rostam.

## Using it

Set `ROSTAM_EMBED_LOCAL` to a model. It accepts either a **curated catalog name**
or **any Hugging Face `org/model` id** (passed straight to `rembed.Load`):

```sh
# MCP server — agent memory + generic vector tools, semantically
ROSTAM_EMBED_LOCAL=minilm-l6-v2 rostam-server mcp

# LLM caching proxy — semantic response caching with no cloud key
ROSTAM_EMBED_LOCAL=minilm-l6-v2 rostam-server llm-proxy

# any Hugging Face id, not just the catalog
ROSTAM_EMBED_LOCAL=intfloat/multilingual-e5-base rostam-server mcp
```

The catalog curates six models (384-dim `minilm-l6-v2` (default),
`bge-small-en-v1.5`, `gte-small`; 768-dim `bge-base-en-v1.5`, `gte-base`,
`all-mpnet-base-v2`); run `rostam-server mcp -list-embed-models` to print it.

Weights download from the Hugging Face Hub on first use into rembed's cache
(`REMBED_CACHE`, default the OS user cache dir; Rostam also bridges its legacy
`ROSTAM_EMBED_MODELS_DIR` to it). Later starts reuse the cached files with no
network call.

## Where it applies

A configured local embedder powers every Rostam surface that needs embeddings
without a hosted endpoint:

- **MCP memory** (`remember`/`recall`) and the generic vector-DB tools (`upsert`
  auto-embeds `content`; `search` embeds `query_text`). See the [MCP server
  docs](https://docs.rostamlabs.com/server/mcp/#local-embeddings).
- **The LLM caching proxy** — a local embedder puts it in semantic mode, so
  paraphrased prompts hit the cache, with no embedding endpoint or API key. See
  the [LLM proxy docs](https://docs.rostamlabs.com/server/llm-proxy/).

`ROSTAM_EMBED_LOCAL` is mutually exclusive with the hosted `ROSTAM_EMBED_ENDPOINT`
— setting both is a startup error.

## Notes

- rembed derives each model's tokenizer, pooling, and normalization from the
  model itself, so Rostam's catalog only needs the model's public identity
  (name, dimension, license).
- The embedder id Rostam stamps into cache scope keys is
  `local:<value of ROSTAM_EMBED_LOCAL>` — selecting the same model by its
  catalog short name and by its full Hub id builds separate caches, so pick one
  form and keep it.
