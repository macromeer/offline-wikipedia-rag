# Dense index (optional)

By default, passages are found with the ZIM's built-in full-text index and exact title lookups. That works well when the question shares words with the answer, and poorly when it does not: "Who was the first person to walk on the moon?" finds the film *A Walk on the Moon* but not *Neil Armstrong*, because the article does not use those words.

A dense index adds search by meaning. Every passage of the ZIM is embedded once with [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B); at question time the question is embedded too (through Ollama, `qwen3-embedding:0.6b`), and the closest passages are fused with the full-text results by reciprocal rank fusion. When an index exists for the ZIM in use, `./run.sh` uses it automatically.

Building it is a one-off job that is long for the full English Wikipedia: it embeds every passage on the GPU.

## What it costs

Measured on the dev box (2x RTX PRO 4000 Blackwell, 24 GB each) with `wikipedia_en_all_nopic_2026-06.zim`:

| | `--scope all` (default) | `--scope leads` |
| --- | --- | --- |
| What is embedded | every section | article introductions only |
| Passages | ~31M | ~7M |
| Text | ~6B tokens | ~1B tokens |
| Build time, 2 GPUs | ~16-18 h | ~3.5 h |
| Index size (512 dims) | ~20 GB | ~5 GB |
| RAM to build the search graph | about the index size | about the index size |

Passage and token counts come from a random sample of 3,000 articles; build time from the leads build (50-56k tokens/s per GPU) and, for `all`, extrapolated from it. Index sizes are ~650 bytes per passage (512 int8 dimensions plus the graph links), also extrapolated.

One GPU embeds about 50-56k tokens/s with the torch backend (the model is compiled with `torch.compile`; 34k without). FP8 was tried and dropped: faster, but it produced NaN vectors and lower accuracy. Through Ollama it is about 2-3x slower and uses one GPU, which is fine for small ZIMs but not for the full dump.

At question time the index is memory-mapped, so it does not need to fit in RAM; the embedding model adds ~1 GB of GPU memory in Ollama and ~0.1 s per search.

## Build

```bash
ollama pull qwen3-embedding:0.6b                               # used for questions

# GPU build (downloads PyTorch and the Hugging Face model the first time)
uv run --group dense-build python -m retrieval.build_dense     # every section
uv run --group dense-build python -m retrieval.build_dense --scope leads

# No PyTorch: through Ollama, one GPU, slow; fine for small ZIMs
uv run python -m retrieval.build_dense --backend ollama
```

Options: `--zim PATH` (default: the same ZIM `./run.sh` would pick), `--out DIR` (default `~/.local/share/offline-wikipedia-rag/dense-<ZIM uuid>`), `--devices cuda:0,cuda:1` (default: every GPU), `--dim N` (default 512; smaller is smaller and slightly less accurate), `--max-blocks N` (embed N blocks and stop, for a trial run).

The build is **resumable**: it works through the ZIM in blocks of 50,000 entries and writes each finished block to `shards/`. Stop it with Ctrl+C at any time and run the same command again to continue. When every block is done, the shards are loaded into one search graph (`index.usearch`) and deleted (`--keep-shards` keeps them).

**Leads first, everything later.** Running `--scope all` on a finished leads index upgrades it instead of starting over: only the sections that are not leads are embedded (into `shards-rest/`, also resumable), then added to the existing search graph. The leads index stays in use until the upgraded one replaces it. Total work is the same as building `--scope all` directly.

Free the GPUs first: each worker needs ~4 GB of GPU memory, and Ollama keeps models loaded (with `OLLAMA_KEEP_ALIVE=-1`, forever). `ollama stop MODEL` unloads one.

## Use

```bash
./run.sh                          # uses the index for this ZIM if it exists
./run.sh --no-dense               # full-text only
./run.sh --dense-index DIR        # an index somewhere else
```

The startup lines say which mode is active:

```
✓ Dense index: 31,012,345 passages (all sections); hybrid retrieval
```

## How it works

- **Passages** are the same chunks the full-text path uses (`Title > Section` plus up to ~1,200 tokens), cut to 2,000 characters for embedding. Redirects, disambiguation pages and `List of` articles are skipped.
- **Vectors**: the first 512 of the model's 1,024 dimensions (it is trained so that prefixes work), stored as int8 with one scale per vector, in a [usearch](https://github.com/unum-cloud/usearch) HNSW graph. Questions get the model's retrieval instruction prefix; passages none.
- **Keys**: each vector's 64-bit key holds the ZIM entry number, the chunk number within the article and a 16-bit checksum of the chunk text. A hit is turned back into text by re-reading the article from the ZIM. If the chunking code changes, the checksums stop matching and the app says the index needs rebuilding.
- **Fusion**: the 30 nearest passages are fetched for each search. Their articles are merged with the full-text candidates by reciprocal rank fusion, and each passage's dense rank is a fourth signal next to BM25 rank, article rank and exact-title match. A passage without any query word can now be chosen if the dense index found it.
- The index belongs to one ZIM file (matched by its uuid). A new Wikipedia dump needs a new index.
