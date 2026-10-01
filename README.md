# 🌐 Offline Wikipedia RAG

**A private, offline AI assistant with the entire English Wikipedia at your fingertips.**

[![Tests](https://github.com/macromeer/offline-wikipedia-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/macromeer/offline-wikipedia-rag/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Buy me a pizza](https://img.shields.io/badge/Sponsor-🍕-8B4513?logo=github&logoColor=white)](https://github.com/sponsors/macromeer)

## TL;DR

- 100% offline: Ollama + Kiwix + local models → answers with inline citations
- Zero API keys, zero telemetry, zero cost
- Uses the Kiwix English Wikipedia dump without images (≈50 GB, 6M+ articles)
- One command (`./run.sh`) to start chatting; optional installer handles everything

## 🚀 Quick Start

```bash
git clone https://github.com/macromeer/offline-wikipedia-rag.git
cd offline-wikipedia-rag
./scripts/install.sh        # installs Ollama, models, uv env, newest Wikipedia dump
./run.sh                    # launches the assistant
```

Installer runtime: a few hours (mostly the download). Disk: ~65 GB free.

Already have Ollama, [uv](https://docs.astral.sh/uv/) and a Wikipedia ZIM? Just run `./run.sh`. It syncs the Python environment, checks Ollama, starts Kiwix on the newest complete ZIM it finds, picks the best local models, and stops Kiwix when you exit.

## Requirements

| Resource | Recommended |
| --- | --- |
| OS | Linux (Ubuntu 20.04+/Fedora 35+) or macOS |
| Disk | ≥65 GB free (≈50 GB Wikipedia + models) |
| RAM | 16 GB for 8B-class models; more for larger ones |
| CPU/GPU | Multi-core CPU. GPU optional (Ollama auto-detects CUDA/ROCm/Metal). |
| Tools | [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/); kiwix-serve optional, for clickable source links (installed by the setup script) |

## Manual Setup

```bash
# 1. Ollama and models (any instruction model ≥4B works; see "Models" below)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b          # selection (only for --retrieval kiwix)
ollama pull llama3.1:8b         # synthesis

# 2. Python environment
uv sync

# 3. kiwix-serve + newest Wikipedia ZIM (resumable; re-run to continue)
./scripts/setup_full_offline_wikipedia.sh                 # asks which edition
./scripts/setup_full_offline_wikipedia.sh --variant mini  # ~13 GB, intros only

# 4. Run
./run.sh
```

## Usage

```bash
./run.sh                                         # interactive
./run.sh --question "What is machine learning?"  # single question
```

Each question is answered on its own; the interactive mode does not keep chat history yet.

```
--zim PATH               ZIM file (default: $WIKI_ZIM, else newest complete *.zim
                         in ~/wikipedia-offline, ~/Downloads, /data/wikipedia, /var/lib/kiwix)
--model NAME             synthesis model (default: auto-detect)
--selection-model NAME   article selection model, --retrieval kiwix only
--max-results N          passages to retrieve (default: 8-12 by question complexity)
--retrieval zim|kiwix    zim (default): read the ZIM directly; kiwix: the v1 pipeline
                         over kiwix-serve HTTP, kept as a baseline for evaluation
--kiwix-url URL          Kiwix server for source links (default: http://localhost:8080)
--no-auto-start          don't start kiwix-serve automatically
```

Partially downloaded ZIM files are detected from their header and skipped, so you can keep a download running in the same folder.

### Models

Auto-detection prefers, in order, Qwen 3.6/3.5, Gemma 4 and Qwen 3 families, then Qwen 2.5, Mistral, Hermes 3 and Llama 3.1 (synthesis prefers Gemma 4 first). It never auto-selects coder, embedding, reranker or reasoning (`r1`, DeepSeek) models, nor models under 4B parameters unless nothing else is installed. Override with `--model` / `--selection-model`.

### Example

```
✓ Wikipedia ZIM: wikipedia_en_all_nopic_2026-06.zim (dated 2026-06-17, full-text index)
✓ Kiwix server started at http://localhost:8080
✓ Summarization model: gemma4:26b

🔍 Searching local Wikipedia for: What are the goals of NASA?
  🎯 Title matches: NASA
✓ Retrieved 8 passage(s) from 3 article(s) in 0.32s (14 articles scored)
  📄 NASA
  📄 NASA > Management > Strategic plan
  ...
🤖 Generating synthesis with gemma4:26b...
  📏 Synthesis: 2867 prompt tokens (num_ctx 16384)
⏱️  Total time: 9.6s

📖 Answer:
   NASA's primary goals involve expanding human knowledge, advancing space
   exploration, and driving technological and economic innovation [2]. ...

📚 Sources (click to open):
   [1] NASA
       http://localhost:8080/content/wikipedia_en_all_nopic_2026-06/NASA
   [2] NASA > Management > Strategic plan
       http://localhost:8080/content/wikipedia_en_all_nopic_2026-06/NASA#Strategic_plan
```

Without kiwix-serve the sources are listed as `Title > Section` only.

## How It Works

```
Question → exact title lookups + ZIM full-text search (libzim) → parse candidate
articles into sections → chunk → BM25 rank → synthesis model → answer with
section-level citations
```

1. **Find articles**: word spans of the question are looked up as exact titles (redirects followed, so "capital of australia" finds *Canberra*; disambiguation pages such as *ETF* are resolved to the listed article the full-text search agrees with). The ZIM's built-in Xapian index supplies further candidates.
2. **Split**: each candidate article is split into its lead and sections (infoboxes, navboxes, references and citation markers removed) and packed into chunks of up to ~1,200 tokens, each prefixed with `Title > Section`.
3. **Rank**: chunks are scored with BM25 using whole-Wikipedia term statistics, fused with the article's rank; the lead of every exact title match is always included.
4. **Synthesis**: the synthesis model answers from the top 8-12 passages with inline `[1][2]` citations.

Retrieval needs no server and typically takes 0.1-0.6 s on the full English dump; a question with very common words can take 1-2 s the first time, until their term statistics are cached in `~/.cache/offline-wikipedia-rag/`. The code lives in [retrieval/zim_store.py](retrieval/zim_store.py). The v1 pipeline (kiwix-serve search, LLM article selection over abstracts) is still available with `--retrieval kiwix`; see [docs/TWO_STAGE_AI_PIPELINE.md](docs/TWO_STAGE_AI_PIPELINE.md) and [docs/AUTOMATIC_SETUP.md](docs/AUTOMATIC_SETUP.md).

## Testing

```bash
uv run pytest -m "not integration"      # unit tests, no ZIM/Kiwix/Ollama needed
uv run pytest                           # also integration tests on ZIMs found on disk
uv run pytest tests/ --cov=. --cov-report=html
```

Unit tests cover section parsing and chunking (on fixture pages in `tests/fixtures/`), title matching and ranking, search-term extraction, model detection, ZIM discovery and context sizing. Integration tests run against `~/wikipedia-dev/wikipedia_en_100_2026-08.zim` (override with `WIKI_DEV_ZIM`) and the full English ZIM if present; they skip otherwise. CI runs the unit tests on Python 3.10-3.12 for every PR.

## Project Layout

```
offline-wikipedia-rag/
├── wikipedia_rag_kiwix.py              # main application
├── retrieval/zim_store.py              # libzim search, section parsing, chunking, BM25
├── run.sh                              # launcher (uv run)
├── pyproject.toml / uv.lock            # Python dependencies
├── scripts/
│   ├── install.sh                      # full installer
│   ├── setup_full_offline_wikipedia.sh # kiwix-serve + newest ZIM download
│   ├── start_offline_rag.sh            # run kiwix-serve on its own
│   └── bench_*.py                      # token speed benchmarks
├── tests/                              # pytest suite
└── docs/
```

## 🆘 Troubleshooting

- **"No Wikipedia ZIM file found"**: run `./scripts/setup_full_offline_wikipedia.sh`, or pass `--zim PATH`. A download still in progress is skipped on purpose.
- **"Ollama model not found"**: `ollama pull llama3.1:8b` (or any model listed under Models).
- **Out of memory**: pick smaller models with `--model` / `--selection-model`.

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more.

## 🤝 Contributing

Issues and PRs are welcome: GUIs, Docker images, alternative model profiles, performance or testing improvements. See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📜 License

MIT License - see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- [Kiwix](https://www.kiwix.org/) - offline Wikipedia technology
- [Ollama](https://ollama.com/) - local model runtime
- [Wikimedia Foundation](https://www.wikimedia.org/) - Wikipedia
- The open-weight model authors (Meta, Mistral AI, Alibaba Qwen, Google Gemma)
