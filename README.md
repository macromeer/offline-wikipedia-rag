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
| Tools | [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/), kiwix-serve (installed by the setup script) |

## Manual Setup

```bash
# 1. Ollama and models (any instruction model ≥4B works; see "Models" below)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b          # selection
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
--selection-model NAME   article selection model (default: auto-detect)
--max-results N          number of articles (default: by question complexity)
--kiwix-url URL          Kiwix server (default: http://localhost:8080)
--no-auto-start          don't start kiwix-serve automatically
```

Partially downloaded ZIM files are detected from their header and skipped, so you can keep a download running in the same folder.

### Models

Auto-detection prefers, in order, Qwen 3.6/3.5, Gemma 4 and Qwen 3 families, then Qwen 2.5, Mistral, Hermes 3 and Llama 3.1 (synthesis prefers Gemma 4 first). It never auto-selects coder, embedding, reranker or reasoning (`r1`, DeepSeek) models, nor models under 4B parameters unless nothing else is installed. Override with `--model` / `--selection-model`.

### Example

```
✓ Wikipedia book: http://localhost:8080/content/wikipedia_en_all_nopic_2026-06
✓ Selection model: qwen3.6:35b
✓ Summarization model: gemma4:26b

🔍 Searching local Wikipedia for: What causes a solar eclipse?
  🤖 Selecting with qwen3.6:35b (using article abstracts)...
  📏 Selection: 545 prompt tokens (num_ctx 16384)
✓ AI selected 4 article(s): Solar eclipse, Hybrid solar eclipse, ...
🤖 Generating synthesis with gemma4:26b...
  📏 Synthesis: 2068 prompt tokens (num_ctx 16384)

📖 Answer:
   A solar eclipse is caused when the Moon passes between the Earth and the
   Sun, obscuring the view of the Sun from a specific part of the Earth [1]. ...

📚 Source Articles (click to open):
   [1] Solar eclipse
       http://localhost:8080/content/wikipedia_en_all_nopic_2026-06/Solar_eclipse
```

## How It Works

```
Question → term extraction → Kiwix search → abstract fetch →
Stage 1 (selection model) → full article fetch → Stage 2 (synthesis model)
→ Answer + clickable citations
```

1. **Search**: extracts likely article titles from the question, queries kiwix-serve's full-text search, and tries direct title lookups.
2. **Selection**: fetches each candidate's first paragraph; the selection model picks the 3-6 most relevant articles.
3. **Synthesis**: reads the selected articles and writes an answer with inline `[1][2]` citations and links to the local Kiwix pages.

More detail: [docs/TWO_STAGE_AI_PIPELINE.md](docs/TWO_STAGE_AI_PIPELINE.md), [docs/AUTOMATIC_SETUP.md](docs/AUTOMATIC_SETUP.md).

## Testing

```bash
uv run pytest -m "not integration"      # unit tests, no Kiwix/Ollama needed
uv run pytest tests/ --cov=. --cov-report=html
```

Unit tests cover search-term extraction, complexity estimation, model detection, ZIM discovery and context sizing. CI runs them on Python 3.10-3.12 for every PR.

## Project Layout

```
offline-wikipedia-rag/
├── wikipedia_rag_kiwix.py              # main application
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
