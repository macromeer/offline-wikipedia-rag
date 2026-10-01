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
ollama pull qwen3:8b            # answer model; must support tool calling
ollama pull mistral:7b          # article selection, only for --retrieval kiwix

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
./run.sh                                         # interactive chat
./run.sh --question "What is machine learning?"  # single question
```

The interactive mode is a chat: follow-up questions ("and when was it founded?") use the last few questions and answers. Type `/reset` to start a new conversation and `quit` to exit.

```
--zim PATH               ZIM file (default: $WIKI_ZIM, else newest complete *.zim
                         in ~/wikipedia-offline, ~/Downloads, /data/wikipedia, /var/lib/kiwix)
--model NAME             answer model (default: auto-detect)
--max-results N          passages per search (default: 8-12 by question complexity)
--max-rounds N           rounds of searches before the model must answer (default: 2)
--history-turns N        earlier turns kept for follow-ups (default: 4; 0 disables)
--no-tools               retrieve once and answer in one call (no tool loop, no history);
                         also used automatically for models without tool calling
--retrieval zim|kiwix    zim (default): read the ZIM directly; kiwix: the v1 pipeline
                         over kiwix-serve HTTP, kept as a baseline for evaluation
--selection-model NAME   article selection model, --retrieval kiwix only
--kiwix-url URL          Kiwix server for source links (default: http://localhost:8080)
--no-auto-start          don't start kiwix-serve automatically
```

Partially downloaded ZIM files are detected from their header and skipped, so you can keep a download running in the same folder.

### Models

The answer model is auto-detected in this order: Gemma 4, Qwen 3.6/3.5, Qwen 3, then Llama 3.1, Gemma 2, Mistral and others. It never auto-selects coder, embedding, reranker or reasoning (`r1`, DeepSeek) models, nor models under 4B parameters unless nothing else is installed. Override with `--model`.

The chat needs a model with tool calling (`ollama show MODEL` lists `tools` under Capabilities; Gemma 4, Qwen 3.x and Llama 3.1 have it). With a model that lacks it, each question is retrieved once and answered without the tool loop or history.

The v1 pipeline (`--retrieval kiwix`) also uses a selection model, auto-detected in the order Qwen 3.6/3.5, Gemma 4, Qwen 3, Qwen 2.5, Mistral, Hermes 3, Llama 3.1; override with `--selection-model`.

### Example

```
✓ Wikipedia ZIM: wikipedia_en_all_nopic_2026-06.zim (dated 2026-06-17, full-text index)
✓ Kiwix server started at http://localhost:8080
✓ Model: gemma4:26b (searches with tools, up to 2 round(s))

❓ Your question: What is the capital of Australia?
  🔎 search "capital of Australia" → 8 passage(s), 8 new

📖 Answer:
   The capital city of Australia is Canberra [1].
   ...
📚 Sources (click to open):
   [1] Canberra
       http://localhost:8080/content/wikipedia_en_all_nopic_2026-06/Canberra
⏱️  3.5s, first answer text after 2.2s; 1 tool call(s), 8 passage(s) shown, largest prompt 5920 tokens

❓ Your question: How many people live there?
  🔎 search "Canberra population" → 8 passage(s), 7 new

📖 Answer:
   At the 2021 census, Canberra had 452,670 residents [6][11][13]. ...
📚 Sources (click to open):
   [6] Canberra > Demographics
       http://localhost:8080/content/wikipedia_en_all_nopic_2026-06/Canberra#Demographics
   ...
⏱️  5.6s, first answer text after 3.8s; 1 tool call(s), 8 passage(s) shown, largest prompt 3403 tokens
```

Without kiwix-serve the sources are listed as `Title > Section` only.

## How It Works

```
Question → model calls search_wikipedia("...") → exact title lookups + ZIM full-text
search (libzim) → sections → chunks → BM25 rank → numbered passages → model answers
(or searches again, at most 2 rounds) → streamed answer with [n] citations
```

1. **Tool loop**: the model gets two tools, `search_wikipedia(query)` and `read_section(title, section)`. It has to search before answering: if it answers from memory, that answer is dropped, a search for the question is run for it, and it is asked again (models skip the search surprisingly often; Qwen 3.6 almost always does). After two rounds of tool calls it must answer.
2. **Find articles**: word spans of the query are looked up as exact titles (redirects followed, so "capital of australia" finds *Canberra*; disambiguation pages such as *ETF* are resolved to the listed article the full-text search agrees with). The ZIM's built-in Xapian index supplies further candidates.
3. **Split**: each candidate article is split into its lead and sections (infoboxes, navboxes, references and citation markers removed) and packed into chunks of up to ~1,200 tokens, each prefixed with `Title > Section`.
4. **Rank**: chunks are scored with BM25 using whole-Wikipedia term statistics, fused with the article's rank; the lead of every exact title match is always included. The top 8-12 go back to the model, numbered.
5. **Answer**: streamed to the terminal with inline `[n]` citations. Numbers stay the same for the whole conversation. Under the answer, the cited passages are listed, and any `[n]` that matches no retrieved passage is flagged.
6. **History**: the last 4 questions and answers are kept, plus the titles of the passages they cited; passage text is not carried over, so a follow-up searches again.

Retrieval needs no server and typically takes 0.1-0.6 s on the full English dump; a question with very common words can take 1-2 s the first time, until their term statistics are cached in `~/.cache/offline-wikipedia-rag/`. On the dev box (RTX PRO 4000, `gemma4:26b`), 11 test questions took 3.5-7.5 s end to end for simple ones, with the first answer text after 2-6 s, and 9-13 s for explanations and comparisons; `qwen3.6:35b` took 2-6 s for the same questions. The v1 pipeline (kiwix-serve search, LLM article selection over abstracts) is still available with `--retrieval kiwix`; see [docs/TWO_STAGE_AI_PIPELINE.md](docs/TWO_STAGE_AI_PIPELINE.md) and [docs/AUTOMATIC_SETUP.md](docs/AUTOMATIC_SETUP.md).

## Testing

```bash
uv run pytest -m "not integration"      # unit tests, no ZIM/Kiwix/Ollama needed
uv run pytest                           # also integration tests on ZIMs found on disk
uv run pytest tests/ --cov=. --cov-report=html
```

Unit tests cover section parsing and chunking (on fixture pages in `tests/fixtures/`), title matching and ranking, the tool loop against a scripted Ollama (forced search, round cap, streaming, history, citation checks), the tools, search-term extraction, model detection, ZIM discovery and context sizing. Integration tests run against `~/wikipedia-dev/wikipedia_en_100_2026-08.zim` (override with `WIKI_DEV_ZIM`) and the full English ZIM if present (the chat tests also need Ollama; pick the model with `WIKI_TEST_MODEL`); they skip otherwise. CI runs the unit tests on Python 3.10-3.12 for every PR.

## Project Layout

```
offline-wikipedia-rag/
├── wikipedia_rag_kiwix.py              # entry point (run.sh calls it)
├── wikirag/
│   ├── cli.py                          # arguments, terminal chat, output
│   ├── agent.py                        # tool loop, streaming, chat history
│   ├── tools.py                        # search_wikipedia / read_section
│   ├── citations.py                    # citation table and [n] checks
│   ├── llm.py                          # Ollama model detection and calls
│   ├── oneshot.py                      # --no-tools path and the v1 kiwix pipeline
│   ├── kiwix.py / zimfiles.py          # kiwix-serve links, ZIM discovery
│   └── questions.py                    # question complexity → passage count
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
