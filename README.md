# 🌐 Free Offline AI with Full Wikipedia Knowledge

**Chat with AI that has complete Wikipedia knowledge - 100% free, 100% offline, 100% private.**

This project combines specialized AI models with the complete English Wikipedia to give you an intelligent assistant that works entirely offline, requires no API keys, and respects your privacy.

[![Tests](https://github.com/macromeer/offline-wikipedia-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/macromeer/offline-wikipedia-rag/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/macromeer/offline-wikipedia-rag/branch/main/graph/badge.svg)](https://codecov.io/gh/macromeer/offline-wikipedia-rag)

## ✨ Features

### Why This Project?
Unlike ChatGPT or other cloud AI services, this gives you:
- **Complete privacy** - Your questions never leave your computer
- **No costs** - Zero API fees, no subscriptions, truly unlimited use
- **Always works** - No internet required, no rate limits, no downtime
- **Full transparency** - See exactly which Wikipedia articles were used
- **Research-grade citations** - Every fact is cited with clickable sources

### Core Features
- 🔒 **100% Private** - Everything runs locally, no data leaves your computer
- 🌍 **Complete Wikipedia** - Full English Wikipedia (6+ million articles, updated 2024)
- 🤖 **Two-Stage AI Pipeline** - Specialized models for accurate article selection and synthesis
- 💰 **Free Forever** - No API keys, no subscriptions, no limits
- 📖 **Academic-Style Citations** - Inline citations [1][2][3] with clickable source URLs

### Smart Intelligence
- 🎯 **Content-Based Article Selection** (85-88% accuracy)
  - Retrieves 25+ candidate articles based on Kiwix search (matching query terms)
  - AI reads article abstracts before selecting (not just titles)
  - Intelligently filters to 3-6 best matches based on relevance
  - Automatically excludes lists, stubs, disambiguation pages
  - Direct lookup finds main articles (e.g., "Earthquake" not "List of earthquakes")
  - Proper noun extraction (people, places, organizations, events)
  
- 🧠 **Adaptive Complexity Detection**
  - Analyzes question structure using complexity scoring
  - Scores based on indicators: multi-part questions, comparisons, analytical depth
  - Simple questions: 3 articles with deep reading (~20 paragraphs each)
  - Complex questions: 6 articles with balanced coverage (~10 paragraphs each)
  - Multi-part questions automatically get more sources
  
- 🔬 **Two-Stage AI Pipeline**
  - **Stage 1 - Selection**: Mistral-7B for fast, accurate classification
  - **Stage 2 - Synthesis**: Llama-3.1-8B for coherent answer generation
  - Specialized models for each task improve accuracy and consistency
  - 10-18 second total response time with recommended setup
  
- 📝 **Research-Ready Output**
  - Every fact cited with inline [1], [2], [3] references
  - Clickable URLs to open source articles in browser
  - Multi-source synthesis (combines info from all selected articles)
  - Proper attribution prevents hallucination

- ⚡ **Practical Performance**
  - 10-18 second responses on standard hardware (16-24GB RAM)
  - Works great on CPU (no GPU required)
  - Automatic model detection and fallback
  - Handles 6+ million Wikipedia articles efficiently

## 🎬 Demo

```bash
$ python wikipedia_rag_kiwix.py --question "What are the goals of NASA?"

✓ Connected to Kiwix server at http://localhost:8080
✓ Selection model: mistral:7b
✓ Summarization model: llama3.1:8b

🔍 Searching local Wikipedia for: What are the goals of NASA?
  ✓ Retrieved 11 unique candidates
✓ Found 11 candidate article(s)
  📄 Fetching article abstracts for AI selection...
  🤖 Selecting with mistral:7b (using article abstracts)...
✓ AI selected 3 article(s): Goals, NASA, Timeline of Solar System exploration
  📊 Reading ~20 paragraphs per article (max 8k chars each)
  📄 Fetching: Goals
  📄 Fetching: NASA
  📄 Fetching: Timeline of Solar System exploration
🤖 Generating synthesis with llama3.1:8b...
⏱️  Total time: 13.6s

======================================================================
❓ Question: What are the goals of NASA?
======================================================================

📖 Answer:

NASA's primary goals encompass a wide range of objectives, from advancing 
space exploration to conducting aeronautics research. The agency was 
established in 1958 [2] as an independent federal agency responsible for 
the civil space program, aeronautics research, and space research.

NASA's early goals focused on achieving human spaceflight, which began with 
Project Mercury [2]. The Apollo Program, launched in response to President 
Kennedy's goal of landing an American on the Moon by the end of the 1960s [3], 
marked a significant achievement in space exploration.

NASA's goals also extend beyond human spaceflight to exploring the Solar 
System [3]. The agency has sent numerous robotic spacecraft to explore 
various planets and celestial bodies, greatly expanding our understanding 
of the universe.

----------------------------------------------------------------------
📚 Source Articles (click to open):
   [1] Goals
       http://localhost:8080/wikipedia_en_all_maxi_2024-01/A/Goals
   [2] NASA
       http://localhost:8080/wikipedia_en_all_maxi_2024-01/A/NASA
   [3] Timeline of Solar System exploration
       http://localhost:8080/content/wikipedia_en_all_maxi_2024-01/A/Timeline_of_Solar_System_exploration
======================================================================
```

## 🚀 Quick Start (One-Line Install)

```bash
# Clone and run automated setup
git clone https://github.com/yourusername/offline-wikipedia-rag.git
cd offline-wikipedia-rag
./scripts/install.sh
```

The installer will:
1. ✅ Install Ollama and recommended AI models (Mistral-7B + Llama-3.1-8B)
2. ✅ Download complete Wikipedia (102GB)
3. ✅ Set up all dependencies
4. ✅ Test the system

**Time needed:** 2-8 hours (mostly downloading)  
**Disk space:** ~120GB

## 📋 System Requirements

### Recommended Setup
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, Fedora 35+) or macOS
- **Disk**: 120GB free space (Wikipedia + models)
- **RAM**: 16-24GB for recommended models (Mistral-7B + Llama-3.1-8B)
- **CPU**: Multi-core processor (x86_64 or ARM64)

### Alternative Configurations
- **Budget**: 12GB RAM - Use smaller models (still works well!)
- **High-end**: 32GB+ RAM - Use larger models (Qwen2.5-32B + Gemma2-27B)

See [docs/TWO_STAGE_AI_PIPELINE.md](docs/TWO_STAGE_AI_PIPELINE.md) for detailed model recommendations.

### GPU Support (Optional)
- **GPU acceleration** is automatically detected and used if available
- Works perfectly fine **without GPU** (CPU-only mode)
- Supported: NVIDIA (CUDA), Apple Silicon (Metal), AMD (ROCm)
- No manual GPU setup required - Ollama handles everything automatically

## 💻 Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

### 1. Install Dependencies

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended AI models
ollama pull mistral:7b      # Selection model (~4.4GB)
ollama pull llama3.1:8b     # Summarization model (~4.9GB)

# Install Python environment
conda env create -f environment.yml
conda activate wikipedia-rag
```

### 2. Download Wikipedia

```bash
# Automated download (~102GB, takes 2-8 hours)
./scripts/setup_full_offline_wikipedia.sh
```

### 3. Start Using

```bash
./scripts/start_offline_rag.sh
python wikipedia_rag_kiwix.py
```

</details>

## ⚡ Performance

### Response Speed
- **Total time**: 10-25 seconds for complete answers (search + selection + synthesis)
- **Token generation**: ~50-60 tokens/sec with GPU, ~5-15 tokens/sec CPU-only
- **Network**: Fully offline, zero network latency
- Performance varies based on hardware, models used, and number of articles retrieved

### GPU Acceleration
GPU support is **completely optional**:
- ✅ Ollama **automatically detects** and uses GPU if available (NVIDIA/AMD/Apple Silicon)
- ✅ No CUDA/ROCm installation needed - Ollama includes everything
- ✅ Works great on **CPU-only** systems - no GPU required
- ✅ Seamlessly falls back to CPU if no GPU found

**Bottom line**: Just install and run - GPU acceleration works automatically if you have one, runs fine without it.

## 🎯 Usage

### Interactive Mode

```bash
python wikipedia_rag_kiwix.py
```

Then type your questions naturally:
```
❓ Your question: Explain photosynthesis
❓ Your question: Who was Marie Curie?
❓ Your question: What caused World War 2?
```

### Single Question Mode

```bash
python wikipedia_rag_kiwix.py --question "What is machine learning?"
```

### Command Line Options

```bash
python wikipedia_rag_kiwix.py --help

Options:
  --question TEXT          Ask a single question
  --model TEXT             Summarization model (default: auto-detect)
  --selection-model TEXT   Article selection model (default: auto-detect)
  --kiwix-url TEXT         Kiwix server URL (default: http://localhost:8080)
  --max-results INT        Number of articles (default: auto by complexity)
```

### Recommended Models

```bash
# Recommended setup (most users)
python wikipedia_rag_kiwix.py \
  --selection-model mistral:7b \
  --model llama3.1:8b

# Alternative for better selection (if you have 32GB+ RAM)
python wikipedia_rag_kiwix.py \
  --selection-model qwen2.5:32b-instruct \
  --model llama3.1:8b
```

## 🏗️ How It Works

```mermaid
graph LR
    A[Your Question] --> B[Search Wikipedia]
    B --> C[Fetch Abstracts]
    C --> D[Stage 1: AI Selection]
    D --> E[Fetch Full Articles]
    E --> F[Stage 2: AI Synthesis]
    F --> G[Answer with Citations]
```

### Two-Stage AI Pipeline

1. **Wikipedia Search**
   - Extracts search terms from your question
   - Searches for 25+ candidate articles
   - Direct lookup finds main articles

2. **Stage 1: Content-Based Selection (Mistral-7B)**
   - Fetches first paragraph (abstract) from each candidate
   - AI evaluates actual content, not just titles
   - Selects 3-6 most relevant articles
   - Filters out lists, stubs, and irrelevant topics

3. **Stage 2: Synthesis with Citations (Llama-3.1-8B)**
   - Reads full content of selected articles
   - Synthesizes comprehensive answer
   - Adds inline citations [1][2][3] for every fact
   - Provides clickable URLs to source articles

### Why Two Models?

Specialized models perform better than one model doing everything:
- **Selection model** (default: Mistral-7B): Fast, accurate classification from article abstracts
- **Summarization model** (default: Llama-3.1-8B): Excellent world knowledge and synthesis
- **Result**: 85-88% selection accuracy + high-quality answers in 10-18 seconds
- System auto-detects available models and selects best options

## 🛠️ Technology Stack

- **AI Models**: 
  - [Mistral-7B](https://mistral.ai/) - Fast, accurate article selection from abstracts
  - [Llama-3.1-8B](https://ai.meta.com/llama/) - High-quality answer synthesis with citations
  - Alternative selection models: Qwen2.5 (32B/14B/7B), Hermes-3-8B
  - Alternative synthesis models: Gemma-2 (27B/9B), Llama-3.3-70B
- **Wikipedia**: [Kiwix](https://www.kiwix.org/) - Offline Wikipedia server (ZIM format)
- **Runtime**: [Ollama](https://ollama.ai/) - Local AI model runner
- **Language**: Python 3.10+

## 📁 Project Structure

```
offline-wikipedia-rag/
├── wikipedia_rag_kiwix.py              # Main RAG application
├── environment.yml                     # Python environment
├── requirements.txt                    # Python dependencies
├── scripts/
│   ├── install.sh                      # One-line installer
│   ├── setup_full_offline_wikipedia.sh # Wikipedia downloader
│   └── start_offline_rag.sh            # Quick start script
├── tests/
│   ├── test_system.py                  # System validation tests
│   ├── test_token_speed.py             # Token generation benchmarks
│   └── test_cpu_speed.py               # CPU performance tests
└── docs/                               # Additional documentation
```

## 🧪 Testing

The project includes a comprehensive pytest test suite:

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rag_functions.py -v

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html

# Run only unit tests (skip integration tests)
pytest -m "not integration"
```

**Test Coverage:**
- ✅ Search term extraction (proper nouns, content words, stopword filtering)
- ✅ Complexity estimation (simple, multi-part, comparison questions)
- ✅ Model detection (selection priority, reasoning model avoidance)
- ✅ Integration tests for Kiwix/Ollama (marked separately)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for other languages (German, French, Spanish Wikipedia)
- GUI interface
- Docker container
- Additional AI models
- Performance optimizations
- More test coverage

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## ⚠️ Known Limitations

- **First download takes time**: Wikipedia is 102GB
- **RAM usage**: Two models need ~10-12GB RAM (or 16GB recommended)
- **CPU-based**: Works great on CPU, GPU support optional
- **English only**: Currently supports English Wikipedia only

## 🆘 Troubleshooting

**Problem: "Connection refused to Kiwix"**
```bash
# Restart Kiwix server
./scripts/start_offline_rag.sh
```

**Problem: "Ollama model not found"**
```bash
# Pull the recommended models
ollama pull mistral:7b
ollama pull llama3.1:8b
```

**Problem: "Out of memory"**
```bash
# System will automatically fall back to smaller available models
# Or manually specify a smaller model
python wikipedia_rag_kiwix.py --model llama3.2:3b
```

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) - For the excellent Mistral-7B model (default selection model)
- [Meta AI](https://ai.meta.com/) - For the powerful Llama-3.1-8B model (default synthesis model)
- [Kiwix](https://www.kiwix.org/) - For offline Wikipedia technology
- [Ollama](https://ollama.ai/) - For easy local AI model deployment
- [Wikimedia Foundation](https://www.wikimedia.org/) - For Wikipedia

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover the project.

---

**Made with ❤️ for the open-source community**
