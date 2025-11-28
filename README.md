# 🌐 Free Offline AI with Full Wikipedia Knowledge

**Chat with AI that has complete Wikipedia knowledge - 100% free, 100% offline, 100% private.**

This project combines DeepSeek R1 (a powerful open-source AI model) with the complete English Wikipedia to give you an intelligent assistant that works entirely offline, requires no API keys, and respects your privacy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

### Core Features
- 🔒 **100% Private** - Everything runs locally, no data leaves your computer
- 🌍 **Complete Wikipedia** - Full English Wikipedia (6+ million articles)
- 🤖 **DeepSeek R1** - State-of-the-art open-source AI model
- 💰 **Free Forever** - No API keys, no subscriptions, no limits
- 📖 **Clean Answers** - Well-formatted, readable responses with proper source citations

### Smart Intelligence
- 🎯 **AI-Powered Article Selection** - Retrieves 3x more articles, then uses AI to select most relevant ones
  - Filters out irrelevant results (TV shows, sports, unrelated topics)
  - Intelligently evaluates article titles against your question
  - Ensures high-quality, topic-focused answers
- 🧠 **Adaptive Complexity** - Automatically retrieves 3-7 articles based on question complexity
  - Simple questions: 3 articles with deep reading
  - Complex multi-part questions: 7 articles with balanced coverage
- ⚖️ **Balanced Reading** - Dynamically adjusts content depth for optimal speed
  - 3 articles: 30 paragraphs each (comprehensive)
  - 7 articles: 12 paragraphs each (broad coverage)
  - Keeps context manageable regardless of article count
- ⚡ **Fast Processing** - Smart content balancing means quick responses even with many articles

## 🎬 Demo

```
❓ Your question: How old is the universe and what is its future?

🔍 Searching local Wikipedia...
  🔎 Search terms: age universe future
✓ Found 18 candidate article(s)
  🤖 AI selecting 6 most relevant from 18 articles...
  ✓ AI selected: Age of the universe, Physical cosmology, Big Bang, 
     Future of an expanding universe, Heat death of the universe, Dark energy
✓ Selected 6 relevant article(s)
  📊 Reading ~15 paragraphs per article
  📄 Fetching articles...

📖 Answer:
The universe is approximately 13.8 billion years old, as determined by 
measurements of the cosmic microwave background radiation and observations 
of distant supernovae. This age is calculated from the Big Bang...

The future of the universe is predicted to be one of continued expansion. 
Due to dark energy, the expansion is accelerating. Eventually, in trillions 
of years, the universe will reach a state called "heat death"...

Sources: [1], [2], [3], [4], [5]

----------------------------------------------------------------------
📚 Retrieved Articles:
   [1] Age of the universe
   [2] Physical cosmology
   [3] Big Bang
   [4] Future of an expanding universe
   [5] Heat death of the universe
   [6] Dark energy
```

## 🚀 Quick Start (One-Line Install)

```bash
# Clone and run automated setup
git clone https://github.com/yourusername/offline-wikipedia-rag.git
cd offline-wikipedia-rag
./install.sh
```

The installer will:
1. ✅ Install Ollama and DeepSeek R1 model
2. ✅ Download complete Wikipedia (102GB)
3. ✅ Set up all dependencies
4. ✅ Test the system

**Time needed:** 2-8 hours (mostly downloading)  
**Disk space:** ~120GB

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, Fedora 35+) or macOS
- **Disk**: 120GB free space
- **RAM**: 8GB minimum, 16GB recommended for better performance
- **CPU**: Multi-core processor (x86_64 or ARM64)

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

# Pull DeepSeek R1 model (~5GB)
ollama pull deepseek-r1:latest

# Install Python environment
conda env create -f environment.yml
conda activate wikipedia-rag
```

### 2. Download Wikipedia

```bash
# Automated download (~102GB, takes 2-8 hours)
./setup_full_offline_wikipedia.sh
```

### 3. Start Using

```bash
./start_offline_rag.sh
python wikipedia_rag_kiwix.py
```

</details>

## ⚡ Performance

### Response Speed
- **With GPU**: Very fast (~20-50 tokens/sec) - instant responses
- **CPU-only**: Still usable (~2-10 tokens/sec) - 5-15 second responses
- **Network**: Fully offline, zero network latency

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
  --question TEXT      Ask a single question
  --model TEXT         Specify AI model (default: auto-detect)
  --kiwix-url TEXT     Kiwix server URL (default: http://localhost:8080)
  --max-results INT    Number of Wikipedia articles to use (default: 3)
```

## 🏗️ How It Works

```mermaid
graph LR
    A[Your Question] --> B[Analyze Complexity]
    B --> C[Search Wikipedia 3x]
    C --> D[AI Filters Relevant]
    D --> E[Balanced Reading]
    E --> F[DeepSeek Synthesis]
    F --> G[Answer + Citations]
```

### Intelligent Pipeline

1. **Complexity Analysis** 
   - Analyzes your question structure
   - Determines optimal article count (3-7)
   - Detects comparisons, multi-part questions, historical queries

2. **Smart Search** (3x Overfetch)
   - Searches Wikipedia for 3x more articles than needed
   - Example: Need 6 articles? Retrieve 18 candidates
   - Casts wide net to ensure relevant articles are included

3. **AI-Powered Selection** ⭐ NEW
   - DeepSeek evaluates all candidate article titles
   - Intelligently selects most relevant ones
   - Filters out irrelevant results (TV shows, sports, etc.)
   - Ensures high-quality, focused answers

4. **Balanced Content Reading** ⭐ NEW
   - Dynamically adjusts paragraphs per article
   - 3 articles: 30 paragraphs each (deep dive)
   - 7 articles: 12 paragraphs each (broad coverage)
   - Keeps total context manageable for fast processing

5. **Synthesis & Generation**
   - DeepSeek reads ALL selected articles
   - Synthesizes information across sources
   - Identifies connections and relationships
   - Generates comprehensive, coherent answer

6. **Clean Output**
   - Well-formatted paragraphs
   - Proper source citations [1], [2], [3]
   - List of retrieved articles with numbers

## 🛠️ Technology Stack

- **AI Model**: [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1) - Open-source reasoning model
- **Wikipedia**: [Kiwix](https://www.kiwix.org/) - Offline Wikipedia server (ZIM format)
- **Runtime**: [Ollama](https://ollama.ai/) - Local AI model runner
- **Language**: Python 3.10+

## 📁 Project Structure

```
offline-wikipedia-rag/
├── wikipedia_rag_kiwix.py              # Main application
├── install.sh                          # One-line installer
├── start_offline_rag.sh                # Quick start
├── environment.yml                     # Python environment
└── docs/                               # Additional documentation
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for other languages (German, French, Spanish Wikipedia)
- GUI interface
- Docker container
- Additional AI models
- Performance optimizations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## ⚠️ Known Limitations

- **First download takes time**: Wikipedia is 102GB
- **RAM usage**: DeepSeek R1 needs ~8GB RAM
- **CPU-based**: GPU support not yet implemented (coming soon)
- **English only**: Currently supports English Wikipedia only

## 🆘 Troubleshooting

**Problem: "Connection refused to Kiwix"**
```bash
# Restart Kiwix server
./start_offline_rag.sh
```

**Problem: "Ollama model not found"**
```bash
# Pull the model
ollama pull deepseek-r1:latest
```

**Problem: "Out of memory"**
```bash
# Use a smaller model
ollama pull deepseek-r1:8b
python wikipedia_rag_kiwix.py --model deepseek-r1:8b
```

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DeepSeek AI](https://www.deepseek.com/) - For the amazing open-source model
- [Kiwix](https://www.kiwix.org/) - For offline Wikipedia technology
- [Ollama](https://ollama.ai/) - For easy local AI model deployment
- [Wikimedia Foundation](https://www.wikimedia.org/) - For Wikipedia

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover the project.

---

**Made with ❤️ for the open-source community**
