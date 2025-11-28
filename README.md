# 🌐 Free Offline AI with Full Wikipedia Knowledge

**Chat with AI that has complete Wikipedia knowledge - 100% free, 100% offline, 100% private.**

This project combines DeepSeek R1 (a powerful open-source AI model) with the complete English Wikipedia to give you an intelligent assistant that works entirely offline, requires no API keys, and respects your privacy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- 🔒 **100% Private** - Everything runs locally, no data leaves your computer
- 🌍 **Complete Wikipedia** - Full English Wikipedia (6+ million articles)
- 🤖 **DeepSeek R1** - State-of-the-art open-source AI model
- 💰 **Free Forever** - No API keys, no subscriptions, no limits
- ⚡ **Fast** - Local processing, no network latency
- 📖 **Clean Answers** - Well-formatted, readable responses with sources

## 🎬 Demo

```
❓ Your question: What is quantum computing?

📖 Answer:
Quantum computing is a type of computation that harnesses quantum mechanical 
phenomena such as superposition and entanglement to perform operations on data.
Unlike classical computers that use bits (0 or 1), quantum computers use quantum 
bits or "qubits" which can exist in multiple states simultaneously...

📚 Sources: Quantum computing, Qubit, Quantum entanglement
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
    A[Your Question] --> B[Search Wikipedia]
    B --> C[Kiwix Server]
    C --> D[Retrieve Articles]
    D --> E[DeepSeek AI]
    E --> F[Formatted Answer]
```

1. **Question** → You ask in natural language
2. **Search** → System searches 6M+ Wikipedia articles locally
3. **Retrieve** → Gets relevant content from Kiwix server
4. **Generate** → DeepSeek creates answer using Wikipedia context
5. **Display** → Clean, readable response with sources

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
