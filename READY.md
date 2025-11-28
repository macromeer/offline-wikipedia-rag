# ✅ Offline Wikipedia RAG - Ready to Use!

## What You Have

- ✅ **Full English Wikipedia** (102GB) downloaded to `~/wikipedia-offline/`
- ✅ **Kiwix server** installed and running on port 8080
- ✅ **RAG system** connected to DeepSeek model
- ✅ **100% Offline** - no internet needed after setup

## Quick Start

### Start the System

```bash
cd ~/Documents/ollama-wikipedia-rag
./start_offline_rag.sh
```

### Activate Environment & Use

```bash
# Activate environment
mamba activate wikipedia-rag

# Interactive mode
python wikipedia_rag_kiwix.py

# Single question
python wikipedia_rag_kiwix.py --question "What is quantum computing?"

# Specific model (DeepSeek is auto-detected, qwen excluded)
python wikipedia_rag_kiwix.py --model deepseek-r1:latest --question "Explain AI"
```

## System Status

**Kiwix Server:**
- Status: ✅ Running
- URL: http://localhost:8080
- Wikipedia File: ~/wikipedia-offline/wikipedia_en_all_maxi_2024-01.zim
- Size: 102GB
- Check if running: `ps aux | grep kiwix-serve`

**Ollama Models:**
- DeepSeek R1: ✅ Auto-detected (qwen excluded as requested)

## Example Session

```bash
$ cd ~/Documents/ollama-wikipedia-rag
$ mamba activate wikipedia-rag
$ python wikipedia_rag_kiwix.py

❓ Your question: What is machine learning?

# System searches local Wikipedia, retrieves articles,
# and DeepSeek generates an answer based on that content
```

## Advantages

- ✅ **100% Offline** - All data local, no API calls
- ✅ **Fast** - Local retrieval, instant Wikipedia access
- ✅ **Complete** - Full English Wikipedia (all articles)
- ✅ **Private** - No external queries, data stays local
- ✅ **Unlimited** - No rate limits or API costs
- ✅ **DeepSeek** - Uses your preferred model automatically

## Files

```
~/Documents/ollama-wikipedia-rag/
├── wikipedia_rag_kiwix.py           # Main RAG script
├── start_offline_rag.sh             # Startup script
├── OFFLINE_SETUP.md                 # Setup guide
└── READY.md                         # This file

~/wikipedia-offline/
└── wikipedia_en_all_maxi_2024-01.zim  # Full Wikipedia

~/.local/bin/
└── kiwix-serve                      # Kiwix server binary
```

## Troubleshooting

**Restart Kiwix server:**
```bash
pkill kiwix-serve
~/.local/bin/kiwix-serve --port=8080 ~/wikipedia-offline/*.zim &
```

**Check Kiwix in browser:**
Open http://localhost:8080

**Model issues:**
```bash
# List models
ollama list

# The script auto-excludes qwen models as requested
```

## Next Steps

Just run:
```bash
cd ~/Documents/ollama-wikipedia-rag
mamba activate wikipedia-rag
python wikipedia_rag_kiwix.py
```

Enjoy your fully offline Wikipedia-powered AI assistant! 🎉
