# Quick Reference Guide

## ✅ What You Need to Know

### 1. Wikipedia Content - Two Approaches

**Online Mode (DEFAULT - Use This!):**
- The script `wikipedia_rag.py` fetches Wikipedia articles **on-demand** using the Wikipedia API
- **No pre-downloading needed** - just run it!
- Requires internet connection each time you ask a question

**Offline Mode (OPTIONAL - Advanced):**
- First, download articles via Wikipedia API and build an index: `python wikipedia_indexer.py --titles "Topic1" "Topic2"`
- Then use: `python wikipedia_rag_offline.py`
- Faster subsequent queries, works offline after initial setup
- Still needs internet for the initial indexing step

### 2. Model Selection

**Auto-detection (easiest):**
```bash
python wikipedia_rag.py
# Uses your first available model (currently: qwen3:32b)
```

**Specify model:**
```bash
python wikipedia_rag.py --model deepseek-r1:latest
python wikipedia_rag.py --model qwen3:32b
```

## 🚀 Common Commands

```bash
# Activate environment first
mamba activate wikipedia-rag

# Interactive mode (auto-detects model)
python wikipedia_rag.py

# Interactive with specific model
python wikipedia_rag.py --model deepseek-r1:latest

# Single question
python wikipedia_rag.py --question "What is quantum computing?"

# Single question with specific model
python wikipedia_rag.py --model deepseek-r1:latest --question "Explain AI"

# Build offline index (optional)
python wikipedia_indexer.py --file sample_titles.txt

# Use offline mode (after building index)
python wikipedia_rag_offline.py --model deepseek-r1:latest
```

## 📝 Your Available Models

- `qwen3:32b` - Your larger model
- `deepseek-r1:latest` - DeepSeek reasoning model

## 💡 Tips

1. **Start with online mode** - no setup required
2. **Use DeepSeek** if you prefer: `--model deepseek-r1:latest`
3. **Build offline index** only if you want faster/offline access later
4. **The indexer downloads articles** from Wikipedia API (needs internet once)
