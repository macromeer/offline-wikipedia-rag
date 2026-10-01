# Troubleshooting Guide

Common issues and solutions for Offline Wikipedia RAG.

## Installation Issues

### Problem: "curl: command not found"
**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install curl

# Fedora/RHEL
sudo dnf install curl
```

### Problem: "Insufficient disk space"
**Solution:**
- Free up ~55GB for the `nopic` edition, or use `--variant mini` (~13GB)
- Use external drive: `ln -s /path/to/external/drive ~/wikipedia-offline`

## Runtime Issues

### Problem: "Connection refused to Kiwix"
**Symptoms:** `Connection refused at http://localhost:8080`

**Solutions:**

1. Check if Kiwix is running:
```bash
ps aux | grep kiwix-serve
```

2. Start Kiwix manually:
```bash
~/.local/bin/kiwix-serve --port=8080 ~/wikipedia-offline/*.zim &
```

3. Check port availability:
```bash
# See if port 8080 is in use
lsof -i :8080

# Use different port
kiwix-serve --port=8090 ~/wikipedia-offline/*.zim &
./run.sh --kiwix-url http://localhost:8090
```

### Problem: "Ollama model not found"
**Symptoms:** `model 'deepseek-r1:latest' not found`

**Solutions:**

1. Check installed models:
```bash
ollama list
```

2. Pull the model:
```bash
ollama pull deepseek-r1:latest
```

3. Use a different model:
```bash
# See available models at ollama.ai/library
ollama pull llama2
./run.sh --model llama2
```

### Problem: "Out of memory"
**Symptoms:** System freezes, `OOM` errors, very slow responses

**Solutions:**

1. Use smaller model:
```bash
# 7B model (needs ~4GB RAM)
ollama pull deepseek-r1:7b
./run.sh --model deepseek-r1:7b
```

2. Increase swap space:
```bash
# Add 8GB swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

3. Close other applications

### Problem: "Wikipedia search returns no results"
**Symptoms:** "No relevant Wikipedia articles found"

**Solutions:**

1. Try more specific query:
```python
# Instead of "Python"
./run.sh --question "Python programming language"
```

2. Check Kiwix in browser:
```bash
# Open http://localhost:8080 and try searching manually
```

3. Verify ZIM file:
```bash
ls -lh ~/wikipedia-offline/
# Should show ~102GB .zim file
```

### Problem: "Slow response times"
**Symptoms:** Takes 30+ seconds to answer

**Solutions:**

1. Check CPU usage:
```bash
htop  # or top
# DeepSeek should be using CPU
```

2. Use GPU (if available):
```bash
# Install CUDA support for Ollama
# See: https://ollama.ai/download
```

3. Reduce articles fetched:
```bash
./run.sh --max-results 1 --question "Your question"
```

## Environment Issues

### Problem: "uv: command not found"
**Solution:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Problem: "Module not found: ollama"
**Solution:**
```bash
uv sync          # or just use ./run.sh, which syncs automatically
```

## Wikipedia/Kiwix Issues

### Problem: "404 Not Found" when downloading Wikipedia
**Solution:**
- Old ZIM files are removed from the mirror when new dumps are published
- `./scripts/setup_full_offline_wikipedia.sh` looks up the newest file automatically; re-run it
- `--dry-run` prints the URL it would download

### Problem: "Kiwix serves wrong content"
**Solution:**
```bash
# Clear Kiwix cache
rm -rf ~/.cache/kiwix/

# Restart Kiwix
pkill kiwix-serve
kiwix-serve --port=8080 ~/wikipedia-offline/*.zim &
```

## Performance Optimization

### Make responses faster:

1. **Use SSD** instead of HDD for Wikipedia file
2. **Add more RAM** (16GB+ recommended)
3. **Use GPU** with CUDA-enabled Ollama
4. **Reduce article count**: `--max-results 2`
5. **Use smaller model**: e.g. `--model llama3.1:8b`

### Reduce disk space:

```bash
# nopic (default, ~49GB) instead of maxi (~119GB), or mini (~13GB, intros only)
./scripts/setup_full_offline_wikipedia.sh --variant mini
```

## Getting Help

If none of these solutions work:

1. **Check logs**:
```bash
# Ollama logs
journalctl -u ollama -f

# Kiwix output
ps aux | grep kiwix-serve
```

2. **Open an issue** on GitHub with:
   - Your OS and version
   - Error messages
   - Steps to reproduce
   - Output of `ollama list` and `python --version`

3. **Community support**:
   - GitHub Discussions
   - Ollama Discord
   - Kiwix forum
