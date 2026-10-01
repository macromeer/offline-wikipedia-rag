# Automatic Setup and Startup

This document explains what happens automatically when you run the Wikipedia RAG script.

## What the Script Does Automatically

When you run `./run.sh` or `uv run python wikipedia_rag_kiwix.py`, the following happens automatically:

### 1. Environment (run.sh only)
- Checks that [uv](https://docs.astral.sh/uv/) is installed
- Runs the script with `uv run`, which creates/syncs `.venv` from `pyproject.toml` on first use

### 2. Dependency Check
The Python script checks:
- ✅ **Ollama**: Verifies Ollama is running by attempting to list models
  - If not running: Shows error with instructions to run `ollama serve`
  
### 3. Kiwix Server Auto-Start
If Kiwix server is not already running, the script will:
- ✅ Search for `kiwix-serve` binary in common locations:
  - `~/.local/bin/kiwix-serve`
  - `/usr/local/bin/kiwix-serve`
  - `/usr/bin/kiwix-serve`
  - System PATH
  
- ✅ Pick the Wikipedia ZIM file: `--zim PATH`, else `$WIKI_ZIM`, else the newest
  complete `*.zim` in these locations (partial downloads are skipped):
  - `~/wikipedia-offline/`
  - `~/Downloads/`
  - `/data/wikipedia/`
  - `/var/lib/kiwix/`
  
- ✅ Start the server on that ZIM (listening on 127.0.0.1 only) if both are found
- ✅ Wait for server to be ready before proceeding
- ✅ Clean up (stop server) when script exits

### 4. Model Auto-Detection
The script automatically:
- Lists all available Ollama models
- Selects the best model for **article selection** (Stage 1):
  - Prefers: Qwen 3.6/3.5, Gemma 4, Qwen 3, then Qwen2.5, Mistral, Hermes-3
  
- Selects the best model for **answer synthesis** (Stage 2):
  - Prefers: Gemma 4, Qwen 3.6/3.5, Qwen 3, then Llama-3.1-8B, Gemma2, Mistral-7B
  
- Never auto-selects coder, embedding, reranker or reasoning (`r1`, DeepSeek) models
- Skips models under 4B parameters unless nothing else is installed (with a warning)

## Manual Control

### Disable Auto-Start
If you want to manage Kiwix yourself:

```bash
./run.sh --no-auto-start
```

### Specify Models
Override automatic detection:

```bash
./run.sh --selection-model mistral:7b --model llama3.1:8b
```

### Custom Kiwix URL
Connect to a different Kiwix server:

```bash
./run.sh --kiwix-url http://192.168.1.100:8080
```

## What Users No Longer Need to Do

❌ **Old way** (manual):
```bash
# Manually activate environment
source .venv/bin/activate

# Manually start Kiwix server
kiwix-serve --port=8080 ~/wikipedia-offline/*.zim &

# Wait a few seconds
sleep 3

# Make sure Ollama is running
ollama serve &

# Finally run the script
python wikipedia_rag_kiwix.py
```

✅ **New way** (automatic):
```bash
./run.sh
```

## Behind the Scenes

### Process Management
- The script keeps track of any Kiwix process it starts
- On exit (normal or Ctrl+C), it automatically stops the server
- Uses `atexit` handler for cleanup
- Runs Kiwix in a separate process group for clean termination

### Error Handling
- Clear error messages if dependencies are missing
- Helpful instructions for fixing issues
- Graceful fallbacks for missing models or configuration
- No silent failures

### Platform Compatibility
- Works on Linux, macOS, and WSL
- Handles different shell configurations
- Respects existing Kiwix installations

## Troubleshooting

### "Ollama is not running"
```bash
# Start Ollama in a separate terminal
ollama serve
```

### "kiwix-serve not found"
The setup script installs it:
```bash
./scripts/setup_full_offline_wikipedia.sh
```

### "No Wikipedia ZIM files found"
Run the setup script:
```bash
./scripts/setup_full_offline_wikipedia.sh
```

Or manually download and place in `~/wikipedia-offline/`, or pass `--zim PATH`.

### "uv not found"
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Advanced: Integration with Other Tools

### Running as a Service
You can set up the script to run as a systemd service or background daemon.

### Docker/Container Support
The auto-start features work well in containerized environments.

### CI/CD Integration
The `--no-auto-start` flag is useful for automated testing where you manage services separately.
