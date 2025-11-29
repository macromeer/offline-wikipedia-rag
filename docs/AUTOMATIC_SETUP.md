# Automatic Setup and Startup

This document explains what happens automatically when you run the Wikipedia RAG script.

## What the Script Does Automatically

When you run `./run.sh` or `python wikipedia_rag_kiwix.py`, the following happens automatically:

### 1. Environment Check (run.sh only)
- Checks if mamba/conda is installed
- Activates the `wikipedia-rag` environment
- Falls back gracefully if environment not found

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
  
- ✅ Search for Wikipedia ZIM files in common locations:
  - `~/wikipedia-offline/`
  - `~/Downloads/`
  - `/data/wikipedia/`
  - `/var/lib/kiwix/`
  
- ✅ Start the server automatically if both are found
- ✅ Wait for server to be ready before proceeding
- ✅ Clean up (stop server) when script exits

### 4. Model Auto-Detection
The script automatically:
- Lists all available Ollama models
- Selects the best model for **article selection** (Stage 1):
  - Prefers: Mistral-7B, Qwen2.5, Hermes-3
  - Optimized for fast, accurate classification
  
- Selects the best model for **answer synthesis** (Stage 2):
  - Prefers: Llama-3.1-8B, Gemma2, Mistral-7B
  - Optimized for coherent text generation
  
- Falls back gracefully if preferred models aren't available
- Warns if using suboptimal models

## Manual Control

### Disable Auto-Start
If you want to manage Kiwix yourself:

```bash
python wikipedia_rag_kiwix.py --no-auto-start
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
mamba activate wikipedia-rag

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
- Automatically detects mamba vs conda
- Handles different shell configurations
- Respects existing Kiwix installations

## Troubleshooting

### "Ollama is not running"
```bash
# Start Ollama in a separate terminal
ollama serve
```

### "kiwix-serve not found"
Download and install:
```bash
wget https://download.kiwix.org/release/kiwix-tools/kiwix-tools_linux-x86_64.tar.gz
tar xzf kiwix-tools_linux-x86_64.tar.gz
mv kiwix-tools_*/kiwix-serve ~/.local/bin/
```

### "No Wikipedia ZIM files found"
Run the setup script:
```bash
./scripts/setup_full_offline_wikipedia.sh
```

Or manually download and place in `~/wikipedia-offline/`

### Environment not found
Create the environment:
```bash
mamba env create -f environment.yml
```

## Advanced: Integration with Other Tools

### Running as a Service
You can set up the script to run as a systemd service or background daemon.

### Docker/Container Support
The auto-start features work well in containerized environments.

### CI/CD Integration
The `--no-auto-start` flag is useful for automated testing where you manage services separately.
