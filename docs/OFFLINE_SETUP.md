# Full Offline Wikipedia with Kiwix

This guide sets up a **complete offline English Wikipedia** using Kiwix.

## What You Get

- ✅ **Complete English Wikipedia** (~95GB compressed, ~100GB+ uncompressed)
- ✅ **Fully offline** - no internet needed after download
- ✅ **Fast local access** via Kiwix server
- ✅ **RAG integration** with your Ollama model (DeepSeek)

## Quick Setup

### 1. Download & Install

Run the automated setup:
```bash
chmod +x setup_full_offline_wikipedia.sh
./setup_full_offline_wikipedia.sh
```

This will:
- Install Kiwix tools
- Download English Wikipedia ZIM file (~95GB, takes hours)
- Place it in `~/wikipedia-offline/`

**Alternative - Manual Download:**

If you prefer to download manually:
```bash
mkdir -p ~/wikipedia-offline
cd ~/wikipedia-offline

# Download latest English Wikipedia (check https://wiki.kiwix.org/wiki/Content for latest)
wget https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_maxi_2024-01.zim
```

### 2. Start Kiwix Server

```bash
# Start the server (runs in foreground)
kiwix-serve ~/wikipedia-offline/*.zim

# Or run in background
kiwix-serve ~/wikipedia-offline/*.zim &
```

Server runs at: http://localhost:8080

### 3. Test in Browser

Open http://localhost:8080 in your browser to verify Wikipedia loads.

### 4. Use with RAG

**Activate environment:**
```bash
mamba activate wikipedia-rag
```

**Install additional dependencies:**
```bash
pip install beautifulsoup4 requests
```

**Run RAG system:**
```bash
# Interactive mode (auto-uses DeepSeek, excludes qwen)
python wikipedia_rag_kiwix.py

# Single question
python wikipedia_rag_kiwix.py --question "What is quantum computing?"

# Specify model explicitly
python wikipedia_rag_kiwix.py --model deepseek-r1:latest --question "Explain AI"
```

## System Requirements

- **Disk Space**: ~110GB (95GB download + extraction)
- **RAM**: 4GB+ recommended
- **Download Time**: 2-8 hours (depends on connection)
- **Kiwix Server**: ~100MB RAM when running

## Advantages

✅ **100% Offline** - No internet needed after setup
✅ **Complete Encyclopedia** - All English Wikipedia articles
✅ **Fast Retrieval** - Local server, instant access
✅ **No API Limits** - Unlimited queries
✅ **Privacy** - All data stays local

## File Structure

```
~/wikipedia-offline/
  └── wikipedia_en_all_maxi_2024-01.zim  (~95GB)

~/Documents/ollama-wikipedia-rag/
  ├── wikipedia_rag_kiwix.py              (RAG with Kiwix)
  ├── setup_full_offline_wikipedia.sh     (Setup script)
  └── OFFLINE_SETUP.md                    (This file)
```

## Troubleshooting

**Kiwix server not found:**
```bash
# Make sure ~/.local/bin is in PATH
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

**Connection refused:**
```bash
# Make sure Kiwix is running
ps aux | grep kiwix-serve

# Start it if not running
kiwix-serve ~/wikipedia-offline/*.zim &
```

**Model issues:**
```bash
# List available models
ollama list

# The script automatically excludes qwen models
# Use DeepSeek explicitly if needed
python wikipedia_rag_kiwix.py --model deepseek-r1:latest
```

## Keeping Kiwix Running

**Start automatically on boot (systemd):**
```bash
# Create service file
cat > ~/.config/systemd/user/kiwix.service <<EOF
[Unit]
Description=Kiwix Wikipedia Server
After=network.target

[Service]
Type=simple
ExecStart=/home/$USER/.local/bin/kiwix-serve /home/$USER/wikipedia-offline/wikipedia_en_all_maxi_2024-01.zim
Restart=on-failure

[Install]
WantedBy=default.target
EOF

# Enable and start
systemctl --user enable kiwix
systemctl --user start kiwix
```

## Notes

- Wikipedia ZIM files are updated monthly at https://download.kiwix.org/zim/wikipedia/
- The `wikipedia_en_all_maxi` version includes all articles but no images
- For images, use `wikipedia_en_all_nopic` (larger, ~90GB+)
