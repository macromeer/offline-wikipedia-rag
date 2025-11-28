# Project Structure

```
ollama-wikipedia-rag/
├── README.md                              # Main documentation
├── READY.md                               # Quick usage guide
├── OFFLINE_SETUP.md                       # Detailed setup instructions
│
├── environment.yml                        # Mamba/Conda environment
├── requirements.txt                       # Python dependencies
│
├── wikipedia_rag_kiwix.py                 # Main RAG application
├── test_system.py                         # System tests
│
├── setup_full_offline_wikipedia.sh        # Download Wikipedia + install Kiwix
├── start_offline_rag.sh                   # Quick start script
│
└── archive/                               # Old implementations
    ├── wikipedia_rag.py                   # Online Wikipedia API version
    ├── wikipedia_rag_offline.py           # Vector database version
    ├── wikipedia_indexer.py               # Indexer for vector DB
    ├── setup.sh                           # Old setup script
    ├── setup_mamba.sh                     # Old mamba setup
    ├── QUICKSTART.md                      # Old quick start
    └── sample_titles.txt                  # Sample article list
```

## External Resources

```
~/wikipedia-offline/
└── wikipedia_en_all_maxi_2024-01.zim     # Full Wikipedia (102GB)

~/.local/bin/
└── kiwix-serve                            # Kiwix server binary
```

## Key Files Explained

### Active Scripts

**wikipedia_rag_kiwix.py**
- Main RAG application
- Connects to local Kiwix server
- Auto-detects DeepSeek model
- Searches Wikipedia and generates answers

**setup_full_offline_wikipedia.sh**
- Downloads full English Wikipedia ZIM file
- Installs Kiwix tools
- One-time setup (~2-8 hours)

**start_offline_rag.sh**
- Checks if Kiwix is running
- Starts Kiwix server if needed
- Quick startup script

**test_system.py**
- Validates Ollama connection
- Tests Wikipedia API (for comparison)
- Verifies RAG functionality

### Configuration

**environment.yml**
- Mamba/Conda environment specification
- Python 3.10
- All required packages

**requirements.txt**
- Alternative pip installation
- Same dependencies as environment.yml

### Documentation

**READY.md**
- Quick reference for daily use
- Common commands
- Troubleshooting

**OFFLINE_SETUP.md**
- Detailed setup instructions
- System requirements
- Advanced configuration

**README.md**
- Project overview
- Quick start guide
- Architecture explanation
