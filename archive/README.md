# Archived Files

This folder contains old implementations that are no longer actively used.

## Contents

- **wikipedia_rag.py** - Original online version using Wikipedia API
- **wikipedia_rag_offline.py** - Vector database version with ChromaDB
- **wikipedia_indexer.py** - Tool to build local vector database
- **setup.sh** - Original setup script
- **setup_mamba.sh** - Original Mamba setup
- **QUICKSTART.md** - Old quick start guide
- **sample_titles.txt** - Sample Wikipedia titles for indexing

## Current Implementation

The current active implementation uses:
- **wikipedia_rag_kiwix.py** - Full offline Wikipedia via Kiwix
- Complete English Wikipedia (102GB ZIM file)
- Kiwix server for local access
- No vector database needed

## Why Archived?

These files were part of earlier approaches:
1. Online API approach (requires internet)
2. Vector database approach (complex setup, partial Wikipedia)

The Kiwix approach is simpler and provides complete offline Wikipedia access.
