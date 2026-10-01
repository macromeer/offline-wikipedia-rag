#!/bin/bash

# Simple usage script - shows common commands

cat << 'USAGE'
========================================
Offline Wikipedia RAG - Usage Guide
========================================

## First Time Setup (do once)
scripts/setup_full_offline_wikipedia.sh     # kiwix-serve + newest Wikipedia ZIM
uv sync                                      # Python environment (.venv)

## Daily Use (starts Kiwix automatically)

   # Interactive mode
   ./run.sh

   # Single question
   ./run.sh --question "Your question here"

   # Specific ZIM file (default: newest complete *.zim in ~/wikipedia-offline)
   ./run.sh --zim /path/to/wikipedia_en_all_nopic_2026-06.zim   # or export WIKI_ZIM=...

   # Specific models
   ./run.sh --model gemma4:26b --selection-model qwen3.6:35b

## Useful Commands

# Run Kiwix on its own (browse at http://localhost:8080)
scripts/start_offline_rag.sh

# List available Ollama models
ollama list

# Unit tests
uv run pytest -m "not integration"

========================================
For more details, see README.md
========================================
USAGE
