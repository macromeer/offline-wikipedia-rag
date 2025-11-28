#!/bin/bash

# Simple usage script - shows common commands

cat << 'EOF'
========================================
Offline Wikipedia RAG - Usage Guide
========================================

## First Time Setup (do once)
./setup_full_offline_wikipedia.sh

## Daily Use

# 1. Make sure Kiwix is running
./start_offline_rag.sh

# 2. Activate environment
mamba activate wikipedia-rag

# 3. Use the RAG system

   # Interactive mode
   python wikipedia_rag_kiwix.py

   # Single question
   python wikipedia_rag_kiwix.py --question "Your question here"

   # Specify model
   python wikipedia_rag_kiwix.py --model deepseek-r1:latest

## Useful Commands

# Check if Kiwix is running
ps aux | grep kiwix-serve

# Restart Kiwix
pkill kiwix-serve
~/.local/bin/kiwix-serve --port=8080 ~/wikipedia-offline/*.zim &

# Access Wikipedia in browser
http://localhost:8080

# List available Ollama models
ollama list

# Test the system
python test_system.py

========================================
For more details, see READY.md
========================================
EOF
