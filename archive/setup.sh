#!/bin/bash

# Quick Start Guide for Wikipedia RAG with Ollama

echo "==========================================="
echo "Wikipedia RAG with Ollama - Quick Setup"
echo "==========================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed"
    echo "💡 Install from: https://ollama.ai"
    exit 1
fi

echo "✓ Ollama found"

# Check if Ollama is running
if ! ollama list &> /dev/null; then
    echo "⚠ Ollama is not running"
    echo "💡 Starting Ollama..."
    ollama serve &
    sleep 3
fi

echo "✓ Ollama is running"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found"

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Run tests
echo ""
echo "🧪 Running system tests..."
python3 test_system.py

# Check if tests passed
if [ $? -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "✅ Setup complete!"
    echo "==========================================="
    echo ""
    echo "🚀 Quick start options:"
    echo ""
    echo "1. Online mode (requires internet for Wikipedia API):"
    echo "   python3 wikipedia_rag.py"
    echo ""
    echo "2. Build offline index first:"
    echo "   python3 wikipedia_indexer.py --file sample_titles.txt"
    echo ""
    echo "3. Then use offline mode:"
    echo "   python3 wikipedia_rag_offline.py"
    echo ""
    echo "4. Single question:"
    echo "   python3 wikipedia_rag.py --question 'What is Python?'"
    echo ""
else
    echo ""
    echo "❌ Setup failed. Check errors above."
    exit 1
fi
