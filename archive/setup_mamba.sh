#!/bin/bash

# Setup script with Mamba environment support

echo "==========================================="
echo "Wikipedia RAG with Ollama - Mamba Setup"
echo "==========================================="
echo ""

# Check if mamba or conda is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✓ Mamba found"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✓ Conda found (consider installing mamba for faster setup: conda install mamba -n base -c conda-forge)"
else
    echo "❌ Neither Mamba nor Conda is installed"
    echo "💡 Install Miniforge (includes mamba): https://github.com/conda-forge/miniforge"
    echo "   or Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

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

# Create/update environment
echo ""
echo "📦 Creating/updating Mamba environment 'wikipedia-rag'..."
$CONDA_CMD env create -f environment.yml 2>/dev/null || $CONDA_CMD env update -f environment.yml

# Activate environment and run tests
echo ""
echo "🧪 Running system tests..."
echo ""

# Run tests in the environment
$CONDA_CMD run -n wikipedia-rag python test_system.py

# Check if tests passed
if [ $? -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "✅ Setup complete!"
    echo "==========================================="
    echo ""
    echo "🔧 Activate the environment:"
    echo "   $CONDA_CMD activate wikipedia-rag"
    echo ""
    echo "🚀 Quick start options:"
    echo ""
    echo "1. Online mode (requires internet for Wikipedia API):"
    echo "   python wikipedia_rag.py"
    echo ""
    echo "2. Build offline index first:"
    echo "   python wikipedia_indexer.py --file sample_titles.txt"
    echo ""
    echo "3. Then use offline mode:"
    echo "   python wikipedia_rag_offline.py"
    echo ""
    echo "4. Single question:"
    echo "   python wikipedia_rag.py --question 'What is Python?'"
    echo ""
else
    echo ""
    echo "❌ Setup failed. Check errors above."
    exit 1
fi
