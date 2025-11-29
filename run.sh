#!/bin/bash
# Convenience wrapper that automatically activates environment and runs the script

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Initialize conda/mamba
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "❌ Conda/Mamba not found. Please install mamba/conda first."
    echo "   See: https://mamba.readthedocs.io/en/latest/installation.html"
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "^wikipedia-rag "; then
    echo "❌ Environment 'wikipedia-rag' not found."
    echo "   Create it with: mamba env create -f environment.yml"
    exit 1
fi

# Activate environment
echo "🔧 Activating wikipedia-rag environment..."
conda activate wikipedia-rag

echo "✓ Environment activated"
echo ""

# Run the Python script with all arguments passed through
python wikipedia_rag_kiwix.py "$@"
