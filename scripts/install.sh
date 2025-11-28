#!/bin/bash

# One-line installer for Offline Wikipedia RAG
# Installs everything needed: Ollama, DeepSeek, Wikipedia, Python environment

set -e  # Exit on error

BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}🌐 Offline Wikipedia RAG Installer${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""
echo "This will install:"
echo "  • Ollama AI runtime"
echo "  • Mistral-7B model (~4.4GB) - article selection"
echo "  • Llama-3.1-8B model (~4.9GB) - answer synthesis"
echo "  • Full English Wikipedia (~102GB)"
echo "  • Python environment"
echo ""
echo -e "${YELLOW}⚠️  Total download: ~112GB${NC}"
echo -e "${YELLOW}⚠️  Time needed: 2-8 hours${NC}"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

# Check system requirements
echo ""
echo -e "${BLUE}📋 Checking system requirements...${NC}"

# Check disk space
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 120 ]; then
    echo -e "${RED}❌ Error: Need at least 120GB free space, have ${AVAILABLE_SPACE}GB${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Disk space: ${AVAILABLE_SPACE}GB available"

# Check RAM
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM" -lt 8 ]; then
    echo -e "${YELLOW}⚠️  Warning: 8GB+ RAM recommended, have ${TOTAL_RAM}GB${NC}"
fi
echo -e "${GREEN}✓${NC} RAM: ${TOTAL_RAM}GB"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo -e "${RED}❌ Unsupported OS: $OSTYPE${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} OS: $OS"

# Step 1: Install Ollama
echo ""
echo -e "${BLUE}🤖 Step 1/5: Installing Ollama...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama already installed"
else
    if [ "$OS" == "linux" ]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [ "$OS" == "macos" ]; then
        echo "Please install Ollama from: https://ollama.ai/download"
        echo "Then run this script again."
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Ollama installed"
fi

# Start Ollama service
echo "Starting Ollama service..."
if [ "$OS" == "linux" ]; then
    if command -v systemctl &> /dev/null; then
        sudo systemctl start ollama || ollama serve &
    else
        ollama serve &
    fi
else
    ollama serve &
fi
sleep 3

# Step 2: Pull AI models
echo ""
echo -e "${BLUE}🧠 Step 2/5: Downloading AI models...${NC}"

# Pull Mistral-7B for article selection
if ollama list | grep -q "mistral:7b"; then
    echo -e "${GREEN}✓${NC} Mistral-7B already downloaded"
else
    echo "Downloading Mistral-7B (~4.4GB)..."
    ollama pull mistral:7b
    echo -e "${GREEN}✓${NC} Mistral-7B downloaded"
fi

# Pull Llama-3.1-8B for synthesis
if ollama list | grep -q "llama3.1:8b"; then
    echo -e "${GREEN}✓${NC} Llama-3.1-8B already downloaded"
else
    echo "Downloading Llama-3.1-8B (~4.9GB)..."
    ollama pull llama3.1:8b
    echo -e "${GREEN}✓${NC} Llama-3.1-8B downloaded"
fi

# Step 3: Install Python environment
echo ""
echo -e "${BLUE}🐍 Step 3/5: Setting up Python environment...${NC}"

# Check for conda/mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo -e "${YELLOW}⚠️  Conda/Mamba not found. Installing Miniforge...${NC}"
    if [ "$OS" == "linux" ]; then
        wget -O /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    elif [ "$OS" == "macos" ]; then
        wget -O /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
    fi
    bash /tmp/miniforge.sh -b -p $HOME/miniforge3
    rm /tmp/miniforge.sh
    export PATH="$HOME/miniforge3/bin:$PATH"
    CONDA_CMD="mamba"
    echo -e "${GREEN}✓${NC} Miniforge installed"
fi

# Create environment
$CONDA_CMD env create -f environment.yml -y 2>/dev/null || $CONDA_CMD env update -f environment.yml -y
echo -e "${GREEN}✓${NC} Python environment ready"

# Step 4: Download Wikipedia
echo ""
echo -e "${BLUE}📚 Step 4/5: Downloading Wikipedia (~102GB)...${NC}"
echo "This will take 2-8 hours depending on your connection."
echo ""

./setup_full_offline_wikipedia.sh

# Step 5: Test the system
echo ""
echo -e "${BLUE}🧪 Step 5/5: Testing the system...${NC}"

# Start Kiwix if not running
if ! pgrep -f "kiwix-serve" > /dev/null; then
    ./start_offline_rag.sh
fi

# Run test
$CONDA_CMD run -n wikipedia-rag python wikipedia_rag_kiwix.py --question "What is AI?" > /tmp/rag_test.txt 2>&1 || true

if grep -q "Answer:" /tmp/rag_test.txt; then
    echo -e "${GREEN}✓${NC} System test passed"
else
    echo -e "${YELLOW}⚠️  System test had issues, but installation complete${NC}"
fi

# Success message
echo ""
echo -e "${GREEN}${BOLD}============================================${NC}"
echo -e "${GREEN}${BOLD}✅ Installation Complete!${NC}"
echo -e "${GREEN}${BOLD}============================================${NC}"
echo ""
echo -e "${BOLD}🚀 Quick Start:${NC}"
echo ""
echo "  1. Activate environment:"
echo "     conda activate wikipedia-rag"
echo ""
echo "  2. Start chatting:"
echo "     python wikipedia_rag_kiwix.py"
echo ""
echo "  Or ask a single question:"
echo "     python wikipedia_rag_kiwix.py --question 'What is quantum computing?'"
echo ""
echo -e "${BOLD}📖 Documentation:${NC} See README.md for more options"
echo -e "${BOLD}🆘 Help:${NC} Run './help' for quick reference"
echo ""
echo "Enjoy your private, offline AI assistant! 🎉"
