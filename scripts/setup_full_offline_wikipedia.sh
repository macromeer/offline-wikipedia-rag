#!/bin/bash

# Full Offline Wikipedia Setup with Kiwix
# This downloads the complete English Wikipedia (~100GB compressed)

echo "==========================================="
echo "Full Offline Wikipedia Setup"
echo "==========================================="
echo ""

# Check if kiwix-tools is installed
if ! command -v kiwix-serve &> /dev/null; then
    echo "📦 Installing Kiwix..."
    
    # Download kiwix-tools
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Detecting Linux system..."
        
        # Download latest kiwix-tools for Linux
        KIWIX_VERSION="3.7.0-3"
        KIWIX_URL="https://download.kiwix.org/release/kiwix-tools/kiwix-tools_linux-x86_64-${KIWIX_VERSION}.tar.gz"
        
        echo "Downloading Kiwix tools..."
        wget -O /tmp/kiwix-tools.tar.gz "$KIWIX_URL"
        
        echo "Extracting..."
        tar -xzf /tmp/kiwix-tools.tar.gz -C /tmp/
        
        echo "Installing to ~/.local/bin/..."
        mkdir -p ~/.local/bin
        cp /tmp/kiwix-tools_linux-x86_64-*/kiwix-serve ~/.local/bin/
        chmod +x ~/.local/bin/kiwix-serve
        
        # Add to PATH if not already there
        if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
            export PATH="$HOME/.local/bin:$PATH"
        fi
        
        rm -rf /tmp/kiwix-tools*
        
        echo "✓ Kiwix installed to ~/.local/bin/kiwix-serve"
    else
        echo "❌ Unsupported OS. Please install Kiwix manually from: https://www.kiwix.org"
        exit 1
    fi
fi

echo "✓ Kiwix is available"

# Create directory for Wikipedia ZIM file
WIKI_DIR="$HOME/wikipedia-offline"
mkdir -p "$WIKI_DIR"

echo ""
echo "📥 Downloading Wikipedia ZIM file..."
echo "⚠  WARNING: This is a ~95GB download!"
echo "   It will take several hours depending on your connection."
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Download Wikipedia ZIM file
cd "$WIKI_DIR"

# Use the latest English Wikipedia (no images, ~95GB)
ZIM_FILE="wikipedia_en_all_maxi_2024-01.zim"
ZIM_URL="https://download.kiwix.org/zim/wikipedia/${ZIM_FILE}"

echo ""
echo "Downloading to: $WIKI_DIR/$ZIM_FILE"
echo "This will take a long time..."

# Use wget with resume capability
wget -c "$ZIM_URL"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Download complete!"
    echo ""
    echo "==========================================="
    echo "Setup Instructions"
    echo "==========================================="
    echo ""
    echo "1. Start Kiwix server:"
    echo "   kiwix-serve $WIKI_DIR/$ZIM_FILE"
    echo ""
    echo "2. Access Wikipedia at: http://localhost:8080"
    echo ""
    echo "3. To use with RAG, run:"
    echo "   python wikipedia_rag_kiwix.py"
    echo ""
else
    echo "❌ Download failed or interrupted"
    echo "💡 You can resume by running this script again"
    exit 1
fi
