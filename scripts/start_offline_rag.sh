#!/bin/bash

# Start the offline Wikipedia RAG system

echo "🚀 Starting Offline Wikipedia RAG System"
echo "========================================"
echo ""

# Check if Kiwix is already running
if pgrep -f "kiwix-serve" > /dev/null; then
    echo "✓ Kiwix server is already running"
else
    echo "📚 Starting Kiwix server..."
    ~/.local/bin/kiwix-serve --port=8080 ~/wikipedia-offline/*.zim &
    sleep 2
    echo "✓ Kiwix server started at http://localhost:8080"
fi

echo ""
echo "✅ Ready! Now run:"
echo ""
echo "   mamba activate wikipedia-rag"
echo "   python wikipedia_rag_kiwix.py"
echo ""
echo "Or for a single question:"
echo "   python wikipedia_rag_kiwix.py --question 'Your question here'"
echo ""
