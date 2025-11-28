#!/bin/bash
# Convenient test runner script

set -e

echo "=== Running Wikipedia RAG Test Suite ==="
echo ""

# Activate environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "wikipedia-rag" ]; then
    echo "⚠️  Activating wikipedia-rag environment..."
    eval "$(conda shell.bash hook)"
    conda activate wikipedia-rag
fi

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo "📦 Installing test dependencies..."
    pip install -q pytest pytest-cov pytest-mock
fi

echo "🧪 Running unit tests..."
echo ""

# Run unit tests
pytest tests/ -v -m "not integration" --tb=short

echo ""
echo "✅ All tests passed!"
echo ""
echo "💡 Tips:"
echo "  - Run with coverage: pytest tests/ --cov=. --cov-report=html"
echo "  - Run integration tests: pytest tests/ -m integration (requires Kiwix/Ollama)"
echo "  - Run specific test: pytest tests/test_rag_functions.py::TestSearchTermExtraction -v"
