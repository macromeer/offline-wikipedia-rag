#!/bin/bash
# Convenient test runner script

set -e

cd "$( dirname "${BASH_SOURCE[0]}" )/.."

echo "=== Running Wikipedia RAG Test Suite ==="
echo ""
echo "🧪 Running unit tests..."
echo ""

uv run pytest tests/ -v -m "not integration" --tb=short

echo ""
echo "✅ All tests passed!"
echo ""
echo "💡 Tips:"
echo "  - Run with coverage: uv run pytest tests/ --cov=. --cov-report=html"
echo "  - Run integration tests: uv run pytest tests/ -m integration (requires Kiwix/Ollama)"
echo "  - Run specific test: uv run pytest tests/test_rag_functions.py::TestSearchTermExtraction -v"
