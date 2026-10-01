#!/bin/bash
# Convenience wrapper: runs the app in the project's uv environment
# (uv creates/syncs .venv on first run). All arguments are passed through.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Install it with:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# --project rather than cd, so relative paths like --zim ./file.zim keep working
exec uv run --quiet --project "$SCRIPT_DIR" python "$SCRIPT_DIR/wikipedia_rag_kiwix.py" "$@"
