#!/bin/bash

# Start kiwix-serve on the newest complete Wikipedia ZIM (for browsing at
# http://localhost:8080). ./run.sh starts it automatically; this is only needed
# to keep the server running on its own. Respects WIKI_ZIM.

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PORT="${PORT:-8080}"

if pgrep -x kiwix-serve > /dev/null; then
    echo "✓ Kiwix server is already running"
    exit 0
fi

KIWIX_SERVE="$(command -v kiwix-serve || echo "$HOME/.local/bin/kiwix-serve")"
if [ ! -x "$KIWIX_SERVE" ]; then
    echo "❌ kiwix-serve not found. Run scripts/setup_full_offline_wikipedia.sh"
    exit 1
fi

ZIM="$(uv run --quiet --project "$REPO_DIR" python -c \
    'from wikipedia_rag_kiwix import resolve_zim_path; print(resolve_zim_path() or "")')"
if [ -z "$ZIM" ]; then
    echo "❌ No complete Wikipedia ZIM found. Run scripts/setup_full_offline_wikipedia.sh"
    exit 1
fi

echo "📚 Starting Kiwix server: $ZIM"
"$KIWIX_SERVE" --port="$PORT" "$ZIM" &
sleep 2
echo "✓ Kiwix server started at http://localhost:$PORT"
