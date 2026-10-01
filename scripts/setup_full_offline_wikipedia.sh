#!/bin/bash

# Offline Wikipedia setup: installs kiwix-serve and downloads the newest
# English Wikipedia ZIM from the Kiwix mirror (resumable).
#
# Usage: setup_full_offline_wikipedia.sh [--variant nopic|mini|maxi] [--yes] [--dry-run]
#   nopic  all articles, no images (~50GB)            [default]
#   mini   lead section + infobox only (~13GB), for small disks
#   maxi   all articles with images (~120GB)
# Environment: WIKI_DIR (download directory, default ~/wikipedia-offline)

set -e

MIRROR="https://download.kiwix.org/zim/wikipedia"
WIKI_DIR="${WIKI_DIR:-$HOME/wikipedia-offline}"
VARIANT=""
ASSUME_YES=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '3,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "==========================================="
echo "Offline Wikipedia Setup"
echo "==========================================="
echo ""

# --- kiwix-serve -------------------------------------------------------------
if [ "$DRY_RUN" -eq 0 ] && ! command -v kiwix-serve &> /dev/null && [ ! -x "$HOME/.local/bin/kiwix-serve" ]; then
    echo "📦 Installing kiwix-serve..."
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        echo "❌ Unsupported OS. Install Kiwix tools manually from: https://www.kiwix.org"
        exit 1
    fi
    case "$(uname -m)" in
        x86_64) KIWIX_ARCH="x86_64" ;;
        aarch64|arm64) KIWIX_ARCH="aarch64" ;;
        *) echo "❌ Unsupported architecture: $(uname -m)"; exit 1 ;;
    esac
    # Unversioned name always points to the latest release
    KIWIX_URL="https://download.kiwix.org/release/kiwix-tools/kiwix-tools_linux-${KIWIX_ARCH}.tar.gz"
    TMP_DIR="$(mktemp -d)"
    wget -q --show-progress -O "$TMP_DIR/kiwix-tools.tar.gz" "$KIWIX_URL"
    tar -xzf "$TMP_DIR/kiwix-tools.tar.gz" -C "$TMP_DIR"
    mkdir -p "$HOME/.local/bin"
    cp "$TMP_DIR"/kiwix-tools_*/kiwix-serve "$HOME/.local/bin/"
    chmod +x "$HOME/.local/bin/kiwix-serve"
    rm -rf "$TMP_DIR"
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo "💡 Add ~/.local/bin to your PATH (the app also finds kiwix-serve there on its own)"
    fi
    echo "✓ kiwix-serve installed to ~/.local/bin/kiwix-serve"
fi
[ "$DRY_RUN" -eq 0 ] && echo "✓ kiwix-serve is available"

# --- choose variant ----------------------------------------------------------
if [ -z "$VARIANT" ]; then
    if [ "$ASSUME_YES" -eq 1 ] || [ ! -t 0 ]; then
        VARIANT="nopic"
    else
        echo ""
        echo "Which Wikipedia edition?"
        echo "  1) nopic  all articles, no images (~50GB)  [recommended]"
        echo "  2) mini   intro + infobox only (~13GB), for small disks"
        echo "  3) maxi   all articles with images (~120GB)"
        read -p "Choice [1]: " -r CHOICE
        case "$CHOICE" in
            2) VARIANT="mini" ;;
            3) VARIANT="maxi" ;;
            *) VARIANT="nopic" ;;
        esac
    fi
fi
case "$VARIANT" in
    nopic|mini|maxi) ;;
    *) echo "❌ Unknown variant: $VARIANT (use nopic, mini or maxi)"; exit 1 ;;
esac

# --- find newest ZIM on the mirror -------------------------------------------
echo ""
echo "🔎 Looking up the newest wikipedia_en_all_${VARIANT} dump..."
LISTING="$(curl -fsSL "$MIRROR/")" || { echo "❌ Could not reach $MIRROR/"; exit 1; }
ZIM_FILE="$(echo "$LISTING" | grep -oE "wikipedia_en_all_${VARIANT}_[0-9]{4}-[0-9]{2}\.zim" | sort -u | tail -1)"
if [ -z "$ZIM_FILE" ]; then
    echo "❌ No wikipedia_en_all_${VARIANT}_*.zim found at $MIRROR/"
    exit 1
fi
ZIM_SIZE="$(echo "$LISTING" | grep -F "$ZIM_FILE\"" | grep -oE '[0-9.]+[KMG]\s*$' | tr -d '[:space:]' | head -1)"
ZIM_URL="$MIRROR/$ZIM_FILE"
echo "✓ Newest: $ZIM_FILE ${ZIM_SIZE:+($ZIM_SIZE)}"
if [ "$VARIANT" = "mini" ]; then
    echo "⚠  mini contains only article intros; answers will be shallower"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "$ZIM_URL"
    exit 0
fi

# --- download ----------------------------------------------------------------
mkdir -p "$WIKI_DIR"
ZIM_SIZE_GB="$(echo "$ZIM_SIZE" | grep -oE '^[0-9]+' || true)"
AVAILABLE_GB="$(df -BG "$WIKI_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')"
EXISTING_GB=0
[ -f "$WIKI_DIR/$ZIM_FILE" ] && EXISTING_GB=$(( $(stat -c %s "$WIKI_DIR/$ZIM_FILE") / 1073741824 ))
if [[ "$ZIM_SIZE" == *G ]] && [ -n "$ZIM_SIZE_GB" ] && [ $((AVAILABLE_GB + EXISTING_GB)) -lt "$ZIM_SIZE_GB" ]; then
    echo "❌ Not enough disk space in $WIKI_DIR: need ~${ZIM_SIZE_GB}GB, have ${AVAILABLE_GB}GB free"
    exit 1
fi

echo ""
echo "📥 Downloading to: $WIKI_DIR/$ZIM_FILE"
echo "   This can take several hours. Re-run this script to resume."
if [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ]; then
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

cd "$WIKI_DIR"

# Two concurrent `wget -c` runs append to the same file and corrupt it
exec 9> "$WIKI_DIR/.$ZIM_FILE.lock"
if ! flock -n 9; then
    echo "❌ Another download of $ZIM_FILE is already running in $WIKI_DIR"
    exit 1
fi

if wget -c "$ZIM_URL"; then
    echo ""
    echo "🔐 Verifying SHA-256 (a few minutes)..."
    if curl -fsSL "$ZIM_URL.sha256" | sha256sum -c -; then
        rm -f "$WIKI_DIR/.$ZIM_FILE.lock"
    else
        echo "❌ Checksum mismatch: $WIKI_DIR/$ZIM_FILE is corrupt."
        echo "   Delete it and run this script again."
        exit 1
    fi
    echo ""
    echo "✅ Download complete: $WIKI_DIR/$ZIM_FILE"
    echo ""
    echo "Run the assistant (it finds the newest complete ZIM in $WIKI_DIR):"
    echo "   ./run.sh"
    echo "Or point it at this file explicitly:"
    echo "   ./run.sh --zim $WIKI_DIR/$ZIM_FILE     (or export WIKI_ZIM=...)"
    echo ""
else
    echo "❌ Download failed or interrupted"
    echo "💡 You can resume by running this script again"
    exit 1
fi
