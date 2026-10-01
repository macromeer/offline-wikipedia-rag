"""Finding the Wikipedia ZIM file to use."""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

ZIM_SEARCH_PATHS = [
    Path.home() / "wikipedia-offline",
    Path.home() / "Downloads",
    Path("/data/wikipedia"),
    Path("/var/lib/kiwix"),
]


def is_complete_zim(path: Path) -> bool:
    """
    Check the ZIM header: magic number, and file size equal to the MD5
    checksum position + 16. Rejects partial downloads without reading the file.
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(80)
        if len(header) < 80 or header[:4] != b'ZIM\x04':
            return False
        checksum_pos = int.from_bytes(header[72:80], 'little')
        return path.stat().st_size == checksum_pos + 16
    except OSError:
        return False


def _zim_sort_key(path: Path):
    """Newest dump first: date suffix in the name (e.g. _2026-06), then mtime"""
    match = re.search(r'_(\d{4}-\d{2})\.zim$', path.name)
    return (match.group(1) if match else '', path.stat().st_mtime)


def find_zim_files(search_paths: List[Path] = None) -> List[Path]:
    """Find complete ZIM files in common locations, newest first"""
    zim_files = []
    for path in search_paths or ZIM_SEARCH_PATHS:
        if not path.exists():
            continue
        for zim in path.glob("*.zim"):
            if is_complete_zim(zim):
                zim_files.append(zim)
            else:
                print(f"⚠ Skipping incomplete ZIM (still downloading?): {zim}", file=sys.stderr)
    return sorted(zim_files, key=_zim_sort_key, reverse=True)


def resolve_zim_path(cli_value: str = None) -> Optional[Path]:
    """Pick the ZIM to use: --zim flag, then WIKI_ZIM env var, then auto-discovery"""
    explicit = cli_value or os.environ.get('WIKI_ZIM')
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ZIM file not found: {path}")
        if not is_complete_zim(path):
            raise ValueError(f"Not a complete ZIM file (still downloading?): {path}")
        return path
    zim_files = find_zim_files()
    return zim_files[0] if zim_files else None
