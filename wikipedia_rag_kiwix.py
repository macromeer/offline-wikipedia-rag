#!/usr/bin/env python3
"""
Offline Wikipedia RAG over a Kiwix ZIM file. Entry point; the code lives in
wikirag/ (app) and retrieval/ (ZIM search and chunking). See --help.
"""

import sys

from wikirag.cli import main

if __name__ == "__main__":
    sys.exit(main())
