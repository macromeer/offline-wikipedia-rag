"""
Dense (embedding) retrieval over a prebuilt vector index of ZIM chunks.

The index is built once per ZIM with `python -m retrieval.build_dense`. Every
chunk (as produced by chunk_sections) is embedded with Qwen3-Embedding-0.6B,
truncated to `dim` dimensions (the model is Matryoshka-trained), normalised,
stored as int8 with a per-vector scale (cosine does not care about scale), and
put into a usearch HNSW graph that is memory-mapped at query time.

Each vector's key packs (ZIM entry index, chunk ordinal, 16-bit checksum of
the chunk text). Nothing else is stored: hits are turned back into chunks by
re-reading the article, and the checksum catches an index built with a
different chunker.
"""

import json
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Same weights in two packagings: Ollama (GGUF Q8_0) embeds queries, and builds
# small ZIMs; the Hugging Face model builds large ones on the GPU. Measured on
# Wikipedia chunks: cosine between the two >= 0.997, identical top-10.
OLLAMA_MODEL = 'qwen3-embedding:0.6b'
HF_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
# Qwen3-Embedding wants an instruction on queries and none on documents
QUERY_PREFIX = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
DEFAULT_DIM = 512
# Text is cut here before embedding, so both backends see the same input and the
# model never truncates (~550 tokens at ~3.7 chars/token for Wikipedia prose)
MAX_EMBED_CHARS = 2000
EMBED_NUM_CTX = 1024
SEARCH_EXPANSION = 128          # HNSW ef at query time; k is up to ~50

META_FILE = 'meta.json'
INDEX_FILE = 'index.usearch'

_ORDINAL_BITS, _CHECK_BITS = 16, 16


def text_check(text: str) -> int:
    return zlib.crc32(text.encode('utf-8')) & ((1 << _CHECK_BITS) - 1)


def pack_key(entry_index: int, ordinal: int, text: str) -> int:
    """64-bit key: entry index (32 bits) | chunk ordinal (16) | checksum of the chunk text (16)"""
    return (entry_index << (_ORDINAL_BITS + _CHECK_BITS)) | (ordinal << _CHECK_BITS) | text_check(text)


def unpack_key(key: int):
    """(entry index, chunk ordinal, checksum)"""
    key = int(key)
    return (key >> (_ORDINAL_BITS + _CHECK_BITS), (key >> _CHECK_BITS) & ((1 << _ORDINAL_BITS) - 1),
            key & ((1 << _CHECK_BITS) - 1))


def embed_text(chunk_text: str) -> str:
    return chunk_text[:MAX_EMBED_CHARS]


def quantize(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Truncate to dim, then scale each vector so its largest component is +-127 (int8)"""
    v = np.asarray(vectors, dtype=np.float32)[:, :dim]
    peak = np.abs(v).max(axis=1, keepdims=True)
    peak[peak == 0] = 1.0
    return np.round(v * (127.0 / peak)).astype(np.int8)


def default_index_dir(zim_uuid: str) -> Path:
    base = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share')
    return base / 'offline-wikipedia-rag' / f'dense-{zim_uuid}'


class OllamaEmbedder:
    """Embeddings through Ollama /api/embed"""

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import ollama
        response = ollama.embed(model=self.model, input=list(texts), truncate=True,
                                options={'num_ctx': EMBED_NUM_CTX})
        return np.asarray(response.embeddings, dtype=np.float32)


@dataclass
class DenseHit:
    entry_index: int
    ordinal: int
    check: int
    score: float                    # cosine similarity (int8-approximate)


class DenseIndex:
    """A finished index directory: meta.json + index.usearch, opened memory-mapped"""

    def __init__(self, directory, embedder=None):
        from usearch.index import Index
        self.directory = Path(directory)
        with open(self.directory / META_FILE, encoding='utf-8') as f:
            self.meta = json.load(f)
        if not self.meta.get('complete'):
            raise ValueError(f"Dense index in {self.directory} is not finished; rerun the builder")
        self.dim = int(self.meta['dim'])
        self.index = Index.restore(str(self.directory / INDEX_FILE), view=True)
        self.index.expansion_search = SEARCH_EXPANSION
        self.embedder = embedder or OllamaEmbedder(self.meta.get('ollama_model', OLLAMA_MODEL))

    @classmethod
    def find(cls, zim_uuid: str, directory=None) -> Optional['DenseIndex']:
        """The finished index for this ZIM in directory (default: default_index_dir), or None"""
        directory = Path(directory) if directory else default_index_dir(zim_uuid)
        try:
            with open(directory / META_FILE, encoding='utf-8') as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return None
        if not meta.get('complete') or meta.get('zim_uuid') != zim_uuid:
            return None
        return cls(directory)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def chunk_tokens(self) -> int:
        return int(self.meta['chunk_tokens'])

    def search(self, query: str, k: int) -> List[DenseHit]:
        vector = self.embedder.embed([QUERY_PREFIX + query.strip()])
        matches = self.index.search(quantize(vector, self.dim)[0], k)
        return [DenseHit(*unpack_key(key), score=1.0 - float(distance))
                for key, distance in zip(matches.keys, matches.distances)]
