"""
Build the dense index for a ZIM (see retrieval/dense.py for the format).

    uv run --group dense-build python -m retrieval.build_dense             # GPU(s), all sections
    uv run --group dense-build python -m retrieval.build_dense --scope leads
    uv run python -m retrieval.build_dense --backend ollama                # no torch; slow, small ZIMs

Entries are processed in blocks; each finished block is written as a shard, so
an interrupted build resumes where it stopped (run the same command again).
With the torch backend every GPU runs its own worker process. When all blocks
are done, the shards are loaded into one HNSW graph, which needs RAM for all
vectors (dim bytes each), and then deleted.

Skipped: redirects, the main page, disambiguation pages and "List of" articles.
--scope leads embeds only article introductions (about 1/6 of the text).
Running --scope all on a finished leads index upgrades it: only the remaining
sections are embedded (into shards-rest/) and then added to the existing graph.
The leads index stays usable until the upgraded one replaces it.
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .dense import (
    DEFAULT_DIM, EMBED_NUM_CTX, HF_MODEL, INDEX_FILE, MAX_EMBED_CHARS, META_FILE, OLLAMA_MODEL,
    OllamaEmbedder, default_index_dir, embed_text, pack_key, quantize,
)
from .zim_store import DEFAULT_CHUNK_TOKENS, ZimStore

FORMAT_VERSION = 1
DEFAULT_BLOCK_SIZE = 50_000       # ZIM entries per shard (~20k articles, ~80k chunks on English Wikipedia)
SHARD_DIR = 'shards'
REST_SHARD_DIR = 'shards-rest'     # non-lead sections, while upgrading a leads index to all
OLLAMA_BATCH = 64


class TorchEmbedder:
    """Qwen3-Embedding on one GPU with transformers: last-token pooling, length-sorted batches.

    On CUDA the model is compiled (torch.compile): 51k vs 34k tokens/s on an RTX PRO 4000 Blackwell,
    same vectors (cosine >= 0.9995). Lengths are padded to multiples of 64 to limit recompiles.
    FP8 (torchao) was 2.3x faster but gave NaN vectors and cosine ~0.99, so it is not used.
    """

    def __init__(self, device: str, model: str = HF_MODEL, batch_tokens: int = 65536, max_batch: int = 512,
                 compile: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device
        self.batch_tokens = batch_tokens
        self.max_batch = max_batch
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
        cuda = device.startswith('cuda')
        dtype = torch.bfloat16 if cuda else torch.float32
        self.model = AutoModel.from_pretrained(model, dtype=dtype, attn_implementation='sdpa').to(device).eval()
        self.bucket = 64 if cuda and compile else 1
        if cuda and compile:
            self.model = torch.compile(self.model, dynamic=True)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        ids = self.tokenizer(list(texts), truncation=True, max_length=EMBED_NUM_CTX)['input_ids']
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out = torch.empty((len(ids), self.model.config.hidden_size), dtype=torch.float32, device=self.device)
        pad = self.tokenizer.pad_token_id
        i = 0
        with torch.inference_mode():
            while i < len(order):
                width = -(-len(ids[order[i]]) // self.bucket) * self.bucket
                batch = order[i:i + max(1, min(self.max_batch, self.batch_tokens // width))]
                input_ids = torch.full((len(batch), width), pad, dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, j in enumerate(batch):       # left padding: the last position is EOS
                    input_ids[row, width - len(ids[j]):] = torch.tensor(ids[j])
                    mask[row, width - len(ids[j]):] = 1
                hidden = self.model(input_ids=input_ids.to(self.device, non_blocking=True),
                                    attention_mask=mask.to(self.device, non_blocking=True)).last_hidden_state
                out[torch.tensor(batch, device=self.device)] = hidden[:, -1].float()
                i += len(batch)
        return out.cpu().numpy()       # one sync per block instead of one per batch


class BatchedOllamaEmbedder(OllamaEmbedder):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        parts = [OllamaEmbedder.embed(self, texts[i:i + OLLAMA_BATCH])
                 for i in range(0, len(texts), OLLAMA_BATCH)]
        return np.concatenate(parts) if parts else np.zeros((0, 1), np.float32)


def block_chunks(store: ZimStore, start: int, stop: int, scope: str) -> Tuple[List[int], List[str]]:
    """Keys and embedding inputs for the chunks of entries [start, stop); scope is all, leads or rest (all - leads)"""
    keys, texts = [], []
    for index, article in store.iter_articles(start, stop):
        if article.is_disambiguation or article.title.startswith('List of'):
            continue
        for ordinal, chunk in enumerate(article.chunks):
            if scope == 'leads' and not chunk.is_lead:
                break       # the lead's chunks come first
            if scope == 'rest' and chunk.is_lead:
                continue
            keys.append(pack_key(index, ordinal, chunk.text))
            texts.append(embed_text(chunk.text))
    return keys, texts


def shard_path(out: Path, start: int, shard_dir: str = SHARD_DIR) -> Path:
    return out / shard_dir / f'{start:010d}.npz'


def write_shard(out: Path, start: int, keys: Sequence[int], vectors: np.ndarray, shard_dir: str = SHARD_DIR) -> None:
    path = shard_path(out, start, shard_dir)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        np.savez(f, keys=np.asarray(keys, dtype=np.uint64), vectors=vectors)
    os.replace(tmp, path)


def _worker(device: str, settings: dict, tasks, results) -> None:
    """One embedding device: takes block starts from tasks until None, parsing the next block while embedding"""
    try:
        store = ZimStore(settings['zim'], chunk_tokens=settings['chunk_tokens'])
        if device == 'ollama':
            embedder = BatchedOllamaEmbedder(OLLAMA_MODEL)
        else:
            embedder = TorchEmbedder(device, batch_tokens=settings['batch_tokens'])
        out, size, scope, dim = Path(settings['out']), settings['block_size'], settings['scope'], settings['dim']

        def prepare(start):
            return start, block_chunks(store, start, start + size, scope)

        with ThreadPoolExecutor(max_workers=1) as parser:
            start = tasks.get()
            pending = parser.submit(prepare, start) if start is not None else None
            while pending:
                start, (keys, texts) = pending.result()
                following = tasks.get()
                pending = parser.submit(prepare, following) if following is not None else None
                t = time.perf_counter()
                vectors = quantize(embedder.embed(texts), dim) if texts else np.zeros((0, dim), np.int8)
                write_shard(out, start, keys, vectors, settings['shard_dir'])
                results.put(('block', device, start, len(texts), sum(map(len, texts)), time.perf_counter() - t))
    except BaseException as e:      # report instead of dying silently in a child process
        results.put(('error', device, f'{type(e).__name__}: {e}'))
        return
    results.put(('done', device))


def _default_devices(backend: str) -> List[str]:
    if backend == 'ollama':
        return ['ollama']
    import torch
    count = torch.cuda.device_count()
    return [f'cuda:{i}' for i in range(count)] if count else ['cpu']


def _format_hours(seconds: float) -> str:
    return f'{seconds / 3600:.1f} h' if seconds >= 3600 else f'{seconds / 60:.0f} min'


def embed_blocks(settings: dict, pending: List[int], devices: List[str]) -> bool:
    """Run one worker per device over the pending block starts; True if all of them finished"""
    ctx = mp.get_context('spawn')
    tasks, results = ctx.Queue(), ctx.Queue()
    for start in pending:
        tasks.put(start)
    for _ in devices:
        tasks.put(None)
    workers = [ctx.Process(target=_worker, args=(d, settings, tasks, results), daemon=True) for d in devices]
    for w in workers:
        w.start()
    started, done, chunks, running, ok = time.perf_counter(), 0, 0, len(workers), True
    try:
        while running:
            message = results.get()
            if message[0] == 'block':
                _, device, start, n, chars, seconds = message
                done += 1
                chunks += n
                elapsed = time.perf_counter() - started
                eta = elapsed / done * (len(pending) - done)
                print(f"  [{device}] entries {start:,}+: {n:,} chunks in {seconds:.0f}s "
                      f"({chars / 3.7 / max(seconds, 1e-6):,.0f} tok/s) | {done}/{len(pending)} blocks, "
                      f"{chunks:,} chunks, ETA {_format_hours(eta)}", flush=True)
            elif message[0] == 'error':
                print(f"❌ [{message[1]}] {message[2]}", flush=True)
                ok = False
                running -= 1
            else:
                running -= 1
    finally:
        for w in workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
    return ok and done == len(pending)


def finalize(out: Path, meta: dict, keep_shards: bool, threads: int, shard_dir: str = SHARD_DIR) -> int:
    """Load every shard into one HNSW graph, save it, mark the index complete.
    For an upgrade (shard_dir REST_SHARD_DIR) the shards are added to the existing graph."""
    from usearch.index import Index
    files = sorted((out / shard_dir).glob('*.npz'))
    if shard_dir == REST_SHARD_DIR:
        index = Index.restore(str(out / INDEX_FILE), view=False)
        index.expansion_add = 128
    else:
        index = Index(ndim=meta['dim'], metric='cos', dtype='i8', connectivity=16, expansion_add=128)
    already = len(index)
    total = already
    for f in files:
        with np.load(f) as shard:
            total += len(shard['keys'])
    t = time.perf_counter()
    for i, f in enumerate(files, 1):
        with np.load(f) as shard:
            if len(shard['keys']):
                index.add(shard['keys'], shard['vectors'], threads=threads)
        if i % 20 == 0 or i == len(files):
            print(f"  graph: {len(index):,}/{total:,} vectors ({time.perf_counter() - t:.0f}s)", flush=True)
    tmp = out / (INDEX_FILE + '.tmp')
    index.save(str(tmp))
    os.replace(tmp, out / INDEX_FILE)
    meta.update(complete=True, count=len(index), built=date.today().isoformat())
    if shard_dir == REST_SHARD_DIR:
        meta['scope'] = meta.pop('upgrade_to')
    write_meta(out, meta)
    if not keep_shards:
        for f in files:
            f.unlink()
        (out / shard_dir).rmdir()
    return len(index)


def write_meta(out: Path, meta: dict) -> None:
    tmp = out / (META_FILE + '.tmp')
    tmp.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    os.replace(tmp, out / META_FILE)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description='Build the dense (embedding) index for a Wikipedia ZIM')
    parser.add_argument('--zim', help='ZIM file (default: $WIKI_ZIM, else the newest in ~/wikipedia-offline)')
    parser.add_argument('--out', help='Index directory (default: ~/.local/share/offline-wikipedia-rag/dense-<zim uuid>)')
    parser.add_argument('--scope', choices=['all', 'leads'], default='all',
                        help='all: every section (default); leads: article introductions only, ~6x faster')
    parser.add_argument('--backend', choices=['torch', 'ollama'], default=None,
                        help='torch: Hugging Face model on the GPU(s), needs `--group dense-build` (default if '
                             f'installed); ollama: {OLLAMA_MODEL} through Ollama, no extra packages, ~3x slower')
    parser.add_argument('--devices', help='Comma-separated torch devices (default: every CUDA GPU)')
    parser.add_argument('--dim', type=int, default=DEFAULT_DIM, help=f'Vector dimensions, 32-1024 (default {DEFAULT_DIM})')
    parser.add_argument('--block-size', type=int, default=DEFAULT_BLOCK_SIZE, help='ZIM entries per shard')
    parser.add_argument('--batch-tokens', type=int, default=65536, help='Tokens per GPU batch (torch)')
    parser.add_argument('--max-blocks', type=int, default=None,
                        help='Embed at most this many pending blocks, then stop without building the graph (trial runs)')
    parser.add_argument('--keep-shards', action='store_true', help='Keep the shard files after building the graph')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    from wikirag.zimfiles import resolve_zim_path
    zim = resolve_zim_path(args.zim)
    if zim is None:
        print("❌ No ZIM file found; pass --zim PATH or set WIKI_ZIM")
        return 1
    backend = args.backend or ('torch' if importlib.util.find_spec('torch') else 'ollama')
    if backend == 'torch' and not importlib.util.find_spec('transformers'):
        print("❌ The torch backend needs: uv run --group dense-build python -m retrieval.build_dense ...")
        return 1
    if not 32 <= args.dim <= 1024:
        print("❌ --dim must be between 32 and 1024")
        return 1

    store = ZimStore(zim)
    uuid = str(store.archive.uuid)
    out = Path(args.out).expanduser() if args.out else default_index_dir(uuid)
    params = {
        'format': FORMAT_VERSION, 'zim_uuid': uuid, 'zim_file': zim.name, 'model': HF_MODEL,
        'ollama_model': OLLAMA_MODEL, 'dim': args.dim, 'scope': args.scope, 'chunk_tokens': DEFAULT_CHUNK_TOKENS,
        'max_embed_chars': MAX_EMBED_CHARS, 'block_size': args.block_size,
    }
    meta_path = out / META_FILE
    shard_dir, scope = SHARD_DIR, args.scope
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        changed = {k: (meta.get(k), v) for k, v in params.items() if meta.get(k) != v}
        upgrade = (meta.get('complete') and args.scope == 'all' and set(changed) == {'scope'}
                   and meta.get('scope') == 'leads')
        if upgrade:
            shard_dir, scope = REST_SHARD_DIR, 'rest'
            if meta.get('upgrade_to') != 'all':
                meta['upgrade_to'] = 'all'
                write_meta(out, meta)
                print(f"⬆ Upgrading the leads index to all sections; it stays usable until the upgrade finishes")
        elif changed:
            details = ', '.join(f'{k}: {old!r} -> {new!r}' for k, (old, new) in changed.items())
            hint = (" Finish the leads build first; --scope all then upgrades it."
                    if set(changed) == {'scope'} and meta.get('scope') == 'leads' else "")
            print(f"❌ {out} holds an index built with other settings ({details}).{hint} "
                  "Use another --out, or delete that directory to start over.")
            return 1
        elif meta.get('complete'):
            print(f"✓ Dense index already built: {out} ({meta.get('count', 0):,} vectors)")
            return 0
    else:
        meta = dict(params, complete=False)
        out.mkdir(parents=True, exist_ok=True)
        write_meta(out, meta)
    (out / shard_dir).mkdir(parents=True, exist_ok=True)

    starts = range(0, store.archive.all_entry_count, args.block_size)
    pending = [s for s in starts if not shard_path(out, s, shard_dir).exists()]
    devices = args.devices.split(',') if args.devices and backend == 'torch' else _default_devices(backend)
    print(f"📚 {zim.name}: {store.archive.all_entry_count:,} entries in {len(starts)} blocks, "
          f"{len(starts) - len(pending)} already done")
    print(f"🧮 {backend} on {', '.join(devices)}; scope {scope}, {args.dim} dimensions → {out}")
    if backend == 'ollama':
        print(f"   (Ollama embeds ~60-70 chunks/s; the full English Wikipedia has ~31M chunks, ~7M lead chunks)")

    selected = pending[:args.max_blocks] if args.max_blocks is not None else pending
    settings = {'zim': str(zim), 'out': str(out), 'block_size': args.block_size, 'scope': scope, 'shard_dir': shard_dir,
                'dim': args.dim, 'chunk_tokens': DEFAULT_CHUNK_TOKENS, 'batch_tokens': args.batch_tokens}
    if selected and not embed_blocks(settings, selected, devices):
        print("❌ Some blocks failed; rerun the same command to retry them")
        return 1
    if len(selected) < len(pending):
        print(f"⏸ Stopped after {len(selected)} block(s) (--max-blocks); rerun without it to continue")
        return 0

    print("🕸 Building the search graph...")
    count = finalize(out, meta, args.keep_shards, threads=os.cpu_count() or 4, shard_dir=shard_dir)
    size = (out / INDEX_FILE).stat().st_size
    print(f"✓ Dense index built: {count:,} vectors, {size / 1e9:.1f} GB in {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
