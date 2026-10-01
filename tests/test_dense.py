"""Tests for the dense index (retrieval/dense.py, retrieval/build_dense.py) and hybrid retrieval"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.build_dense import block_chunks, finalize, write_meta, write_shard, REST_SHARD_DIR, SHARD_DIR
from retrieval.dense import (
    DenseHit, DenseIndex, META_FILE, embed_text, pack_key, quantize, text_check, unpack_key, MAX_EMBED_CHARS,
)
from tests.fakes import FakeStore, _article


class TestKeysAndVectors:
    def test_key_roundtrip(self):
        key = pack_key(19_707_095, 4321, 'Moon > Exploration\nApollo 11 landed in 1969.')
        assert unpack_key(key) == (19_707_095, 4321, text_check('Moon > Exploration\nApollo 11 landed in 1969.'))
        assert key < 2 ** 64

    def test_quantize_keeps_direction(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((50, 1024)).astype(np.float32)
        q = quantize(v, 512)
        assert q.dtype == np.int8 and q.shape == (50, 512)
        assert np.abs(q).max(axis=1).tolist() == [127] * 50
        a, b = v[:, :512], q.astype(np.float32)
        cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        assert cos.min() > 0.999

    def test_quantize_zero_vector(self):
        assert not quantize(np.zeros((1, 8)), 8).any()

    def test_embed_text_is_cut(self):
        assert len(embed_text('x' * 5000)) == MAX_EMBED_CHARS


@pytest.fixture
def store():
    articles = {
        'Apollo_11': _article(
            'Apollo 11',
            ('', 'Apollo 11 was the spaceflight that first landed humans on the Moon.'),
            ('Lunar surface operations', 'Armstrong stepped onto the lunar surface six hours later.'),
            ('Splashdown', 'The crew returned to Earth in the Pacific Ocean.'),
        ),
        'A_Walk_on_the_Moon': _article(
            'A Walk on the Moon', ('', 'A Walk on the Moon is a 1999 film about the first moon walk summer.')),
        'Neil_Armstrong': _article(
            'Neil Armstrong', ('', 'Neil Armstrong was an American astronaut and aeronautical engineer.'),
            ('Early life', 'He was born in Wapakoneta, Ohio.')),
        'Moon_(disambiguation)': _article('Moon (disambiguation)', ('', 'Moon may refer to:'), disambig=True),
        'List_of_moonwalkers': _article('List of moonwalkers', ('', 'Twelve people have walked on the Moon.')),
    }
    return FakeStore(articles, {}, ['A_Walk_on_the_Moon'], dfs={'first': 3_000_000, 'walk': 50_000, 'moon': 200_000})


class FakeDense:
    """Stands in for DenseIndex: fixed hits per query, given as chunk ids"""

    def __init__(self, store, results, stale=()):
        self.meta = {'zim_uuid': 'fake-uuid'}
        self.chunk_tokens = 1200
        self.directory = 'fake'
        paths = [store.archive._get_entry_by_id(i).path for i in range(store.archive.all_entry_count)]
        self._hits = {}
        for query, chunk_ids in results.items():
            hits = []
            for rank, chunk_id in enumerate(chunk_ids):
                path, _, ordinal = chunk_id.rpartition('#')
                check = text_check(store.get_chunk(chunk_id).text) ^ (1 if chunk_id in stale else 0)
                hits.append(DenseHit(paths.index(path), int(ordinal), check, 0.9 - rank / 100))
            self._hits[query] = hits

    def search(self, query, k):
        return self._hits.get(query, [])[:k]


class TestHybridRetrieve:
    QUESTION = 'Who was the first person to walk on the moon?'

    def test_dense_hits_without_query_terms_are_included(self, store):
        store.attach_dense(FakeDense(store, {self.QUESTION: ['Neil_Armstrong#0', 'Apollo_11#1']}))
        result = store.retrieve(self.QUESTION, k=4)
        labels = [sc.chunk.label for sc in result.chunks]
        assert 'Neil Armstrong' in labels and 'Apollo 11 > Lunar surface operations' in labels
        reasons = {sc.chunk.label: sc.reason for sc in result.chunks}
        assert reasons['Neil Armstrong'] == 'dense'
        assert result.dense == ['Neil_Armstrong#0', 'Apollo_11#1']
        assert 'dense' in result.timings

    def test_dense_and_bm25_agreement_ranks_first(self, store):
        store.attach_dense(FakeDense(store, {self.QUESTION: ['Apollo_11#0', 'Neil_Armstrong#0']}))
        result = store.retrieve(self.QUESTION, k=3)
        assert result.chunks[0].chunk.chunk_id == 'Apollo_11#0'

    def test_fulltext_only_without_dense(self, store):
        result = store.retrieve(self.QUESTION, k=4)
        assert all(sc.chunk.path != 'Neil_Armstrong' for sc in result.chunks)
        assert result.dense == []

    def test_stale_hits_dropped(self, store, capsys):
        store.attach_dense(FakeDense(store, {self.QUESTION: ['Neil_Armstrong#0', 'Apollo_11#1', 'Apollo_11#2']},
                                     stale={'Neil_Armstrong#0', 'Apollo_11#2'}))
        result = store.retrieve(self.QUESTION, k=4)
        assert result.dense == ['Apollo_11#1']
        assert 'rebuild the dense index' in capsys.readouterr().out

    def test_attach_checks_zim_and_chunk_size(self, store):
        other = FakeDense(store, {})
        other.meta = {'zim_uuid': 'another'}
        with pytest.raises(ValueError, match='another ZIM'):
            store.attach_dense(other)
        bigger = FakeDense(store, {})
        bigger.chunk_tokens = 600
        with pytest.raises(ValueError, match='600-token'):
            store.attach_dense(bigger)


class TestBuilder:
    def test_block_chunks_skips_disambiguation_and_lists(self, store):
        keys, texts = block_chunks(store, 0, 100, 'all')
        assert [unpack_key(k)[:2] for k in keys] == [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)]
        assert texts[1].startswith('Apollo 11 > Lunar surface operations\n')
        assert unpack_key(keys[1])[2] == text_check(store.get_chunk('Apollo_11#1').text)

    def test_block_chunks_leads_scope_and_range(self, store):
        keys, _ = block_chunks(store, 1, 3, 'leads')
        assert [unpack_key(k)[:2] for k in keys] == [(1, 0), (2, 0)]

    def test_block_chunks_rest_is_all_minus_leads(self, store):
        every, _ = block_chunks(store, 0, 100, 'all')
        leads, _ = block_chunks(store, 0, 100, 'leads')
        rest, _ = block_chunks(store, 0, 100, 'rest')
        assert sorted(leads + rest) == sorted(every)
        assert not set(leads) & set(rest)

    def test_upgrade_adds_rest_to_leads_index(self, store, tmp_path):
        """A finished leads index plus shards-rest/ -> the same keys as an index built with scope all"""
        dim = 32
        rng = np.random.default_rng(2)
        leads, _ = block_chunks(store, 0, 100, 'leads')
        rest, _ = block_chunks(store, 0, 100, 'rest')
        (tmp_path / SHARD_DIR).mkdir()
        write_shard(tmp_path, 0, leads, quantize(rng.standard_normal((len(leads), 64)), dim))
        meta = {'zim_uuid': 'fake-uuid', 'dim': dim, 'chunk_tokens': 1200, 'scope': 'leads', 'complete': False}
        write_meta(tmp_path, meta)
        assert finalize(tmp_path, meta, keep_shards=False, threads=2) == len(leads)

        meta['upgrade_to'] = 'all'
        (tmp_path / REST_SHARD_DIR).mkdir()
        write_shard(tmp_path, 0, rest, quantize(rng.standard_normal((len(rest), 64)), dim), REST_SHARD_DIR)
        assert DenseIndex.find('fake-uuid', tmp_path) is not None      # still usable while upgrading
        assert finalize(tmp_path, meta, keep_shards=False, threads=2, shard_dir=REST_SHARD_DIR) == len(leads + rest)
        saved = json.loads((tmp_path / META_FILE).read_text())
        assert saved['scope'] == 'all' and 'upgrade_to' not in saved and saved['count'] == len(leads + rest)
        assert not (tmp_path / REST_SHARD_DIR).exists()
        index = DenseIndex(tmp_path, embedder=SimpleNamespace(embed=None))
        assert sorted(int(k) for k in index.index.keys) == sorted(leads + rest)

    def test_finalize_and_search(self, store, tmp_path):
        """Shards -> HNSW graph -> DenseIndex.search with a fake query embedder"""
        dim = 32
        rng = np.random.default_rng(1)
        keys, _ = block_chunks(store, 0, 100, 'all')
        vectors = rng.standard_normal((len(keys), 64)).astype(np.float32)
        (tmp_path / SHARD_DIR).mkdir()
        write_shard(tmp_path, 0, keys[:4], quantize(vectors[:4], dim))
        write_shard(tmp_path, 4, keys[4:], quantize(vectors[4:], dim))
        write_shard(tmp_path, 8, [], np.zeros((0, dim), np.int8))
        meta = {'zim_uuid': 'fake-uuid', 'dim': dim, 'chunk_tokens': 1200, 'complete': False}
        write_meta(tmp_path, meta)
        assert DenseIndex.find('fake-uuid', tmp_path) is None     # not finished yet

        assert finalize(tmp_path, meta, keep_shards=False, threads=2) == len(keys)
        assert not (tmp_path / SHARD_DIR).exists()
        assert json.loads((tmp_path / META_FILE).read_text())['complete'] is True
        assert DenseIndex.find('other-uuid', tmp_path) is None

        target = 4      # Neil Armstrong lead
        embedder = SimpleNamespace(embed=lambda texts: vectors[target:target + 1] + 0.01)
        index = DenseIndex(tmp_path, embedder=embedder)
        assert len(index) == len(keys)
        hits = index.search('who walked on the moon', 3)
        assert (hits[0].entry_index, hits[0].ordinal) == unpack_key(keys[target])[:2]
        assert hits[0].score > 0.95

        store.attach_dense(index)
        result = store.retrieve('Who was the first person to walk on the moon?', k=4)
        assert result.dense[0] == 'Neil_Armstrong#0'


# ---------------------------------------------------------------- real index (built with retrieval.build_dense)

@pytest.mark.integration
class TestDevZimDense:
    """Needs the dev ZIM, its dense index in the default location, and Ollama with the embedding model"""

    @pytest.fixture(scope='class')
    def store(self, tmp_path_factory):
        from retrieval import ZimStore
        from tests.test_zim_store import DEV_ZIM
        if not DEV_ZIM.is_file():
            pytest.skip(f'dev ZIM not found: {DEV_ZIM}')
        store = ZimStore(DEV_ZIM, cache_dir=tmp_path_factory.mktemp('cache'))
        index = DenseIndex.find(str(store.archive.uuid))
        if index is None:
            pytest.skip('no dense index for the dev ZIM (uv run --group dense-build python -m retrieval.build_dense '
                        f'--zim {DEV_ZIM})')
        try:
            index.search('test', 1)
        except Exception as e:
            pytest.skip(f'cannot embed queries with Ollama: {e}')
        store.attach_dense(index)
        return store

    def test_semantic_match_without_shared_words(self, store):
        """'space flights' + animals: the Turtle section on space flights shares no word with 'creatures orbit'"""
        result = store.retrieve('Which creatures have been sent into orbit?', k=10)
        assert any(cid.startswith(('Turtle#', 'Frog#', 'Rodent#')) for cid in result.dense[:10])

    def test_hits_resolve_to_current_chunks(self, store):
        result = store.retrieve('history of the Roman Empire', k=8)
        assert len(result.dense) >= 20      # checksums match, nothing dropped as stale
