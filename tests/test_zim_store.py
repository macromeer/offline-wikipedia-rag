"""Tests for the libzim retrieval layer (retrieval/zim_store.py)"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import (
    BM25, Section, ZimStore, article_to_sections, chunk_sections,
    disambiguation_links, is_disambiguation, query_terms, title_spans, tokenize,
)
from retrieval.zim_store import TitleHit, reciprocal_rank_fusion
from tests.fakes import FakeStore, _article

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture(scope='module')
def article_html():
    return (FIXTURES / 'article_parsoid.html').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def sections(article_html):
    return article_to_sections(article_html, 'Glacier Lake')


@pytest.fixture(scope='module')
def disambiguation_html():
    return (FIXTURES / 'disambiguation.html').read_text(encoding='utf-8')


class TestSectionSplitting:
    def test_lead_first_then_headings_in_order(self, sections):
        assert [s.heading_path for s in sections] == [
            (),
            ('Geography',),
            ('Geography', 'Climate (1991–2020)'),
            ('History',),
            ('Culture', 'In literature'),
        ]

    def test_labels_and_anchors(self, sections):
        assert sections[0].label == 'Glacier Lake'
        assert sections[2].label == 'Glacier Lake > Geography > Climate (1991–2020)'
        assert sections[2].anchor == 'Climate_(1991–2020)'
        assert sections[0].anchor is None

    def test_reference_sections_and_their_subsections_skipped(self, sections):
        text = '\n'.join(s.text for s in sections)
        for gone in ('List of lakes', 'Survey of Glacier Lake', 'Jones, B.', 'Official website'):
            assert gone not in text

    def test_noise_removed(self, sections):
        text = '\n'.join(s.text for s in sections)
        for gone in ('For the song', 'Area 12', 'float:right', 'ɡleɪʃər', 'The lake in winter',
                     'Old photograph', 'Pine Lake', '47°N', 'issued from Wikipedia', 'edit'):
            assert gone not in text, gone

    def test_citation_markers_stripped(self, sections):
        lead = sections[0].text
        assert '[1]' not in lead and '[2]' not in lead and 'citation needed' not in lead
        assert lead.startswith('Glacier Lake is a proglacial lake in the Northern Range. It formed')
        assert 'feeds the Silver River.' in lead

    def test_lead_paragraphs_kept_as_blocks(self, sections):
        assert sections[0].text.count('\n\n') == 1

    def test_math_uses_alttext(self, sections):
        assert r'A=\pi r^{2}' in sections[1].text

    def test_wikitable_rows(self, sections):
        assert 'January | −14 °C' in sections[2].text

    def test_nested_lists(self, sections):
        assert '- 1902: first survey\n  - led by A. Smith\n- 1950: dam proposal rejected' in sections[3].text

    def test_heading_without_own_text_is_still_in_path(self, sections):
        assert not any(s.heading_path == ('Culture',) for s in sections)
        assert sections[4].text == 'The lake appears in the novel Cold Water.'

    def test_flat_layout(self):
        html = """<html><body><div class="mw-parser-output">
            <p>Lead text here.</p>
            <h2 id="A">Alpha</h2><p>Alpha text.</p>
            <h3 id="A1">Alpha one</h3><p>Alpha one text.</p>
            <h2 id="Notes">Notes</h2><p>a note</p>
            <h2 id="B">Beta</h2><p>Beta text.</p>
        </div></body></html>"""
        result = [(s.heading_path, s.text) for s in article_to_sections(html, 'T')]
        assert result == [
            ((), 'Lead text here.'),
            (('Alpha',), 'Alpha text.'),
            (('Alpha', 'Alpha one'), 'Alpha one text.'),
            (('Beta',), 'Beta text.'),
        ]


class TestDisambiguation:
    def test_detected_from_lead(self, disambiguation_html):
        secs = article_to_sections(disambiguation_html, 'GLX')
        assert is_disambiguation(secs, 'GLX', disambiguation_html)

    def test_regular_article_is_not(self, sections, article_html):
        assert not is_disambiguation(sections, 'Glacier Lake', article_html)

    def test_detected_from_title(self):
        assert is_disambiguation([], 'Mercury (disambiguation)')

    def test_links_first_per_item_internal_only(self, disambiguation_html):
        assert disambiguation_links(disambiguation_html, 'GLX') == [
            'Glacier_Lake', 'Galactic_X-ray_source', 'Record_label',
        ]

    def test_links_relative_to_nested_path(self):
        html = '<ul><li><a href="../B_(x)">b</a></li><li><a href="C%20D">c</a></li></ul>'
        assert disambiguation_links(html, 'A/B') == ['B_(x)', 'A/C D']


class TestChunking:
    def test_one_chunk_per_small_section_with_label(self, sections):
        chunks = chunk_sections(sections, 'Glacier_Lake')
        assert len(chunks) == len(sections)
        assert chunks[0].text.startswith('Glacier Lake\nGlacier Lake is')
        assert chunks[2].text.startswith('Glacier Lake > Geography > Climate (1991–2020)\nWinters')
        assert [c.chunk_id for c in chunks] == [f'Glacier_Lake#{i}' for i in range(len(chunks))]
        assert chunks[0].is_lead and not chunks[1].is_lead
        assert chunks[2].anchor == 'Climate_(1991–2020)'

    def test_long_section_split_at_block_boundaries(self):
        paras = [f'Paragraph {i}. ' + 'word ' * 150 for i in range(6)]
        section = Section('T', ('H',), '\n\n'.join(paras))
        chunks = chunk_sections([section], 'T', max_tokens=500)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.text.startswith('T > H\n')
            assert len(chunk.text) <= 500 * 4
        body = '\n\n'.join(c.text.split('\n', 1)[1] for c in chunks)
        assert body == section.text

    def test_oversize_paragraph_split_at_sentences(self):
        sentence = 'This sentence has some words in it. '
        section = Section('T', (), (sentence * 200).strip())
        chunks = chunk_sections([section], 'T', max_tokens=300)
        assert len(chunks) > 1
        assert all(len(c.text) <= 300 * 4 for c in chunks)
        assert all(c.text.split('\n', 1)[1].endswith('in it.') for c in chunks)

    def test_sections_never_merged(self):
        secs = [Section('T', (), 'a'), Section('T', ('B',), 'b')]
        assert [c.text for c in chunk_sections(secs, 'T')] == ['T\na', 'T > B\nb']


class TestQueryProcessing:
    def test_query_terms(self):
        assert query_terms("What are the goals of NASA's Artemis program?") == ['goals', 'NASA', 'Artemis', 'program']
        assert query_terms('Tell me about earthquakes') == ['earthquakes']

    def test_tokenize_folds_plurals_and_possessives(self):
        assert tokenize("Earthquakes, NASA's goals, countries; boxes; glass") == [
            'earthquake', 'nasa', 'goal', 'country', 'box', 'glass']

    def test_title_spans_allow_capitalised_article(self):
        spans = [text for _, _, text in title_spans('Is The Expanse a good show?')]
        assert 'The Expanse' in spans and 'Expanse' in spans
        assert 'Is The Expanse' not in spans  # sentence-initial stopword cannot start a span

    def test_title_spans_do_not_cross_clauses(self):
        spans = [text for _, _, text in title_spans('Paris, France')]
        assert 'Paris France' not in spans and 'Paris' in spans and 'France' in spans

    def test_title_spans_keep_inner_stopwords(self):
        spans = [text for _, _, text in title_spans('who wrote the lord of the rings')]
        assert 'lord of the rings' in spans


class TestRanking:
    def test_bm25_prefers_matching_doc(self):
        corpus = [tokenize('the cat sat on the mat'), tokenize('dogs chase cats'), tokenize('nothing here')]
        scores = BM25(corpus).scores(tokenize('cat'))
        assert scores[0] > 0 and scores[1] > 0 and scores[2] == 0

    def test_rrf(self):
        assert reciprocal_rank_fusion([['a', 'b', 'c'], ['b', 'a'], ['b']]) == ['b', 'a', 'c']


class TestTitleHits:
    @pytest.fixture
    def store(self):
        articles = {
            'NASA': _article('NASA', ('', 'NASA is the space agency.')),
            'Goal': _article('Goal', ('', 'A goal is an objective.')),
            'First_person': _article('First person', ('', 'Grammatical person.')),
            'Canberra': _article('Canberra', ('', 'Canberra is the capital city of Australia.')),
            'Earthquake': _article('Earthquake', ('', 'An earthquake is shaking.')),
        }
        titles = {'NASA': 'NASA', 'Goals': 'Goal', 'First person': 'First_person',
                  'Capital of australia': 'Canberra', 'Earthquakes': 'Earthquake'}
        return FakeStore(articles, titles, [])

    def test_capitalised_beats_common_word(self, store):
        hits = store.find_title_hits('What are the goals of NASA?')
        assert [(h.path, h.strong) for h in hits] == [('NASA', True)]

    def test_span_covering_question_is_strong(self, store):
        hits = store.find_title_hits('what is the capital of australia')
        assert [(h.path, h.strong) for h in hits] == [('Canberra', True)]

    def test_single_topic_word_is_strong(self, store):
        assert store.find_title_hits('Tell me about earthquakes') == [
            TitleHit('Earthquake', 'Earthquake', 'earthquakes', True)]

    def test_lowercase_phrase_inside_longer_question_is_weak(self, store):
        hits = store.find_title_hits('Who was the first person to walk on the moon?')
        assert [(h.path, h.strong) for h in hits] == [('First_person', False)]


class TestRetrieve:
    @pytest.fixture
    def store(self):
        articles = {
            'NASA': _article(
                'NASA',
                ('', 'NASA is the United States space agency.'),
                ('Strategic plan', 'NASA goals: expand knowledge, explore the Moon and Mars.'),
                ('History', 'NASA was founded in 1958.'),
                ('Budget', 'NASA budget figures.'),
                ('Programs', 'NASA programs pursue goals in science.'),
            ),
            'Nasa_(footballer)': _article(
                'Nasa (footballer)',
                ('', 'Nasa is a footballer.'),
                ('Career', 'Nasa scored goals goals goals for the club.'),
            ),
            'ETF': _article('ETF', ('', 'ETF may refer to:'), disambig=True,
                            links=['Exchange-traded_fund', 'Escape_the_Fate']),
            'Exchange-traded_fund': _article(
                'Exchange-traded fund', ('', 'An exchange-traded fund (ETF) is an investment fund.'),
                ('Structure', 'ETF shares trade on exchanges.')),
            'Escape_the_Fate': _article('Escape the Fate', ('', 'Escape the Fate is a rock band.')),
        }
        titles = {'NASA': 'NASA', 'ETF': 'ETF'}
        return FakeStore(articles, titles, ['Nasa_(footballer)', 'NASA', 'Exchange-traded_fund'],
                         dfs={'nasa': 5, 'goals': 50, 'etf': 20})

    def test_title_hit_lead_first_and_article_preferred(self, store):
        result = store.retrieve('What are the goals of NASA?', k=4)
        labels = [sc.chunk.label for sc in result.chunks]
        assert labels[0] == 'NASA' and result.chunks[0].reason == 'title'
        assert 'NASA > Strategic plan' in labels[:3]
        assert result.title_hits == [('NASA', 'NASA')]

    def test_per_article_cap(self, store):
        result = store.retrieve('What are the goals of NASA?', k=10, per_article=1)
        paths = [sc.chunk.path for sc in result.chunks]
        assert paths.count('NASA') <= 2 and paths.count('Nasa_(footballer)') <= 1

    def test_disambiguation_resolved_via_fulltext(self, store):
        result = store.retrieve('What is an ETF?', k=3)
        assert result.chunks[0].chunk.label == 'Exchange-traded fund'
        assert result.chunks[0].reason == 'title'
        assert all(sc.chunk.path != 'ETF' for sc in result.chunks)

    def test_get_chunk_roundtrip(self, store):
        chunk = store.get_chunk('NASA#1')
        assert chunk.label == 'NASA > Strategic plan'
        assert store.get_chunk('NASA#99') is None
        assert store.get_chunk('Missing#0') is None


# ---------------------------------------------------------------- real ZIM files

DEV_ZIM = Path(os.environ.get('WIKI_DEV_ZIM', Path.home() / 'wikipedia-dev' / 'wikipedia_en_100_2026-08.zim'))


def _full_zim():
    from wikirag.zimfiles import resolve_zim_path
    try:
        path = resolve_zim_path()
    except (FileNotFoundError, ValueError):
        return None
    return path if path and '_all_' in path.name else None


@pytest.mark.integration
class TestDevZim:
    """Small ZIM (top-100 articles, has a full-text index); set WIKI_DEV_ZIM to override"""

    @pytest.fixture(scope='class')
    @classmethod
    def store(cls, tmp_path_factory):
        if not DEV_ZIM.is_file():
            pytest.skip(f'dev ZIM not found: {DEV_ZIM}')
        return ZimStore(DEV_ZIM, cache_dir=tmp_path_factory.mktemp('cache'))

    def test_metadata(self, store):
        assert store.has_fulltext
        assert store.date

    def test_sections_of_real_article(self, store):
        secs = store.get_sections('Antarctica')
        headings = [s.heading_path for s in secs]
        assert headings[0] == ()
        assert ('Climate',) in headings
        assert not any(h and h[0] in ('References', 'External links') for h in headings)
        assert not any('[1]' in s.text for s in secs)

    def test_retrieve(self, store):
        result = store.retrieve('What is the climate of Antarctica like?', k=6)
        assert result.chunks[0].chunk.label == 'Antarctica'
        assert any(sc.chunk.heading_path[:1] == ('Climate',) for sc in result.chunks)


@pytest.mark.integration
class TestFullZim:
    """Full English Wikipedia (WIKI_ZIM or auto-discovered *_all_*.zim)"""

    @pytest.fixture(scope='class')
    @classmethod
    def store(cls):
        path = _full_zim()
        if not path:
            pytest.skip('no full English Wikipedia ZIM found')
        return ZimStore(path)

    def test_nasa_goals(self, store):
        result = store.retrieve('What are the goals of NASA?')
        nasa = [sc.chunk for sc in result.chunks if sc.chunk.title == 'NASA']
        assert nasa and nasa[0].is_lead and result.chunks[0].chunk.title == 'NASA'
        assert any(any(w in ' '.join(c.heading_path).lower() for w in ('plan', 'program', 'mission', 'goal'))
                   for c in nasa[1:])
        assert not any(sc.chunk.title in ('Goal', 'Goals') for sc in result.chunks)

    def test_abbreviation_through_disambiguation(self, store):
        result = store.retrieve('What is an ETF?')
        assert result.chunks[0].chunk.title == 'Exchange-traded fund'

    def test_redirect_title(self, store):
        result = store.retrieve('what is the capital of australia')
        assert result.chunks[0].chunk.title == 'Canberra'
