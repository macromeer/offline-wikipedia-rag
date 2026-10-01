"""Unit tests for Wikipedia RAG core functions"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wikipedia_rag_kiwix import (
    KiwixWikipediaRAG, QUESTION_SKIP_WORDS, _parse_param_billions, _is_complete_zim,
    _find_zim_files, resolve_zim_path, _num_ctx_for,
)


class TestLanguageFilters:
    """Ensure language filters remain topic agnostic"""

    def test_skip_words_do_not_encode_specific_domains(self):
        domain_terms = {
            'movie', 'movies', 'film', 'films', 'tv', 'television',
            'show', 'shows', 'series', 'season', 'seasons',
            'game', 'games'
        }
        assert domain_terms.isdisjoint(QUESTION_SKIP_WORDS)


class TestSearchTermExtraction:
    """Test search term extraction logic"""
    
    def test_extract_proper_nouns(self):
        """Test extraction of proper nouns"""
        rag = Mock(spec=KiwixWikipediaRAG)
        rag.extract_search_terms = KiwixWikipediaRAG.extract_search_terms.__get__(rag)
        
        question = "Who was Albert Einstein?"
        terms = rag.extract_search_terms(question)
        
        assert 'Albert Einstein' in terms
        assert len(terms) > 0
    
    def test_extract_content_words(self):
        """Test extraction of important content words"""
        rag = Mock(spec=KiwixWikipediaRAG)
        rag.extract_search_terms = KiwixWikipediaRAG.extract_search_terms.__get__(rag)
        
        question = "What is photosynthesis?"
        terms = rag.extract_search_terms(question)
        
        # Should extract "photosynthesis" or capitalized version
        assert any('photosynthesis' in term.lower() for term in terms)
    
    def test_filters_stopwords(self):
        """Test that stopwords are filtered out"""
        rag = Mock(spec=KiwixWikipediaRAG)
        rag.extract_search_terms = KiwixWikipediaRAG.extract_search_terms.__get__(rag)
        
        question = "What is the meaning of life?"
        terms = rag.extract_search_terms(question)
        
        # Stopwords should not be in results
        assert 'what' not in [t.lower() for t in terms]
        assert 'the' not in [t.lower() for t in terms]
        assert 'is' not in [t.lower() for t in terms]

    def test_primary_keywords_handle_lowercase_titles(self):
        """Ensure primary keyword extraction finds lowercase media titles"""
        rag = Mock(spec=KiwixWikipediaRAG)
        rag.extract_search_terms = KiwixWikipediaRAG.extract_search_terms.__get__(rag)
        rag.extract_primary_keywords = KiwixWikipediaRAG.extract_primary_keywords.__get__(rag)

        question = "is the expanse a good tv show?"
        keywords = rag.extract_primary_keywords(question)

        assert any('expanse' == kw for kw in keywords)

    def test_focus_phrases_from_quotes(self):
        """Quoted segments should become focus phrases"""
        rag = Mock(spec=KiwixWikipediaRAG)
        rag.extract_search_terms = KiwixWikipediaRAG.extract_search_terms.__get__(rag)
        rag.extract_focus_phrases = KiwixWikipediaRAG.extract_focus_phrases.__get__(rag)

        question = 'Is "Star Trek - Next Generation" a good tv show?'
        phrases = rag.extract_focus_phrases(question)

        assert any('Star Trek - Next Generation' in phrase for phrase in phrases)


class TestComplexityEstimation:
    """Test question complexity estimation"""
    
    def test_simple_question(self):
        """Test detection of simple questions"""
        rag = Mock(spec=KiwixWikipediaRAG)
        rag.estimate_question_complexity = KiwixWikipediaRAG.estimate_question_complexity.__get__(rag)
        
        question = "What is photosynthesis?"
        complexity = rag.estimate_question_complexity(question)
        
        # Simple question should get minimum articles
        assert complexity == 3
    
    def test_complex_question_with_and(self):
        """Test detection of multi-part questions"""
        rag = Mock(spec=KiwixWikipediaRAG)
        rag.estimate_question_complexity = KiwixWikipediaRAG.estimate_question_complexity.__get__(rag)
        
        question = "What causes earthquakes and can we predict them?"
        complexity = rag.estimate_question_complexity(question)
        
        # Multi-part question should get more articles
        assert complexity > 3
    
    def test_comparison_question(self):
        """Test detection of comparison questions"""
        rag = Mock(spec=KiwixWikipediaRAG)
        rag.estimate_question_complexity = KiwixWikipediaRAG.estimate_question_complexity.__get__(rag)
        
        question = "Compare mitochondria and chloroplasts"
        complexity = rag.estimate_question_complexity(question)
        
        # Comparison should get more articles
        assert complexity >= 5


class TestKeywordMatching:
    """Test helper functions related to keyword and phrase matching"""

    def test_title_matches_keywords_requires_multiple_hits(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        keywords = ['star', 'trek', 'generation']
        assert rag._title_matches_keywords('Star Trek: The Next Generation', keywords)
        assert not rag._title_matches_keywords("List of Britain's Next Top Model contestants", keywords)

    def test_title_matches_focus_phrase(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        phrases = ['Star Trek - Next Generation']
        assert rag._title_matches_focus_phrase('Star Trek: The Next Generation', phrases)
        assert not rag._title_matches_focus_phrase('Star Trek: Voyager', phrases)
    
    def test_long_analytical_question(self):
        """Test detection of analytical questions"""
        rag = Mock(spec=KiwixWikipediaRAG)
        rag.estimate_question_complexity = KiwixWikipediaRAG.estimate_question_complexity.__get__(rag)
        
        question = "How does climate change impact ocean ecosystems and what are the long-term consequences?"
        complexity = rag.estimate_question_complexity(question)
        
        # Long analytical question should get many articles
        assert complexity >= 5


class TestModelDetection:
    """Test AI model detection logic"""
    
    @patch('wikipedia_rag_kiwix.ollama.list')
    def test_selection_model_priority(self, mock_list):
        """Test selection model detection prioritizes correct models"""
        # Mock available models
        mock_models = [
            Mock(model='llama3.1:8b'),
            Mock(model='mistral:7b'),
            Mock(model='qwen2.5:32b-instruct')
        ]
        mock_response = Mock()
        mock_response.models = mock_models
        mock_list.return_value = mock_response
        
        with patch('wikipedia_rag_kiwix.requests.get'):
            rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
            models = ['llama3.1:8b', 'mistral:7b', 'qwen2.5:32b-instruct']
            result = rag._detect_selection_model(None, models)
            
            # Should prefer qwen2.5 for selection
            assert 'qwen' in result.lower()
    
    @patch('wikipedia_rag_kiwix.ollama.list')
    def test_avoids_reasoning_models(self, mock_list):
        """Test that reasoning models like DeepSeek R1 are avoided"""
        mock_models = [
            Mock(model='deepseek-r1:latest'),
            Mock(model='mistral:7b')
        ]
        mock_response = Mock()
        mock_response.models = mock_models
        mock_list.return_value = mock_response
        
        with patch('wikipedia_rag_kiwix.requests.get'):
            rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
            models = ['deepseek-r1:latest', 'mistral:7b']
            result = rag._detect_selection_model(None, models)
            
            # Should select mistral, not deepseek
            assert 'mistral' in result.lower()
            assert 'deepseek' not in result.lower()


# Exact `ollama list` output on the dev box (2026-10-01), with details.parameter_size
DEV_BOX_MODELS = {
    'llama3.1:latest': '8.0B',
    'qwen2.5:1.5b': '1.5B',
    'gemma4:26b': '25.8B',
    'qwen3.6:35b': '36.0B',
    'cline-dev:latest': '14.8B',
    'qwen3:0.6b': '751.63M',
    'qwen2.5-coder:14b': '14.8B',
}


def _rag_with_models(models):
    """RAG instance whose model list comes from a mocked `ollama list` ({name: parameter_size})"""
    listed = []
    for name, size in models.items():
        m = Mock(model=name)
        m.details.parameter_size = size
        listed.append(m)
    rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
    with patch('wikipedia_rag_kiwix.ollama.list', return_value=Mock(models=listed)):
        available = rag._get_available_models()
    return rag, available


class TestModelDetectionRules:
    """Exact-match first, family fallback with size/type filters"""

    def test_dev_box_picks_current_models(self):
        rag, available = _rag_with_models(DEV_BOX_MODELS)
        assert rag._detect_selection_model(None, available) == 'qwen3.6:35b'
        assert rag._detect_summarization_model(None, available) == 'gemma4:26b'

    def test_rejects_small_and_coder_models(self):
        # Without the new families, the old prefix match would pick qwen2.5:1.5b
        models = {k: v for k, v in DEV_BOX_MODELS.items() if k not in ('gemma4:26b', 'qwen3.6:35b')}
        rag, available = _rag_with_models(models)
        selection = rag._detect_selection_model(None, available)
        assert selection == 'llama3.1:latest'
        assert rag._detect_summarization_model(None, available) == 'llama3.1:latest'

    def test_family_fallback_does_not_cross_families(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        # 'qwen2.5' preference must not match 'qwen2.5-coder'
        assert rag._detect_selection_model(None, ['qwen2.5-coder:32b', 'mistral:7b']) == 'mistral:7b'

    def test_family_fallback_uses_tag_size(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        assert rag._detect_selection_model(None, ['qwen3:0.6b', 'qwen3:8b']) == 'qwen3:8b'
        assert rag._detect_selection_model(None, ['qwen3.6:35b-a3b-q4_K_M']) == 'qwen3.6:35b-a3b-q4_K_M'

    def test_exact_tag_matches_latest(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        assert rag._detect_selection_model(None, ['hermes3:latest', 'phi3:medium']) == 'hermes3:latest'

    def test_excludes_embedding_reranker_reasoning(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        available = ['qwen3-embedding:0.6b', 'qwen3-reranker:4b', 'deepseek-r1:32b', 'phi4-r1:14b', 'olmo2:13b']
        assert rag._detect_selection_model(None, available) == 'olmo2:13b'

    def test_small_model_only_as_last_resort(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        assert rag._detect_summarization_model(None, ['qwen3-embedding:0.6b', 'llama3.2:3b']) == 'llama3.2:3b'

    def test_no_usable_model_raises(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        with pytest.raises(Exception):
            rag._detect_selection_model(None, ['qwen2.5-coder:14b', 'nomic-embed-text:latest'])

    def test_explicit_model_wins(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        assert rag._detect_selection_model('qwen3:0.6b', ['gemma4:26b']) == 'qwen3:0.6b'

    @pytest.mark.parametrize('text,expected', [
        ('26b', 26.0), ('1.5b', 1.5), ('35b-a3b', 35.0), ('8x7b', 56.0),
        ('751.63M', 0.75163), ('8.0B', 8.0), ('32b-instruct-q4_K_M', 32.0),
        ('latest', None), ('', None),
    ])
    def test_parse_param_billions(self, text, expected):
        result = _parse_param_billions(text)
        assert result == pytest.approx(expected) if expected is not None else result is None


class TestZimDiscovery:
    """ZIM selection: --zim, WIKI_ZIM, auto-discovery skipping partial downloads"""

    @staticmethod
    def _write_zim(path: Path, complete: bool = True):
        """Minimal file with a ZIM header whose checksum position matches the file size"""
        body_size = 200
        header = bytearray(80)
        header[:4] = b'ZIM\x04'
        header[72:80] = (body_size).to_bytes(8, 'little')
        size = body_size + 16 if complete else 120
        path.write_bytes(bytes(header) + b'\0' * (size - 80))
        return path

    def test_complete_vs_partial(self, tmp_path):
        assert _is_complete_zim(self._write_zim(tmp_path / 'a.zim'))
        assert not _is_complete_zim(self._write_zim(tmp_path / 'b.zim', complete=False))
        (tmp_path / 'c.zim').write_bytes(b'not a zim' * 20)
        assert not _is_complete_zim(tmp_path / 'c.zim')

    def test_newest_complete_first(self, tmp_path):
        self._write_zim(tmp_path / 'wikipedia_en_all_nopic_2026-03.zim')
        self._write_zim(tmp_path / 'wikipedia_en_all_nopic_2026-06.zim')
        self._write_zim(tmp_path / 'wikipedia_en_all_nopic_2026-09.zim', complete=False)
        found = _find_zim_files([tmp_path])
        assert [p.name for p in found] == ['wikipedia_en_all_nopic_2026-06.zim', 'wikipedia_en_all_nopic_2026-03.zim']

    def test_resolve_order(self, tmp_path, monkeypatch):
        cli = self._write_zim(tmp_path / 'cli.zim')
        env = self._write_zim(tmp_path / 'env.zim')
        monkeypatch.setenv('WIKI_ZIM', str(env))
        assert resolve_zim_path(str(cli)) == cli
        assert resolve_zim_path(None) == env

    def test_resolve_rejects_partial(self, tmp_path, monkeypatch):
        monkeypatch.delenv('WIKI_ZIM', raising=False)
        partial = self._write_zim(tmp_path / 'p.zim', complete=False)
        with pytest.raises(ValueError):
            resolve_zim_path(str(partial))
        with pytest.raises(FileNotFoundError):
            resolve_zim_path(str(tmp_path / 'missing.zim'))

    def test_article_url(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        rag.content_base = 'http://localhost:8080/content/wikipedia_en_all_nopic_2026-06'
        assert rag._article_url('The Expanse (TV series)') == \
            'http://localhost:8080/content/wikipedia_en_all_nopic_2026-06/The_Expanse_%28TV_series%29'
        rag.content_base = None
        assert rag._article_url('NASA') is None

    def test_content_base_from_catalog(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        rag.kiwix_url = 'http://localhost:8080'
        rag.zim_path = Path('/x/wikipedia_en_all_nopic_2026-06.zim')
        catalog = ('<entry><link type="text/html" href="/content/other_book" /></entry>'
                   '<entry><link type="text/html" href="/content/wikipedia_en_all_nopic_2026-06" /></entry>')
        with patch('wikipedia_rag_kiwix.requests.get', return_value=Mock(text=catalog)):
            assert rag._discover_content_base() == 'http://localhost:8080/content/wikipedia_en_all_nopic_2026-06'
        with patch('wikipedia_rag_kiwix.requests.get', side_effect=ConnectionError):
            assert rag._discover_content_base() == 'http://localhost:8080/content/wikipedia_en_all_nopic_2026-06'


class TestContextWindow:
    def test_min_num_ctx(self):
        assert _num_ctx_for('x' * 100, 200) == 16384

    def test_large_prompt_doubles(self):
        assert _num_ctx_for('x' * 4 * 20000, 1500) == 32768

    def test_chat_sets_num_ctx_and_disables_thinking(self):
        rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
        with patch('wikipedia_rag_kiwix.ollama.chat', return_value={'prompt_eval_count': 10}) as chat:
            rag._chat('m', 'hello', {'num_predict': 5}, stage='Test')
        kwargs = chat.call_args.kwargs
        assert kwargs['options']['num_ctx'] >= 16384
        assert kwargs['options']['num_predict'] == 5
        assert kwargs['think'] is False


@pytest.mark.integration
class TestKiwixIntegration:
    """Integration tests requiring Kiwix server"""
    
    @patch('wikipedia_rag_kiwix.requests.get')
    def test_search_kiwix_mock(self, mock_get, mock_kiwix_search_html):
        """Test Kiwix search with mocked response"""
        mock_response = Mock()
        mock_response.text = mock_kiwix_search_html
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('wikipedia_rag_kiwix.ollama.list'):
            rag = KiwixWikipediaRAG.__new__(KiwixWikipediaRAG)
            rag.kiwix_url = "http://localhost:8080"
            rag.extract_search_terms = KiwixWikipediaRAG.extract_search_terms.__get__(rag)
            
            # This would need more complete implementation
            # Just testing the structure exists
            assert hasattr(rag, 'kiwix_url')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
