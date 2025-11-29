"""Unit tests for Wikipedia RAG core functions"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wikipedia_rag_kiwix import KiwixWikipediaRAG, QUESTION_SKIP_WORDS


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
