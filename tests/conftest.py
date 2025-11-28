"""Pytest configuration and fixtures"""

import pytest
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama chat response"""
    return {
        'message': {
            'role': 'assistant',
            'content': 'This is a test response from the AI model.'
        }
    }


@pytest.fixture
def mock_ollama_list():
    """Mock Ollama model list"""
    mock_model = Mock()
    mock_model.model = 'llama3.1:8b'
    mock_response = Mock()
    mock_response.models = [mock_model]
    return mock_response


@pytest.fixture
def mock_kiwix_search_html():
    """Mock Kiwix search results HTML"""
    return """
    <html>
        <body>
            <div class="suggestion">
                <a href="/content/wikipedia/A/Photosynthesis">Photosynthesis</a>
            </div>
            <div class="suggestion">
                <a href="/content/wikipedia/A/Biology">Biology</a>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def mock_kiwix_article_html():
    """Mock Kiwix article content HTML"""
    return """
    <html>
        <body>
            <div id="mw-content-text">
                <div class="mw-parser-output">
                    <p>Photosynthesis is the process by which plants convert light energy into chemical energy.</p>
                    <p>This process is essential for life on Earth as it produces oxygen and organic compounds.</p>
                    <p>The process occurs in chloroplasts and involves two main stages: light reactions and dark reactions.</p>
                </div>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def sample_question():
    """Sample question for testing"""
    return "What is photosynthesis?"


@pytest.fixture
def sample_search_results():
    """Sample search results"""
    return [
        {
            'title': 'Photosynthesis',
            'url': 'http://localhost:8080/content/wikipedia/A/Photosynthesis',
            'abstract': 'Photosynthesis is a process used by plants to convert light into energy.'
        },
        {
            'title': 'Chloroplast',
            'url': 'http://localhost:8080/content/wikipedia/A/Chloroplast',
            'abstract': 'Chloroplasts are organelles found in plant cells where photosynthesis occurs.'
        },
        {
            'title': 'Biology',
            'url': 'http://localhost:8080/content/wikipedia/A/Biology',
            'abstract': 'Biology is the study of living organisms.'
        }
    ]
