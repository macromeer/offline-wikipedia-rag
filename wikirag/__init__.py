"""Offline Wikipedia RAG: chat with a Kiwix ZIM of English Wikipedia through local Ollama models."""

from .agent import TurnResult, WikiChat
from .citations import CitationTable, cited_numbers
from .oneshot import OneShotRAG
from .zimfiles import resolve_zim_path

__all__ = ['CitationTable', 'OneShotRAG', 'TurnResult', 'WikiChat', 'cited_numbers', 'resolve_zim_path']
