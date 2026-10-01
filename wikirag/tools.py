"""Tools the model calls during a chat turn: search_wikipedia and read_section."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from retrieval import Chunk, ZimStore

from .citations import CitationTable

MIN_K, MAX_K = 3, 12
READ_MAX_CHUNKS = 4

SEARCH_TOOL = {
    'type': 'function',
    'function': {
        'name': 'search_wikipedia',
        'description': ('Search the offline English Wikipedia. Returns numbered passages '
                        '("[n] Article > Section" followed by the text). Use names and specific terms, '
                        'not a whole question.'),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'Keywords, e.g. "Apollo 11 Neil Armstrong moonwalk"'},
            },
            'required': ['query'],
        },
    },
}

READ_TOOL = {
    'type': 'function',
    'function': {
        'name': 'read_section',
        'description': ('Read a section of a Wikipedia article by exact title. Omit section to read the '
                        'introduction. If the section does not exist, the list of sections is returned.'),
        'parameters': {
            'type': 'object',
            'properties': {
                'title': {'type': 'string', 'description': 'Article title, e.g. "Apollo 11"'},
                'section': {'type': 'string', 'description': 'Section heading, e.g. "Lunar surface operations"'},
            },
            'required': ['title'],
        },
    },
}

TOOLS = [SEARCH_TOOL, READ_TOOL]


@dataclass
class ToolResult:
    content: str                                         # what the model sees
    numbers: List[int] = field(default_factory=list)     # citation numbers shown (new and repeated)
    new: int = 0                                         # how many of them are new to the session


def _body(chunk: Chunk) -> str:
    """Chunk text without its leading "Title > Heading" line"""
    return chunk.text.split('\n', 1)[-1]


class WikiTools:
    """Runs tool calls against a ZimStore, adding every passage shown to the citation table"""

    def __init__(self, store: ZimStore, table: CitationTable):
        self.store = store
        self.table = table
        self._shown: set = set()   # numbers whose text the model has seen this turn

    def begin_turn(self) -> None:
        """Passages from earlier turns are no longer in context; show their text again if retrieved"""
        self._shown = set()

    def run(self, name: str, arguments: Dict[str, Any], k: int) -> ToolResult:
        args = arguments if isinstance(arguments, dict) else {}
        if name == 'search_wikipedia':
            return self.search(str(args.get('query') or ''), k)
        if name == 'read_section':
            return self.read_section(str(args.get('title') or ''), args.get('section'))
        return ToolResult(f"Unknown tool {name!r}. Available tools: search_wikipedia, read_section.")

    def _render(self, chunks: Sequence[Chunk], header: str) -> ToolResult:
        lines, seen, numbers, new = [header], [], [], 0
        for chunk in chunks:
            number, is_new = self.table.add(chunk)
            numbers.append(number)
            new += is_new
            if number in self._shown:
                seen.append(f"[{number}] {chunk.label}")
            else:
                self._shown.add(number)
                lines.append(f"\n[{number}] {chunk.label}\n{_body(chunk)}")
        if seen:
            lines.append(f"\nAlready shown above: {'; '.join(seen)}")
        return ToolResult('\n'.join(lines), numbers, new)

    def search(self, query: str, k: int) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult('Empty query. Pass keywords, e.g. {"query": "Apollo 11"}.')
        result = self.store.retrieve(query, k=max(MIN_K, min(MAX_K, k)))
        if not result.chunks:
            return ToolResult(f'No passages found for "{query}". Try other keywords: names, places, specific terms.')
        return self._render([sc.chunk for sc in result.chunks], f'Passages for "{query}":')

    def _resolve_title(self, title: str) -> Optional[str]:
        hit = self.store.lookup_title(title) or next(iter(self.store.suggest_titles(title, 1)), None)
        return hit[0] if hit else None

    def read_section(self, title: str, section: Optional[str] = None) -> ToolResult:
        title = title.strip()
        path = self._resolve_title(title) if title else None
        if not path:
            return ToolResult(f'No article titled "{title}". Use search_wikipedia to find the right title.')
        article = self.store.get_article(path)
        if article.is_disambiguation:
            targets = ', '.join(p.replace('_', ' ') for p in article.links[:20])
            return ToolResult(f'"{article.title}" is a disambiguation page. It lists: {targets}')

        wanted = (section or '').strip().lower()
        if not wanted or wanted in ('introduction', 'lead', 'summary', article.title.lower()):
            chunks = [c for c in article.chunks if c.is_lead]
        else:
            exact = [c for c in article.chunks if any(h.lower() == wanted for h in c.heading_path)]
            chunks = exact or [c for c in article.chunks if any(wanted in h.lower() for h in c.heading_path)]
        if not chunks:
            headings = list(dict.fromkeys(' > '.join(c.heading_path) for c in article.chunks if c.heading_path))
            return ToolResult(f'"{article.title}" has no section "{section}". Sections: {"; ".join(headings[:40])}')
        return self._render(chunks[:READ_MAX_CHUNKS], f'From "{article.title}":')
