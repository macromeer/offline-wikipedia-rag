"""Numbered citation table for retrieved passages, and checks on [n] markers in answers."""

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from retrieval import Chunk

# [3], [1][4], [1, 4], [2-4], [2–4]
_CITATION_RE = re.compile(r'\[(\d+(?:\s*[,–-]\s*\d+)*)\]')
# A trailing "References"/"Sources" section the model adds despite instructions
_REFERENCE_SECTION_RE = re.compile(
    r'\n\s*(?:#+\s*|\*\*)?\[?(References?|Sources?|Bibliography|Citations?)\]?(?:\*\*)?\s*[:\-]?\s*(?:\*\*)?\s*(\n.*)?$',
    re.DOTALL | re.IGNORECASE)
_REFERENCE_HEADING_RE = re.compile(
    r'(?:#+\s*|\*\*)?\[?(References?|Sources?|Bibliography|Citations?)\]?(?:\*\*)?\s*[:\-]?\s*(?:\*\*)?\s*$', re.IGNORECASE)


@dataclass
class Citation:
    number: int
    chunk: Chunk
    url: Optional[str] = None

    @property
    def label(self) -> str:
        return self.chunk.label


class CitationTable:
    """
    Passages shown to the model, numbered in the order they were first shown.
    Numbers are stable for a whole chat session so that [n] in earlier answers
    (kept in the history) still refer to the same passage.
    """

    def __init__(self, url_for: Optional[Callable[[Chunk], Optional[str]]] = None):
        self._by_number: Dict[int, Citation] = {}
        self._by_chunk: Dict[str, int] = {}
        self._url_for = url_for

    def __len__(self) -> int:
        return len(self._by_number)

    def __contains__(self, number: int) -> bool:
        return number in self._by_number

    def __getitem__(self, number: int) -> Citation:
        return self._by_number[number]

    def add(self, chunk: Chunk) -> Tuple[int, bool]:
        """Number for chunk, and whether it is new to the table"""
        if chunk.chunk_id in self._by_chunk:
            return self._by_chunk[chunk.chunk_id], False
        number = len(self._by_number) + 1
        url = self._url_for(chunk) if self._url_for else None
        self._by_number[number] = Citation(number, chunk, url)
        self._by_chunk[chunk.chunk_id] = number
        return number, True

    def citations(self, numbers) -> List[Citation]:
        return [self._by_number[n] for n in sorted(set(numbers)) if n in self._by_number]


def cited_numbers(text: str) -> List[int]:
    """Citation numbers in order of first appearance; ranges like [2-4] are expanded"""
    numbers = []
    for match in _CITATION_RE.finditer(text):
        for part in re.split(r'\s*,\s*', match.group(1)):
            bounds = [int(b) for b in re.split(r'\s*[–-]\s*', part)]
            span = range(bounds[0], bounds[-1] + 1) if len(bounds) == 2 and 0 < bounds[1] - bounds[0] < 20 else bounds
            for n in span:
                if n not in numbers:
                    numbers.append(n)
    return numbers


def strip_reference_section(text: str) -> str:
    return _REFERENCE_SECTION_RE.sub('', text).rstrip()


class ReferenceSectionFilter:
    """
    Streaming counterpart of strip_reference_section: passes text through,
    holding back each new line until it is clearly not a "References" heading,
    and swallowing everything from such a heading on.
    """

    def __init__(self):
        self._line = ''        # start of the current line, not yet emitted
        self._holding = True   # still deciding about the current line
        self._stopped = False

    def feed(self, text: str) -> str:
        out = []
        for ch in text:
            if self._stopped:
                break
            if ch == '\n':
                if self._holding and _REFERENCE_HEADING_RE.fullmatch(self._line.strip()) and self._line.strip():
                    self._stopped = True
                    break
                out.append(self._line + '\n')
                self._line, self._holding = '', True
            elif self._holding:
                self._line += ch
                stripped = self._line.strip()
                if len(stripped) > 16 or (stripped and not _could_be_heading(stripped)):
                    out.append(self._line)
                    self._line, self._holding = '', False
            else:
                out.append(ch)
        return ''.join(out)

    def flush(self) -> str:
        if self._stopped or (self._holding and _REFERENCE_HEADING_RE.fullmatch(self._line.strip())
                             and self._line.strip()):
            return ''
        line, self._line = self._line, ''
        return line


def _could_be_heading(prefix: str) -> bool:
    """True if prefix could still grow into a references heading"""
    word = re.sub(r'^(?:#+\s*|\*\*)?\[?', '', prefix).lower()
    return any(h.startswith(word) or word.startswith(h)
               for h in ('references', 'sources', 'bibliography', 'citations'))
