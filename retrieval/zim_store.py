"""
Retrieval straight from a Wikipedia ZIM file with python-libzim.

Articles are found with the ZIM's Xapian full-text index plus exact title
lookups, split into sections, chunked, and the chunks are ranked with BM25
against the question. No kiwix-serve and no network access are needed.
"""

import json
import math
import os
import posixpath
import re
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote

import lxml.html
from libzim.reader import Archive
from libzim.search import Query, Searcher
from libzim.suggestion import SuggestionSearcher


CHARS_PER_TOKEN = 4  # same rough estimate as the generator side; consistency matters
DEFAULT_CHUNK_TOKENS = 1200

# English function words and question scaffolding. Xapian drops stopwords on its
# own; this list is for title-span detection and the chunk-level BM25.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each
few for from further had has have having he her here hers herself him himself his how
i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that
the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom whose why will with
would you your yours yourself yourselves
tell explain describe compare give please know anything something
""".split())

# Sections that carry no article prose
SKIPPED_SECTIONS = frozenset({
    'see also', 'notes', 'references', 'bibliography', 'further reading',
    'external links', 'sources', 'citations', 'footnotes', 'works cited',
    'explanatory notes', 'general references', 'notes and references',
    'references and notes', 'gallery', 'general and cited sources', 'cited sources',
})

# Elements dropped before text extraction, matched on any class token
NOISE_CLASSES = frozenset({
    'infobox', 'navbox', 'navbox-styles', 'vertical-navbox', 'sidebar', 'metadata',
    'ambox', 'mbox-small', 'hatnote', 'noprint', 'mw-editsection', 'mw-editsection-like',
    'reflist', 'mw-references-wrap', 'references', 'reference', 'mw-ref', 'thumb',
    'tmulti', 'ext-phonos', 'IPA', 'shortdescription', 'sistersitebox', 'side-box',
    'portalbox', 'toc', 'mw-empty-elt', 'authority-control', 'geo-default', 'geo-multi-punct',
    'mwe-math-fallback-image-inline', 'mwe-math-fallback-image-display', 'navigation-not-searchable',
    'zim-footer', 'tocright', 'tocleft',
})
NOISE_TAGS = frozenset({'style', 'script', 'figure', 'meta', 'link', 'img', 'audio', 'video', 'noscript'})
HEADING_TAGS = frozenset({'h2', 'h3', 'h4', 'h5', 'h6'})
BLOCK_TAGS = frozenset({'p', 'ul', 'ol', 'dl', 'blockquote', 'pre'})

_CITATION_RE = re.compile(r'\[\s*(?:\d+|[a-z]{1,2}|(?:note|nb|n)\s*\d+|[a-z ]{0,30}needed)\s*\]', re.IGNORECASE)
_EMPTY_PARENS_RE = re.compile(r'\(\s*[;,]?\s*\)')
_WS_RE = re.compile(r'\s+')
_WORD_RE = re.compile(r"[\w][\w'’\-]*", re.UNICODE)
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')
_REFERENCE_HEADING_RE = re.compile(r'\b(references|footnotes|citations|bibliography)\b')
_REFER_TO_RE = re.compile(r'\b(?:may|might|can|could|commonly)\s+(?:also\s+)?(?:refer|stand)\s+(?:to|for)\b', re.IGNORECASE)


@dataclass
class Section:
    title: str                      # article title
    heading_path: Tuple[str, ...]   # () for the lead, ('History', '20th century') for a subsection
    text: str                       # blocks separated by blank lines
    anchor: Optional[str] = None    # HTML id of the heading, for #fragment links

    @property
    def label(self) -> str:
        return ' > '.join((self.title,) + self.heading_path)


@dataclass
class Chunk:
    chunk_id: str                   # "<zim path>#<ordinal>", stable for a given chunk size
    path: str
    title: str
    heading_path: Tuple[str, ...]
    text: str                       # starts with the "Title > Heading" label line
    anchor: Optional[str] = None

    @property
    def label(self) -> str:
        return ' > '.join((self.title,) + self.heading_path)

    @property
    def is_lead(self) -> bool:
        return not self.heading_path


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    reason: str = 'bm25'            # 'bm25' or 'title' (forced lead of an exact title hit)


@dataclass(frozen=True)
class TitleHit:
    path: str
    title: str
    span: str                       # the words of the question that matched
    strong: bool


@dataclass
class RetrievalResult:
    chunks: List[ScoredChunk]
    title_hits: List[Tuple[str, str]]       # (path, title) of exact title matches
    candidates: List[Tuple[str, str]]       # (path, title) of articles whose chunks were scored
    timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- text helpers

def _clean(text: str) -> str:
    text = _CITATION_RE.sub('', text)
    text = _EMPTY_PARENS_RE.sub('', text)
    text = _WS_RE.sub(' ', text).strip()
    return text.replace(' ,', ',').replace(' .', '.')


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens with possessives and plural -s/-es/-ies folded."""
    tokens = []
    for raw in _WORD_RE.findall(text.lower()):
        word = raw.replace('’', "'").strip("'-")
        if word.endswith("'s"):
            word = word[:-2]
        tokens.append(_light_stem(word))
    return [t for t in tokens if t]


def _light_stem(word: str) -> str:
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith(('ches', 'shes', 'sses', 'xes', 'zes')):
        return word[:-2]
    if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
        return word[:-1]
    return word


def query_terms(question: str) -> List[str]:
    """Content words of a question, in order, original case, without duplicates."""
    seen, terms = set(), []
    for raw in _WORD_RE.findall(question):
        word = raw.replace('’', "'").strip("'-")
        if word.lower().endswith("'s"):
            word = word[:-2]
        key = word.lower()
        if not word or key in STOPWORDS or key in seen:
            continue
        seen.add(key)
        terms.append(word)
    return terms


def title_spans(question: str, max_words: int = 6) -> List[Tuple[int, int, str]]:
    """
    Candidate article titles in a question: word n-grams within a clause that do
    not start or end with a stopword. A capitalised stopword that is not the first
    word may start a span ("Is The Expanse good?" -> "The Expanse").
    Returns (start, end, text) with word offsets across the whole question.
    """
    spans = []
    offset = 0
    for clause in re.split(r'[,;:?!()"“”]|\.(?:\s|$)', question):
        words = []
        for raw in clause.split():
            word = raw.strip("'\"‘’“”.")
            if word.lower().endswith(("'s", "’s")):
                word = word[:-2]
            if word:
                words.append(word)
        for i in range(len(words)):
            first = words[i]
            starts_ok = first.lower() not in STOPWORDS or (first[0].isupper() and offset + i > 0)
            if not starts_ok:
                continue
            for j in range(i + 1, min(i + max_words, len(words)) + 1):
                if words[j - 1].lower() in STOPWORDS:
                    continue
                spans.append((offset + i, offset + j, ' '.join(words[i:j])))
        offset += len(words)
    return spans


# ---------------------------------------------------------------- HTML -> sections

def _is_skipped_heading(text: str) -> bool:
    text = text.lower().strip()
    return text in SKIPPED_SECTIONS or bool(_REFERENCE_HEADING_RE.search(text))


def _classes(el) -> set:
    return set((el.get('class') or '').split())


def _is_noise(el) -> bool:
    if el.tag in NOISE_TAGS:
        return True
    if el.get('id') in ('coordinates', 'toc'):
        return True
    return bool(_classes(el) & NOISE_CLASSES)


def _strip_noise(root) -> None:
    for math_el in list(root.iter('math')):
        alt = math_el.get('alttext') or ''
        alt = re.sub(r'^\{\\displaystyle\s*(.*)\}$', r'\1', alt.strip(), flags=re.S)
        replacement = lxml.html.Element('span')
        replacement.text = f' {alt} ' if alt else ''
        replacement.tail = math_el.tail
        math_el.getparent().replace(math_el, replacement)
    for el in list(root.iter()):
        if not isinstance(el.tag, str):     # comments, processing instructions
            if el.getparent() is not None:
                el.drop_tree()
            continue
        if el.getparent() is not None and _is_noise(el):
            el.drop_tree()


def _list_text(el, depth: int = 0) -> str:
    lines = []
    for item in el:
        if not isinstance(item.tag, str):
            continue
        nested = [c for c in item if c.tag in ('ul', 'ol', 'dl')]
        for c in nested:
            item.remove(c)
        text = _clean(item.text_content())
        if text:
            bullet = '' if item.tag in ('dt', 'dd') else '- '
            lines.append('  ' * depth + bullet + text)
        for c in nested:
            sub = _list_text(c, depth + 1)
            if sub:
                lines.append(sub)
    return '\n'.join(lines)


def _table_text(table) -> str:
    rows = []
    for tr in table.iter('tr'):
        cells = [_clean(c.text_content()) for c in tr if c.tag in ('td', 'th')]
        if any(cells):
            rows.append(' | '.join(cells))
    return '\n'.join(rows)


def _block_text(el) -> str:
    if el.tag in ('ul', 'ol', 'dl'):
        return _list_text(el)
    if el.tag == 'table':
        return _table_text(el)
    return _clean(el.text_content())


def article_to_sections(html: str, title: str) -> List[Section]:
    """
    Split an article into its lead (heading_path == ()) and one Section per
    heading, in document order. Works on Parsoid HTML (nested <section>s) and on
    flat MediaWiki HTML (headings as siblings of paragraphs). Infoboxes, navboxes,
    figures, reference lists and citation markers are removed; reference-type
    sections (References, See also, External links, ...) are skipped with their
    subsections.
    """
    doc = lxml.html.document_fromstring(html)
    roots = doc.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " mw-parser-output ")]')
    root = roots[0] if roots else (doc.find('body') if doc.find('body') is not None else doc)
    _strip_noise(root)

    sections: List[Section] = []
    state = {'path': (), 'anchor': None, 'blocks': [], 'skip_level': None}

    def flush():
        if state['skip_level'] is None and state['blocks']:
            sections.append(Section(title, state['path'], '\n\n'.join(state['blocks']), state['anchor']))
        state['blocks'] = []

    def heading(h):
        flush()
        level = int(h.tag[1])
        text = _clean(h.text_content())
        if state['skip_level'] is not None and level > state['skip_level']:
            return
        state['skip_level'] = level if _is_skipped_heading(text) else None
        state['path'] = state['path'][:level - 2] + (text,)
        state['anchor'] = h.get('id')

    def walk(el):
        for child in el:
            if not isinstance(child.tag, str):
                continue
            if child.tag in HEADING_TAGS:
                heading(child)
            elif child.tag in BLOCK_TAGS or (child.tag == 'table' and 'wikitable' in _classes(child)):
                if state['skip_level'] is None:
                    text = _block_text(child)
                    if text:
                        state['blocks'].append(text)
            elif child.tag != 'table':
                walk(child)

    walk(root)
    flush()
    return sections


def is_disambiguation(sections: Sequence[Section], title: str = '', html: str = '') -> bool:
    """
    Disambiguation pages in Kiwix dumps have lost their {{disambiguation}} box;
    recognise them by title, by leftover markup, or by a short "X may refer to:" lead.
    """
    if title.endswith('(disambiguation)'):
        return True
    if 'id="disambigbox"' in html or 'dmbox-disambig' in html:
        return True
    lead = sections[0].text if sections and not sections[0].heading_path else ''
    return len(lead) < 400 and bool(_REFER_TO_RE.search(lead))


def disambiguation_links(html: str, path: str) -> List[str]:
    """Target paths of the first link in each list item, in page order."""
    doc = lxml.html.document_fromstring(html)
    base = posixpath.dirname(path)
    targets = []
    for anchor in doc.xpath('//li/a[1][@href]'):
        href = anchor.get('href').split('#')[0]
        if not href or '://' in href or href.startswith(('/', '//')):
            continue
        target = posixpath.normpath(posixpath.join(base, unquote(href)))
        if target not in targets:
            targets.append(target)
    return targets


# ---------------------------------------------------------------- chunking

def _split_long(block: str, budget: int) -> List[str]:
    """Split one oversize block at sentence boundaries, hard-cutting run-on sentences."""
    pieces, current = [], ''
    for sentence in _SENTENCE_END_RE.split(block):
        while len(sentence) > budget:
            if current:
                pieces.append(current)
                current = ''
            pieces.append(sentence[:budget])
            sentence = sentence[budget:]
        if current and len(current) + 1 + len(sentence) > budget:
            pieces.append(current)
            current = sentence
        else:
            current = f'{current} {sentence}' if current else sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_sections(sections: Sequence[Section], path: str, max_tokens: int = DEFAULT_CHUNK_TOKENS) -> List[Chunk]:
    """
    Pack each section's blocks into chunks of at most ~max_tokens, never mixing
    sections, no overlap. Every chunk starts with its "Title > Heading" label.
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    chunks: List[Chunk] = []
    for section in sections:
        budget = max(200, max_chars - len(section.label) - 1)
        pieces, current = [], ''
        for block in section.text.split('\n\n'):
            parts = _split_long(block, budget) if len(block) > budget else [block]
            for part in parts:
                if current and len(current) + 2 + len(part) > budget:
                    pieces.append(current)
                    current = part
                else:
                    current = f'{current}\n\n{part}' if current else part
        if current:
            pieces.append(current)
        for piece in pieces:
            chunks.append(Chunk(
                chunk_id=f'{path}#{len(chunks)}', path=path, title=section.title,
                heading_path=section.heading_path, text=f'{section.label}\n{piece}',
                anchor=section.anchor,
            ))
    return chunks


# ---------------------------------------------------------------- ranking

class BM25:
    """Okapi BM25 over a small in-memory corpus of token lists."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(doc) for doc in corpus]
        self.lengths = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.lengths) / len(corpus)) if corpus else 0.0
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: Sequence[str]) -> List[float]:
        out = []
        for tf, dl in zip(self.tfs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for term in query:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda key: -scores[key])


# ---------------------------------------------------------------- the store

@dataclass
class Article:
    path: str
    title: str
    chunks: List[Chunk]
    is_disambiguation: bool = False
    links: List[str] = field(default_factory=list)  # targets listed on a disambiguation page


class ZimStore:
    """Read-only access to one Wikipedia ZIM: search, title lookup, sections, chunks."""

    def __init__(self, zim_path, chunk_tokens: int = DEFAULT_CHUNK_TOKENS, cache_size: int = 256,
                 cache_dir: Optional[Path] = None):
        """
        cache_dir: where per-ZIM term statistics are kept between runs
        (default $XDG_CACHE_HOME/offline-wikipedia-rag; pass a Path to override).
        """
        self.zim_path = Path(zim_path)
        self.archive = Archive(str(self.zim_path))
        self.chunk_tokens = chunk_tokens
        self.has_fulltext = self.archive.has_fulltext_index
        self._searcher = Searcher(self.archive) if self.has_fulltext else None
        self._suggester = SuggestionSearcher(self.archive) if self.archive.has_title_index else None
        self._cache: 'OrderedDict[str, Article]' = OrderedDict()
        self._cache_size = cache_size
        if cache_dir is None:
            cache_dir = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'offline-wikipedia-rag'
        self._df_cache_path = Path(cache_dir) / f'df-{self.archive.uuid}.json'
        self._df_cache: Dict[str, int] = self._load_df_cache()
        self._main_path = self._resolve(self.archive.main_entry).path if self.archive.has_main_entry else None

    def metadata(self, key: str) -> Optional[str]:
        try:
            return bytes(self.archive.get_metadata(key)).decode('utf-8', 'replace')
        except (KeyError, RuntimeError):
            return None

    @property
    def date(self) -> Optional[str]:
        return self.metadata('Date')

    @property
    def name(self) -> str:
        return self.metadata('Name') or self.zim_path.stem

    # -- lookups

    def _resolve(self, entry):
        """Follow redirects; returns the final entry."""
        for _ in range(10):
            if not entry.is_redirect:
                return entry
            entry = entry.get_redirect_entry()
        return entry

    def _entry(self, path: str):
        return self._resolve(self.archive.get_entry_by_path(path))

    def search_fulltext(self, query: str, k: int = 20) -> List[Tuple[str, str]]:
        """Xapian full-text search. All terms must match (libzim parses queries as AND)."""
        if not self._searcher or not query.strip():
            return []
        search = self._searcher.search(Query().set_query(query))
        hits, seen = [], set()
        for path in search.getResults(0, k):
            try:
                entry = self._entry(path)
            except KeyError:
                continue
            if entry.path not in seen:
                seen.add(entry.path)
                hits.append((entry.path, entry.title))
        return hits

    def document_frequencies(self, terms: Sequence[str]) -> Dict[str, Optional[int]]:
        """
        Number of articles matching each term, from the full-text index. libzim
        walks the whole posting list for this (~0.1 s per 200k matches), so
        lookups run in parallel and are cached on disk per ZIM.
        """
        keys = [t.lower() for t in terms]
        if not self._searcher:
            return {key: None for key in keys}
        missing = [key for key in OrderedDict.fromkeys(keys) if key not in self._df_cache]
        if missing:
            def count(term):
                return Searcher(self.archive).search(Query().set_query(term)).getEstimatedMatches()
            with ThreadPoolExecutor(max_workers=min(8, len(missing))) as pool:
                self._df_cache.update(zip(missing, pool.map(count, missing)))
            self._save_df_cache()
        return {key: self._df_cache[key] for key in keys}

    def _load_df_cache(self) -> Dict[str, int]:
        try:
            with open(self._df_cache_path, encoding='utf-8') as f:
                data = json.load(f)
            return {k: int(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_df_cache(self) -> None:
        if not self._df_cache_path:
            return
        try:
            self._df_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._df_cache_path.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self._df_cache, f)
            os.replace(tmp, self._df_cache_path)
        except OSError:
            pass

    def suggest_titles(self, text: str, k: int = 10) -> List[Tuple[str, str]]:
        """Title-index suggestions (prefix / word matches on titles)."""
        if not self._suggester or not text.strip():
            return []
        hits, seen = [], set()
        for path in self._suggester.suggest(text).getResults(0, k):
            try:
                entry = self._entry(path)
            except KeyError:
                continue
            if entry.path not in seen:
                seen.add(entry.path)
                hits.append((entry.path, entry.title))
        return hits

    def lookup_title(self, text: str) -> Optional[Tuple[str, str]]:
        """Exact article for a title as typed, trying Wikipedia case conventions."""
        text = text.strip()
        if not text:
            return None
        variants = [text[0].upper() + text[1:], text, text.title()]
        if len(text) <= 6:
            variants.append(text.upper())
        for variant in OrderedDict.fromkeys(variants):
            for getter, key in ((self.archive.get_entry_by_title, variant),
                                (self.archive.get_entry_by_path, variant.replace(' ', '_'))):
                try:
                    entry = self._resolve(getter(key))
                except KeyError:
                    continue
                if entry.path == self._main_path:
                    continue
                return entry.path, entry.title
        return None

    def get_article_html(self, path: str) -> str:
        return bytes(self._entry(path).get_item().content).decode('utf-8', 'replace')

    def get_sections(self, path: str) -> List[Section]:
        entry = self._entry(path)
        return article_to_sections(self.get_article_html(entry.path), entry.title)

    def get_article(self, path: str) -> Article:
        """Parsed and chunked article (redirects followed), cached. Raises KeyError if absent."""
        entry = self._entry(path)
        if entry.path in self._cache:
            self._cache.move_to_end(entry.path)
            return self._cache[entry.path]
        html = self.get_article_html(entry.path)
        sections = article_to_sections(html, entry.title)
        disambiguation = is_disambiguation(sections, entry.title, html)
        article = Article(
            path=entry.path, title=entry.title,
            chunks=chunk_sections(sections, entry.path, self.chunk_tokens),
            is_disambiguation=disambiguation,
            links=disambiguation_links(html, entry.path) if disambiguation else [],
        )
        self._cache[entry.path] = article
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return article

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        path, _, ordinal = chunk_id.rpartition('#')
        try:
            return self.get_article(path).chunks[int(ordinal)]
        except (KeyError, ValueError, IndexError):
            return None

    # -- retrieval

    def find_title_hits(self, question: str) -> List[TitleHit]:
        """
        Articles whose title appears verbatim in the question, longest spans
        first, overlapping shorter spans dropped. A hit is strong when the span
        is capitalised as typed (not just sentence-initial: "NASA", "The
        Expanse") or covers every content word of the question ("capital of
        australia"). Weak single-word hits ("goals") are dropped when a strong
        hit exists.
        """
        content = {t.lower() for t in query_terms(question)}
        found = []
        for start, end, text in title_spans(question):
            hit = self.lookup_title(text)
            if hit:
                found.append((start, end, text, hit))
        found.sort(key=lambda f: (-(f[1] - f[0]), f[0]))
        accepted, used = [], set()
        for start, end, text, hit in found:
            if used.intersection(range(start, end)):
                continue
            used.update(range(start, end))
            words = {w.lower() for w in query_terms(text)}
            strong = (any(c.isupper() for c in text[1:]) or (text[0].isupper() and start > 0)
                      or (bool(content) and content <= words))
            accepted.append((start, TitleHit(hit[0], hit[1], text, strong)))
        if any(h.strong for _, h in accepted):
            accepted = [(s, h) for s, h in accepted if h.strong or ' ' in h.span]
        hits, seen = [], set()
        for _, hit in sorted(accepted, key=lambda a: a[0]):
            if hit.path not in seen:
                seen.add(hit.path)
                hits.append(hit)
        return hits

    def _fulltext_candidates(self, terms: List[str], k: int) -> List[Tuple[str, str]]:
        """
        All-terms query first; if it returns fewer than k articles, add
        leave-one-out queries, then single-term queries, and fuse with RRF.
        """
        words = [t.lower() for t in terms]
        if not words:
            return []
        lists = [self.search_fulltext(' '.join(words), k)]
        if len(lists[0]) < k and len(words) > 2:
            for i in range(len(words)):
                lists.append(self.search_fulltext(' '.join(words[:i] + words[i + 1:]), k))
        if sum(len(hits) for hits in lists) < k and len(words) > 1:
            for word in words:
                lists.append(self.search_fulltext(word, k))
        titles = {path: title for hits in lists for path, title in hits}
        return [(path, titles[path]) for path in reciprocal_rank_fusion([[p for p, _ in hits] for hits in lists])]

    def _idf(self, terms: List[str], dfs: Dict[str, Optional[int]],
             local_corpus: Sequence[Sequence[str]]) -> Dict[str, float]:
        """
        IDF per stemmed query token from whole-ZIM document frequencies. The
        candidate pool is biased toward the query, so its own statistics would
        make the main topic word worthless ("earthquake" occurs in every chunk).
        Falls back to pool statistics when the ZIM has no full-text index.
        """
        idf = {}
        for term in terms:
            df, n = dfs.get(term.lower()), self.archive.article_count
            for token in tokenize(term):
                if df is None:
                    df, n = sum(1 for doc in local_corpus if token in doc), max(len(local_corpus), 1)
                idf[token] = max(idf.get(token, 0.0), math.log(1 + (n - df + 0.5) / (df + 0.5)))
        return idf

    def retrieve(self, question: str, k: int = 10, max_articles: int = 12, per_article: int = 3) -> RetrievalResult:
        """
        Top-k chunks for a question.

        Articles: exact title hits first (a disambiguation hit is replaced by the
        pages it lists, preferring those the full-text search also found), then
        full-text candidates. Chunks are ranked by BM25 with whole-ZIM IDF, fused
        by RRF with their article's rank. The lead of every exact title hit is
        always included; at most per_article chunks come from one article
        (twice that for title hits).
        """
        timings = {}
        t0 = time.perf_counter()
        terms = query_terms(question)
        df_lookup = ThreadPoolExecutor(max_workers=1)
        df_future = df_lookup.submit(self.document_frequencies, terms)  # slowest step; overlap it
        hits = self.find_title_hits(question)
        timings['titles'] = time.perf_counter() - t0

        t1 = time.perf_counter()
        if self.has_fulltext:
            fulltext = self._fulltext_candidates(terms, max_articles)
        else:
            fulltext = [hit for term in terms for hit in self.suggest_titles(term, 5)]
        timings['search'] = time.perf_counter() - t1

        t2 = time.perf_counter()
        fulltext_paths = {path for path, _ in fulltext}
        # weak hits the full-text search agrees with ("earthquakes" in "How do
        # earthquakes cause tsunamis?") count as strong; the rest are candidates
        title_hits = [h for h in hits if h.strong or h.path in fulltext_paths]
        weak = [h.path for h in hits if h not in title_hits]
        primary: List[Article] = []      # title hits (and disambiguation targets the search agrees with)
        secondary: List[str] = []        # other disambiguation targets
        for hit in title_hits:
            article = self.get_article(hit.path)
            if not article.is_disambiguation:
                primary.append(article)
                continue
            for target in article.links:
                try:
                    resolved = self._entry(target).path
                except KeyError:
                    continue
                if resolved in fulltext_paths:
                    primary.append(self.get_article(resolved))
                else:
                    secondary.append(resolved)

        ordered: 'OrderedDict[str, Article]' = OrderedDict((a.path, a) for a in primary)
        budget = max_articles + len(ordered)
        for path in [p for p, _ in fulltext[:max_articles // 2]] + weak + secondary[:6] + [p for p, _ in fulltext]:
            if len(ordered) >= budget:
                break
            if path in ordered:
                continue
            try:
                article = self.get_article(path)
            except KeyError:
                continue
            if not article.is_disambiguation:
                ordered[article.path] = article
        timings['parse'] = time.perf_counter() - t2

        t3 = time.perf_counter()
        pool = [(rank, chunk) for rank, article in enumerate(ordered.values()) for chunk in article.chunks]
        corpus = [tokenize(chunk.text) for _, chunk in pool]
        dfs = df_future.result()
        df_lookup.shutdown()
        idf = self._idf(terms, dfs, corpus)
        bm25 = BM25(corpus)
        bm25.idf = idf
        scores = bm25.scores(list(idf)) if pool else []
        by_bm25 = sorted(range(len(pool)), key=lambda i: -scores[i])
        bm25_rank = {i: r for r, i in enumerate(by_bm25)}
        # RRF over three signals: chunk BM25 rank, article rank, and membership in
        # the title hits (the question names that article, so its sections lead)
        primary_paths = {a.path for a in primary}
        fused = {i: 1.0 / (60 + bm25_rank[i] + 1) + 1.0 / (60 + pool[i][0] + 1)
                 + (1.0 / 61 if pool[i][1].path in primary_paths else 0.0)
                 for i in range(len(pool)) if scores[i] > 0}

        selected: List[ScoredChunk] = []
        per_path: Counter = Counter()
        for article in primary:
            if article.chunks and article.chunks[0].is_lead:
                lead = article.chunks[0]
                selected.append(ScoredChunk(lead, 0.0, 'title'))
                per_path[lead.path] += 1
        chosen = {sc.chunk.chunk_id for sc in selected}
        for i in sorted(fused, key=lambda i: -fused[i]):
            if len(selected) >= max(k, len(chosen)):
                break
            chunk = pool[i][1]
            cap = per_article * 2 if chunk.path in primary_paths else per_article
            if chunk.chunk_id in chosen or per_path[chunk.path] >= cap:
                continue
            selected.append(ScoredChunk(chunk, scores[i]))
            per_path[chunk.path] += 1
            chosen.add(chunk.chunk_id)
        for sc in selected:
            if sc.reason == 'title':
                sc.score = scores[next(i for i, (_, c) in enumerate(pool) if c.chunk_id == sc.chunk.chunk_id)]
        timings['rank'] = time.perf_counter() - t3
        timings['total'] = time.perf_counter() - t0
        return RetrievalResult(selected, [(h.path, h.title) for h in title_hits],
                               [(a.path, a.title) for a in ordered.values()], timings)
