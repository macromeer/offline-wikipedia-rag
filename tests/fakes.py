"""In-memory stand-ins for a ZIM, shared by the retrieval and agent tests"""

from types import SimpleNamespace

from retrieval import Article, Section, ZimStore, chunk_sections


class FakeStore(ZimStore):
    """ZimStore over in-memory articles: {path: (title, [Section...], is_disambiguation, links)}"""

    date = '2026-06-17'

    def __init__(self, articles, titles, fulltext, dfs=None):
        self._articles = articles
        self._titles = titles              # typed title -> path
        self._fulltext = fulltext          # list of paths
        self._dfs = dfs or {}
        self.has_fulltext = True
        self.archive = SimpleNamespace(article_count=1000)

    def suggest_titles(self, text, k=10):
        return []

    def lookup_title(self, text):
        for variant in (text, text[0].upper() + text[1:], text.title()):
            if variant in self._titles:
                path = self._titles[variant]
                return path, self._articles[path][0]
        return None

    def _entry(self, path):
        if path not in self._articles:
            raise KeyError(path)
        return SimpleNamespace(path=path)

    def get_article(self, path):
        title, secs, disambig, links = self._articles[path]
        return Article(path, title, chunk_sections(secs, path), disambig, links)

    def _fulltext_candidates(self, terms, k):
        return [(p, self._articles[p][0]) for p in self._fulltext]

    def document_frequencies(self, terms):
        return {t.lower(): self._dfs.get(t.lower(), 10) for t in terms}


def _article(title, *sections, disambig=False, links=()):
    secs = [Section(title, tuple(h for h in heading.split(' > ') if h), text) for heading, text in sections]
    return (title, secs, disambig, list(links))
