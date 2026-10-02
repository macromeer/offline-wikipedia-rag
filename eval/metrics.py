"""
Scoring for the eval. Pure functions, no ZIM or model needed.

Answers are long-form (a direct first sentence, then details), so exact match
against a short gold answer is meaningless. The headline answer metric is
`answer_match`: a gold alias appears in the answer as a whole-token sequence
after SQuAD-style normalisation (the PopQA convention). Token F1 is computed
against the first sentence only, where the prompt asks for the direct answer.
"""

import re
import string
import unicodedata
from collections import Counter
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence

_ARTICLES_RE = re.compile(r'\b(a|an|the)\b')
_PUNCT = set(string.punctuation) | {'–', '—', '‘', '’', '“', '”', '°', '′', '″'}
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(])')
_CITATION_RE = re.compile(r'\[\d+(?:\s*[,–-]\s*\d+)*\]')
_NUMBER_SEP_RE = re.compile(r'(?<=\d)[,   ](?=\d{3}\b)')
YES_NO = ('yes', 'no')


def normalize(text: str) -> str:
    """Lowercase, fold accents, drop punctuation, articles and [n] markers, squeeze spaces"""
    text = _CITATION_RE.sub(' ', text)
    text = _NUMBER_SEP_RE.sub('', text)            # 299,792,458 == 299792458
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = ''.join(' ' if ch in _PUNCT else ch for ch in text)
    text = _ARTICLES_RE.sub(' ', text)
    return ' '.join(text.split())


def first_sentence(text: str) -> str:
    text = text.strip()
    return _SENTENCE_END_RE.split(text, 1)[0] if text else ''


def _contains_tokens(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    n = len(needle)
    if not n:
        return False
    return any(list(haystack[i:i + n]) == list(needle) for i in range(len(haystack) - n + 1))


def answer_match(answer: str, golds: Sequence[str]) -> bool:
    """
    A gold alias occurs in the answer as whole tokens. Yes/no golds are judged
    on the first sentence only, and fail if it contains both words.
    """
    if not golds:
        return False
    lowered = {g.strip().lower() for g in golds}
    if lowered <= set(YES_NO):
        tokens = normalize(first_sentence(answer)).split()
        said = {w for w in YES_NO if w in tokens}
        return len(said) == 1 and said <= lowered
    tokens = normalize(answer).split()
    return any(_contains_tokens(tokens, normalize(g).split()) for g in golds)


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(gold).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def answer_f1(answer: str, golds: Sequence[str]) -> float:
    """Best token F1 of the answer's first sentence against any gold alias"""
    head = first_sentence(answer)
    return max((token_f1(head, g) for g in golds), default=0.0)


def passages_contain_answer(texts: Iterable[str], golds: Sequence[str]) -> bool:
    """Whether a non-yes/no gold alias occurs in any passage (an upper bound on answerability)"""
    golds = [g for g in golds if g.strip().lower() not in YES_NO]
    if not golds:
        return False
    needles = [normalize(g).split() for g in golds]
    for text in texts:
        tokens = normalize(text).split()
        if any(_contains_tokens(tokens, n) for n in needles):
            return True
    return False


def title_recall(ranked_paths: Sequence[str], gold_paths: Sequence[str], k: Optional[int], mode: str) -> float:
    """
    Recall of gold articles among the first k entries of a ranked list of
    article paths (k=None: the whole list). mode 'any': 1 if any gold is
    present (alternative answers); 'all': fraction of golds present (multi-hop).
    """
    if not gold_paths:
        return 0.0
    top = set(ranked_paths if k is None else ranked_paths[:k])
    found = sum(1 for g in gold_paths if g in top)
    if mode == 'any':
        return 1.0 if found else 0.0
    return found / len(gold_paths)


def mcnemar_p(better: int, worse: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs (a binomial test with p = 0.5)"""
    from math import comb
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(better, worse) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float('nan')
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def summarize(rows: List[Dict], fields: Sequence[str]) -> Dict[str, float]:
    """Mean of each field over the rows that have it (booleans count as 0/1); plus n"""
    out: Dict[str, float] = {'n': len(rows)}
    for name in fields:
        values = [float(r[name]) for r in rows if r.get(name) is not None]
        out[name] = mean(values) if values else float('nan')
    return out


def latency(rows: List[Dict], field: str) -> Dict[str, float]:
    values = [r[field] for r in rows if r.get(field) is not None]
    return {'p50': median(values) if values else float('nan'), 'p95': percentile(values, 0.95)}
