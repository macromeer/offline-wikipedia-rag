"""
Build eval/questions.jsonl from HotpotQA, PopQA and eval/handwritten.jsonl.

    uv run --group eval python -m eval.prepare [--zim PATH]

Sources (downloaded once into ~/.cache/offline-wikipedia-rag/eval-sources):
  HotpotQA distractor validation set (CC BY-SA 4.0), hotpotqa/hotpot_qa on Hugging Face
  PopQA test set (MIT), akariasai/PopQA on Hugging Face

Sampling is seeded. A question is kept only if its gold articles exist in the
ZIM and are not disambiguation pages (for gold_mode 'all', every one of them),
so the set measures retrieval and answering, not dump drift since 2017/2022.
"""

import argparse
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from retrieval import ZimStore
from wikirag.zimfiles import resolve_zim_path

from .dataset import HANDWRITTEN, QUESTIONS, read_jsonl, resolve_title, write_jsonl

SOURCES_DIR = Path.home() / '.cache' / 'offline-wikipedia-rag' / 'eval-sources'
HOTPOT_URL = 'https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet'
POPQA_URL = 'https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv'
SEED = 20261002
# answers from 2017 that may have changed since ("the current Attorney General of Missouri")
_TIME_SENSITIVE_RE = re.compile(r'\b(current|currently|present|now|incumbent)\b', re.IGNORECASE)


def fetch(url: str, name: str) -> Path:
    path = SOURCES_DIR / name
    if not path.exists():
        SOURCES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        tmp = path.with_suffix('.part')
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


class GoldChecker:
    def __init__(self, store: ZimStore):
        self.store = store
        self._cache = {}

    def ok(self, title: str) -> bool:
        if title not in self._cache:
            path = resolve_title(self.store, title)
            try:
                self._cache[title] = bool(path) and not self.store.get_article(path).is_disambiguation
            except KeyError:
                self._cache[title] = False
        return self._cache[title]

    def keep(self, question) -> bool:
        good = [t for t in question['gold_titles'] if self.ok(t)]
        if question['gold_mode'] == 'all':
            return len(good) == len(question['gold_titles'])
        question['gold_titles'] = good
        return bool(good)


def hotpot_questions(rng: random.Random, checker: GoldChecker, quotas):
    import pyarrow.parquet as pq
    table = pq.read_table(fetch(HOTPOT_URL, 'hotpot_validation.parquet'),
                          columns=['id', 'question', 'answer', 'type', 'supporting_facts'])
    rows = table.to_pylist()
    rng.shuffle(rows)
    taken, rejected = defaultdict(list), 0
    for row in rows:
        kind = row['type']
        if len(taken[kind]) >= quotas.get(kind, 0) or _TIME_SENSITIVE_RE.search(row['question']):
            continue
        titles = list(dict.fromkeys(row['supporting_facts']['title']))
        q = {'id': f"hotpot-{row['id']}", 'source': 'hotpotqa', 'tag': kind, 'question': row['question'].strip(),
             'answers': [row['answer'].strip()], 'gold_titles': titles, 'gold_mode': 'all'}
        if len(titles) == 2 and checker.keep(q):
            taken[kind].append(q)
        else:
            rejected += 1
        if all(len(taken[k]) >= n for k, n in quotas.items()):
            break
    print(f"  HotpotQA: {rejected} sampled question(s) dropped, gold article missing or a disambiguation page")
    return [q for kind in quotas for q in taken[kind]]


def popqa_questions(rng: random.Random, checker: GoldChecker, total: int):
    """Stratified over relation and subject popularity (thirds by s_pop within each relation)"""
    import csv
    with open(fetch(POPQA_URL, 'popqa_test.tsv'), encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    by_prop = defaultdict(list)
    for row in rows:
        by_prop[row['prop']].append(row)
    strata = []
    for prop in sorted(by_prop):
        group = sorted(by_prop[prop], key=lambda r: int(r['s_pop']))
        third = len(group) // 3
        for band, part in (('rare', group[:third]), ('mid', group[third:2 * third]), ('popular', group[2 * third:])):
            part = part[:]
            rng.shuffle(part)
            strata.append((prop, band, part))
    taken, cursors, rejected = [], [0] * len(strata), 0
    while len(taken) < total and any(c < len(s[2]) for c, s in zip(cursors, strata)):
        for i, (prop, band, part) in enumerate(strata):
            while cursors[i] < len(part) and len(taken) < total:
                row = part[cursors[i]]
                cursors[i] += 1
                q = {'id': f"popqa-{row['id']}", 'source': 'popqa', 'tag': prop, 'band': band,
                     'question': row['question'].strip(), 'answers': json.loads(row['possible_answers']),
                     'gold_titles': [row['s_wiki_title']], 'gold_mode': 'any'}
                if checker.keep(q):
                    taken.append(q)
                    break
                rejected += 1
    print(f"  PopQA: {rejected} sampled question(s) dropped, gold article missing or a disambiguation page")
    return taken


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--zim', default=None, help='ZIM the gold titles must exist in (default: as the app)')
    parser.add_argument('--bridge', type=int, default=60, help='HotpotQA bridge questions (default 60)')
    parser.add_argument('--comparison', type=int, default=40, help='HotpotQA comparison questions (default 40)')
    parser.add_argument('--popqa', type=int, default=100, help='PopQA questions (default 100)')
    parser.add_argument('--out', default=str(QUESTIONS))
    args = parser.parse_args(argv)

    zim = resolve_zim_path(args.zim)
    if zim is None:
        print("No ZIM found; pass --zim", file=sys.stderr)
        return 1
    checker = GoldChecker(ZimStore(zim))
    rng = random.Random(SEED)

    handwritten = []
    for q in read_jsonl(HANDWRITTEN):
        q = dict(q, source='handwritten')
        before = list(q['gold_titles'])
        if checker.keep(q):
            handwritten.append(q)
            if q['gold_titles'] != before:
                print(f"  {q['id']}: gold titles not in the ZIM dropped: {sorted(set(before) - set(q['gold_titles']))}")
        else:
            print(f"  {q['id']}: dropped, gold titles not in the ZIM: {before}")
    hotpot = hotpot_questions(rng, checker, {'bridge': args.bridge, 'comparison': args.comparison})
    popqa = popqa_questions(rng, checker, args.popqa)

    rows = handwritten + hotpot + popqa
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} questions to {args.out}: {len(handwritten)} hand-written, "
          f"{len(hotpot)} HotpotQA, {len(popqa)} PopQA (gold titles checked against {zim.name})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
