"""
Question set I/O and gold-title resolution.

One JSON object per line:
  id          unique, prefixed by source (hotpot-, popqa-, hw-)
  source      hotpotqa | popqa | handwritten
  tag         subset for the report: bridge/comparison (HotpotQA), the PopQA
              relation, or the failure mode a hand-written question covers
  question
  answers     gold answer aliases ([] = no short answer; retrieval only)
  gold_titles Wikipedia titles; resolved against the ZIM at run time, so
              renamed articles still match through their redirects
  gold_mode   all: every gold article is needed (multi-hop)
              any: any one of them is enough (alternatives)
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / 'questions.jsonl'
HANDWRITTEN = EVAL_DIR / 'handwritten.jsonl'


def read_jsonl(path) -> Iterator[Dict]:
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_questions(path=QUESTIONS, sources: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict]:
    rows = [q for q in read_jsonl(path) if not sources or q['source'] in sources]
    return rows[:limit] if limit else rows


def subset(question: Dict) -> str:
    """Report group: source plus HotpotQA type; PopQA relations are pooled"""
    if question['source'] == 'hotpotqa':
        return f"hotpotqa-{question['tag']}"
    return question['source']


def resolve_title(store, title: str) -> Optional[str]:
    """ZIM path of the article with this exact title (redirects followed), or None"""
    archive = store.archive
    for getter, key in ((archive.get_entry_by_title, title), (archive.get_entry_by_path, title.replace(' ', '_'))):
        try:
            return store._resolve(getter(key)).path
        except KeyError:
            continue
    return None


def resolve_golds(store, question: Dict) -> List[str]:
    """Paths of the question's gold articles that exist in this ZIM, in order, de-duplicated"""
    paths = (resolve_title(store, t) for t in question['gold_titles'])
    return list(dict.fromkeys(p for p in paths if p))
