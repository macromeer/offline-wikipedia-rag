"""
Run the eval and write a markdown report.

    uv run python -m eval.run retrieval [--systems fulltext,hybrid,v1]
    uv run python -m eval.run answer    [--systems chat-hybrid,oneshot-hybrid,v1] [--model M]
    uv run python -m eval.run report    RESULTS_DIR

Systems (retrieval stage):
  fulltext   ZimStore.retrieve, full-text only
  hybrid     ZimStore.retrieve with the dense index attached
  v1         the v1 pipeline: kiwix-serve search + LLM article selection (needs kiwix-serve
             and the selection model; its "passages" are whole articles)
Systems (answer stage):
  chat-fulltext, chat-hybrid         WikiChat tool loop, one fresh session per question
  oneshot-fulltext, oneshot-hybrid   retrieve once, answer in one call
  v1                                 the v1 pipeline end to end

Per-question rows go to RESULTS_DIR/<stage>-<system>.jsonl as they finish; a
rerun with the same --out skips questions already there, so an interrupted
run resumes. report.md is rewritten after every system.
"""

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import unquote, urlparse

from retrieval import ZimStore
from retrieval.zim_store import query_terms
from wikirag import llm
from wikirag.citations import cited_numbers
from wikirag.zimfiles import resolve_zim_path

from . import metrics
from .dataset import EVAL_DIR, QUESTIONS, load_questions, read_jsonl, resolve_golds, subset

RETRIEVAL_SYSTEMS = ('fulltext', 'hybrid', 'v1')
ANSWER_SYSTEMS = ('chat-fulltext', 'chat-hybrid', 'oneshot-fulltext', 'oneshot-hybrid', 'v1')
RANK_KS = (1, 3, 5, 10)


def log(text: str = '', end: str = '\n') -> None:
    print(text, end=end, file=sys.stderr, flush=True)


@contextlib.contextmanager
def quiet():
    """The app prints progress to stdout; keep it out of the eval output"""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ------------------------------------------------------------------ setup

class Env:
    """ZIM, dense index, kiwix-serve and models, created on first use"""

    def __init__(self, args):
        self.args = args
        self.zim_path = resolve_zim_path(args.zim)
        if self.zim_path is None:
            raise SystemExit("No ZIM found; pass --zim PATH or set WIKI_ZIM")
        self.store = ZimStore(self.zim_path)
        self._dense = None
        self._kiwix = None
        self._models = None

    @property
    def available(self):
        if self._models is None:
            self._models = llm.list_models()
        return self._models

    def dense(self):
        if self._dense is None:
            from retrieval.dense import DenseIndex
            self._dense = DenseIndex.find(str(self.store.archive.uuid), self.args.dense_index)
            if self._dense is None:
                raise SystemExit("No finished dense index for this ZIM (see docs/DENSE_INDEX.md)")
        return self._dense

    def store_for(self, dense: bool) -> ZimStore:
        """A fresh store (empty article cache) so systems do not warm each other up"""
        store = ZimStore(self.zim_path)
        if dense:
            store.attach_dense(self.dense())
        return store

    def kiwix(self):
        if self._kiwix is None:
            from wikirag.kiwix import KiwixServer
            kiwix = KiwixServer(self.args.kiwix_url, self.zim_path)
            with quiet():
                ok = kiwix.connect(True)
            if not ok:
                raise SystemExit(f"v1 needs kiwix-serve; could not connect to or start it at {kiwix.url}")
            self._kiwix = kiwix
        return self._kiwix

    def answer_model(self) -> str:
        return llm.detect_summarization_model(self.args.model, self.available)

    def selection_model(self) -> str:
        return llm.detect_selection_model(self.args.selection_model, self.available)

    def v1(self, model: str):
        from wikirag.oneshot import OneShotRAG
        return OneShotRAG(model, 'kiwix', kiwix=self.kiwix(), selection_model=self.selection_model())

    def url_path(self, url: Optional[str]) -> Optional[str]:
        """ZIM path (redirects followed) of a kiwix-serve article URL"""
        if not url:
            return None
        path = unquote(urlparse(url).path)
        base = unquote(urlparse(self.kiwix().content_base or '').path).rstrip('/') + '/'
        if path.startswith(base):
            path = path[len(base):]
        if path.startswith('A/'):
            path = path[2:]
        try:
            return self.store._entry(path).path
        except KeyError:
            return None

    def describe(self) -> Dict:
        info = {'zim': self.zim_path.name, 'zim_date': self.store.date, 'questions': str(self.args.questions),
                'started': datetime.now().isoformat(timespec='seconds'), 'commit': _git_commit()}
        index = None
        try:
            from retrieval.dense import DenseIndex
            index = DenseIndex.find(str(self.store.archive.uuid), self.args.dense_index)
        except Exception:
            pass
        if index is not None:
            info['dense'] = f"{len(index):,} passages, scope {index.meta.get('scope')}, dense_k {ZimStore.dense_k}"
        return info


def _git_commit() -> str:
    try:
        out = subprocess.run(['git', 'describe', '--always', '--dirty'], cwd=EVAL_DIR,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def warm_document_frequencies(store: ZimStore, questions: Sequence[Dict]) -> None:
    """
    Term statistics are cached on disk per ZIM after first use. Fill the cache
    up front so latency reflects the steady state, not which system ran first.
    """
    terms = sorted({t for q in questions for t in query_terms(q['question'])})
    t = time.perf_counter()
    store.document_frequencies(terms)
    log(f"Term statistics ready for {len(terms)} terms ({time.perf_counter() - t:.1f}s)")


# ------------------------------------------------------------------ rows

def _base_row(question: Dict, system: str, golds: List[str]) -> Dict:
    return {'id': question['id'], 'subset': subset(question), 'tag': question.get('tag'), 'system': system,
            'question': question['question'], 'gold_paths': golds, 'gold_mode': question['gold_mode']}


def retrieval_row(question: Dict, system: str, golds: List[str], passages: List[Dict], seconds: float,
                  extra: Optional[Dict] = None) -> Dict:
    """passages: [{'path', 'text', 'id'?}] in rank order"""
    row = _base_row(question, system, golds)
    articles = list(dict.fromkeys(p['path'] for p in passages if p['path']))
    mode = question['gold_mode']
    by_passage = [p['path'] for p in passages]
    for k in RANK_KS:
        row[f'r@{k}'] = metrics.title_recall(by_passage, golds, k, mode)
    row['r@ctx'] = metrics.title_recall(articles, golds, None, mode)
    # every gold needed; for 'any' questions the golds are alternatives, so this equals R@ctx
    row['all@ctx'] = (float(all(g in articles for g in golds)) if golds else 0.0) if mode == 'all' else row['r@ctx']
    row['ans@ctx'] = (metrics.passages_contain_answer((p['text'] for p in passages), question['answers'])
                      if [a for a in question['answers'] if a.lower() not in metrics.YES_NO] else None)
    row['passages'] = len(passages)
    row['articles'] = articles
    row['ctx_chars'] = sum(len(p['text']) for p in passages)
    row['ids'] = [p.get('id') or p['path'] for p in passages]
    row['time'] = seconds
    if extra:
        row.update(extra)
    return row


def run_retrieval(env: Env, system: str, questions: Sequence[Dict], k: int):
    if system == 'v1':
        app = env.v1(env.answer_model())
        warm_model(env.selection_model())

        def retrieve(question):
            with quiet():
                contents = app._retrieve_articles(question['question'])
            if isinstance(contents, dict):
                return [], {}
            return [{'path': env.url_path(c['url']), 'text': c['content']} for c in contents], {}
    else:
        store = env.store_for(dense=system == 'hybrid')
        if store.dense:
            store.dense.search('warm up the embedding model', 1)

        def retrieve(question):
            with quiet():
                result = store.retrieve(question['question'], k=k)
            passages = [{'path': sc.chunk.path, 'text': sc.chunk.text, 'id': sc.chunk.chunk_id}
                        for sc in result.chunks]
            reasons = [sc.reason for sc in result.chunks]
            return passages, {'reasons': reasons, 'timings': {n: round(v, 3) for n, v in result.timings.items()}}

    for question in questions:
        golds = resolve_golds(env.store, question)
        t = time.perf_counter()
        try:
            passages, extra = retrieve(question)
        except Exception as e:     # one bad question must not end a long run
            passages, extra = [], {'error': f'{type(e).__name__}: {e}'}
        yield retrieval_row(question, system, golds, passages, time.perf_counter() - t, extra)


def answer_row(question: Dict, system: str, golds: List[str], answer: str, cited_paths: List[str],
               shown_paths: List[str], invalid: List[int], n_cited: int, seconds: float, extra: Dict) -> Dict:
    row = _base_row(question, system, golds)
    answers = question['answers']
    row['answer'] = answer
    row['match'] = metrics.answer_match(answer, answers) if answers else None
    row['f1'] = metrics.answer_f1(answer, answers) if answers else None
    row['cited'] = n_cited
    row['cite_ok'] = n_cited > 0 and not invalid
    row['invalid'] = invalid
    row['cited_gold'] = any(p in golds for p in cited_paths) if golds else None
    row['shown_r'] = metrics.title_recall(list(dict.fromkeys(shown_paths)), golds, None, question['gold_mode'])
    row['time'] = seconds
    row.update(extra)
    return row


def warm_model(model: str) -> None:
    """Load the model before timing, so the first question does not pay for it"""
    import ollama
    ollama.chat(model=model, messages=[{'role': 'user', 'content': 'Hi'}], think=False,
                options={'num_predict': 1, 'num_ctx': 16384})


def run_answer(env: Env, system: str, questions: Sequence[Dict]):
    model = env.answer_model()
    log(f"Answer model: {model}")
    warm_model(model)
    if system == 'v1':
        warm_model(env.selection_model())
    kind, _, flavour = system.partition('-')
    if kind in ('chat', 'oneshot'):
        store = env.store_for(dense=flavour == 'hybrid')
        if store.dense:
            store.dense.search('warm up the embedding model', 1)

    for question in questions:
        golds = resolve_golds(env.store, question)
        t = time.perf_counter()
        try:
            if kind == 'chat':
                from wikirag.agent import WikiChat
                chat = WikiChat(store, model, max_rounds=env.args.max_rounds, history_turns=0)
                with quiet():
                    r = chat.ask(question['question'])
                shown = [chat.table[n].chunk.path for n in r.shown if n in chat.table]
                extra = {'calls': len(r.calls), 'forced': r.forced_search,
                         'queries': [c.arguments for c in r.calls],
                         'first_token': r.first_token,
                         'prompt_tokens': max(r.prompt_tokens) if r.prompt_tokens else None}
                row = answer_row(question, system, golds, r.answer, [c.chunk.path for c in r.citations], shown,
                                 r.invalid_citations, len(r.citations), time.perf_counter() - t, extra)
            else:
                if kind == 'oneshot':
                    from wikirag.oneshot import OneShotRAG
                    app = OneShotRAG(model, 'zim', store=store)
                else:
                    app = env.v1(model)
                with quiet():
                    result = app.query_with_rag(question['question'])
                sources = result.get('sources') or []
                if kind == 'oneshot':
                    paths = [s['chunk_id'].rpartition('#')[0] for s in sources]
                else:
                    paths = [env.url_path(s.get('url')) for s in sources]
                numbers = cited_numbers(result['answer'])
                valid = [n for n in numbers if 1 <= n <= len(sources)]
                invalid = [n for n in numbers if n not in valid]
                row = answer_row(question, system, golds, result['answer'], [paths[n - 1] for n in valid], paths,
                                 invalid, len(set(valid)), time.perf_counter() - t, {})
        except Exception as e:
            row = answer_row(question, system, golds, '', [], [], [], 0, time.perf_counter() - t,
                             {'error': f'{type(e).__name__}: {e}'})
        yield row


# ------------------------------------------------------------------ report

RETRIEVAL_COLUMNS = [('n', 'n'), ('r@1', 'R@1'), ('r@3', 'R@3'), ('r@5', 'R@5'), ('r@10', 'R@10'),
                     ('r@ctx', 'R@ctx'), ('all@ctx', 'all@ctx'), ('ans@ctx', 'ans@ctx'), ('passages', 'passages')]
ANSWER_COLUMNS = [('n', 'n'), ('match', 'answer'), ('f1', 'F1'), ('cite_ok', 'cites ok'), ('cited_gold', 'cites gold'),
                  ('shown_r', 'shown R'), ('calls', 'tool calls'), ('forced', 'forced')]


def _fmt(value, key: str) -> str:
    if value != value:     # NaN: no row had this field
        return '–'
    if key == 'n':
        return str(int(value))
    if key in ('passages', 'calls'):
        return f'{value:.1f}'
    return f'{value:.2f}'


def _table(rows_by_system: Dict[str, List[Dict]], columns, subsets: Sequence[str]) -> List[str]:
    header = ['system', 'subset'] + [label for _, label in columns] + ['p50 s', 'p95 s']
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    fields = [key for key, _ in columns if key != 'n']
    for system, rows in rows_by_system.items():
        for name in ('all',) + tuple(subsets):
            part = rows if name == 'all' else [r for r in rows if r['subset'] == name]
            if not part:
                continue
            stats = metrics.summarize(part, fields)
            lat = metrics.latency(part, 'time')
            cells = [system if name == 'all' else '', name] + [_fmt(stats[k], k) for k, _ in columns]
            cells += [f"{lat['p50']:.2f}", f"{lat['p95']:.2f}"]
            lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


def _pairwise(rows_by_system: Dict[str, List[Dict]], key: str) -> List[str]:
    """Per-question wins/losses of each system against the first one (the baseline), with McNemar's p"""
    systems = list(rows_by_system)
    if len(systems) < 2:
        return []
    base = {r['id']: r for r in rows_by_system[systems[0]]}
    lines = []
    for system in systems[1:]:
        better = worse = 0
        for r in rows_by_system[system]:
            b = base.get(r['id'])
            if b is None or r.get(key) is None or b.get(key) is None:
                continue
            better += r[key] > b[key]
            worse += r[key] < b[key]
        p = metrics.mcnemar_p(better, worse)
        lines.append(f"- {system} vs {systems[0]} on {key}: better on {better} question(s), worse on {worse} "
                     f"(McNemar p = {p:.3g})")
    return lines


def _handwritten(rows_by_system: Dict[str, List[Dict]], key: str) -> List[str]:
    systems = list(rows_by_system)
    by_id: Dict[str, Dict[str, Dict]] = {}
    for system, rows in rows_by_system.items():
        for r in rows:
            if r['subset'] == 'handwritten':
                by_id.setdefault(r['id'], {})[system] = r
    if not by_id:
        return []
    lines = ['| question | tag | ' + ' | '.join(systems) + ' |', '|' + '---|' * (len(systems) + 2)]
    for rows in by_id.values():
        first = next(iter(rows.values()))
        cells = []
        for s in systems:
            v = rows.get(s, {}).get(key)
            cells.append('–' if v is None else ('✓' if v >= 1 else ('½' if v > 0 else '✗')))
        lines.append(f"| {first['question']} | {first['tag']} | " + ' | '.join(cells) + ' |')
    return lines


# report order; the first system present is the baseline of the pairwise comparisons
SYSTEM_ORDER = ('v1', 'fulltext', 'hybrid', 'chat-fulltext', 'chat-hybrid', 'oneshot-fulltext', 'oneshot-hybrid')


def load_results(out: Path, stage: str) -> Dict[str, List[Dict]]:
    found = {path.stem[len(stage) + 1:]: list(read_jsonl(path)) for path in out.glob(f'{stage}-*.jsonl')}
    rank = {name: i for i, name in enumerate(SYSTEM_ORDER)}
    return {name: found[name] for name in sorted(found, key=lambda n: (rank.get(n, len(rank)), n))}


def write_report(out: Path) -> str:
    meta = json.loads((out / 'meta.json').read_text()) if (out / 'meta.json').exists() else {}
    lines = [f"# Eval results: {out.name}", '']
    for key, value in meta.items():
        lines.append(f"- {key}: {value}")
    subsets = ('handwritten', 'hotpotqa-bridge', 'hotpotqa-comparison', 'popqa')
    retrieval = load_results(out, 'retrieval')
    if retrieval:
        lines += ['', '## Retrieval', '',
                  'R@k: gold articles among the first k passages (fraction of golds for multi-hop questions, '
                  'any gold otherwise). R@ctx: over everything returned. all@ctx: every gold article returned. '
                  'ans@ctx: a gold answer string occurs in the returned text (questions with a non-yes/no answer). '
                  'v1 returns whole articles, so its R@k counts articles, not passages.', '']
        lines += _table(retrieval, RETRIEVAL_COLUMNS, subsets)
        lines += [''] + _pairwise(retrieval, 'r@ctx')
        errors = sum(1 for rows in retrieval.values() for r in rows if r.get('error'))
        if errors:
            lines.append(f"- {errors} retrieval error(s); see the jsonl files")
        lines += ['', '### Hand-written questions (R@ctx: ✓ found, ½ some golds, ✗ none)', '']
        lines += _handwritten(retrieval, 'r@ctx')
    answers = load_results(out, 'answer')
    if answers:
        lines += ['', '## Answers', '',
                  'answer: a gold alias occurs in the answer (yes/no: in the first sentence). F1: first sentence '
                  'vs best gold alias. cites ok: at least one [n] and none unresolvable. cites gold: a cited '
                  'passage is from a gold article. shown R: gold recall over every passage the model saw. '
                  'p50/p95: seconds per question, end to end.', '']
        lines += _table(answers, ANSWER_COLUMNS, subsets)
        lines += [''] + _pairwise(answers, 'match')
        first = [r['first_token'] for rows in answers.values() for r in rows if r.get('first_token')]
        if first:
            lines.append(f"- first answer text (chat systems): p50 {metrics.latency([{'t': v} for v in first], 't')['p50']:.1f}s")
        errors = sum(1 for rows in answers.values() for r in rows if r.get('error'))
        if errors:
            lines.append(f"- {errors} answer error(s); see the jsonl files")
        lines += ['', '### Hand-written questions (answer: ✓ match, ✗ no match, – no short answer)', '']
        lines += _handwritten(answers, 'match')
    text = '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(text, encoding='utf-8')
    return text


# ------------------------------------------------------------------ main

def _run_stage(args, stage: str, systems: Sequence[str]) -> int:
    questions = load_questions(args.questions, args.source, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = Env(args)
    meta_path = out / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else env.describe()
    meta[f'{stage}_systems'] = ', '.join(systems)
    if stage == 'retrieval':
        meta['k'] = args.k
    else:
        meta['model'] = env.answer_model()
        meta['max_rounds'] = args.max_rounds
    meta_path.write_text(json.dumps(meta, indent=1))
    log(f"{len(questions)} question(s) from {args.questions}; results in {out}")
    warm_document_frequencies(env.store, questions)

    for system in systems:
        path = out / f'{stage}-{system}.jsonl'
        done = {r['id'] for r in read_jsonl(path)} if path.exists() else set()
        todo = [q for q in questions if q['id'] not in done]
        if not todo:
            log(f"[{system}] all {len(questions)} done")
            continue
        log(f"[{system}] {len(todo)} question(s){f', {len(done)} already done' if done else ''}")
        started = time.perf_counter()
        rows = run_retrieval(env, system, todo, args.k) if stage == 'retrieval' else run_answer(env, system, todo)
        with open(path, 'a', encoding='utf-8') as f:
            for i, row in enumerate(rows, 1):
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                score = row.get('r@ctx') if stage == 'retrieval' else row.get('match')
                mark = '!' if row.get('error') else ('-' if score is None else ('✓' if score >= 1 else '✗'))
                log(f"  {i:>3}/{len(todo)} {mark} {row['time']:5.1f}s  {row['question'][:70]}")
        log(f"[{system}] done in {time.perf_counter() - started:.0f}s")
        write_report(out)
    print(write_report(out))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Offline Wikipedia RAG eval (see docs/EVALUATION.md)')
    sub = parser.add_subparsers(dest='stage', required=True)

    def common(p, systems, default):
        p.add_argument('--systems', default=default,
                       help=f"comma-separated, from: {', '.join(systems)} (default: {default})")
        p.add_argument('--questions', default=str(QUESTIONS), help='question set (jsonl)')
        p.add_argument('--source', default=None, help='only these sources: handwritten,hotpotqa,popqa')
        p.add_argument('--limit', type=int, default=None, help='first N questions only')
        p.add_argument('--out', default=None, help='results directory (default eval/results/<date-time>); '
                                                   'reuse one to resume or add systems')
        p.add_argument('--zim', default=None)
        p.add_argument('--dense-index', default=None)
        p.add_argument('--model', default=None, help='answer model (default: auto-detect, as the app)')
        p.add_argument('--selection-model', default=None, help='v1 selection model (default: auto-detect)')
        p.add_argument('--kiwix-url', default='http://localhost:8080')

    r = sub.add_parser('retrieval', help='retrieval only: gold article recall, answer-in-context, latency')
    common(r, RETRIEVAL_SYSTEMS, 'fulltext,hybrid')
    r.add_argument('--k', type=int, default=10, help='passages per question (default 10)')
    a = sub.add_parser('answer', help='end to end: answer match, citations, latency')
    common(a, ANSWER_SYSTEMS, 'chat-hybrid')
    a.add_argument('--max-rounds', type=int, default=2)
    rep = sub.add_parser('report', help='rewrite report.md from a results directory')
    rep.add_argument('out')
    args = parser.parse_args(argv)
    if args.stage != 'report':
        args.source = args.source.split(',') if args.source else None
        args.systems = [s.strip() for s in args.systems.split(',') if s.strip()]
        allowed = RETRIEVAL_SYSTEMS if args.stage == 'retrieval' else ANSWER_SYSTEMS
        unknown = [s for s in args.systems if s not in allowed]
        if unknown:
            parser.error(f"unknown system(s) for {args.stage}: {', '.join(unknown)}")
        if args.out is None:
            args.out = str(EVAL_DIR / 'results' / datetime.now().strftime('%Y%m%d-%H%M'))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.stage == 'report':
        print(write_report(Path(args.out)))
        return 0
    return _run_stage(args, args.stage, args.systems)


if __name__ == '__main__':
    sys.exit(main())
