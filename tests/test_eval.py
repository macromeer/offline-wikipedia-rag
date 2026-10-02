"""Eval harness: scoring functions and the report, on synthetic rows (no ZIM or model needed)"""

import json

import pytest

from eval import metrics
from eval.dataset import HANDWRITTEN, QUESTIONS, read_jsonl, subset
from eval.run import retrieval_row, write_report


class TestNormalize:
    def test_squad_style(self):
        assert metrics.normalize("The Starry Night!") == "starry night"

    def test_accents_and_citations(self):
        assert metrics.normalize("Café Müller [2][3] opened.") == "cafe muller opened"

    def test_thousands_separators(self):
        assert metrics.normalize("299,792,458 m/s") == metrics.normalize("299792458 m/s")


class TestAnswerMatch:
    def test_alias_in_long_answer(self):
        answer = "The Starry Night was painted by Vincent van Gogh in June 1889 [1]. It shows..."
        assert metrics.answer_match(answer, ["Vincent van Gogh", "van Gogh"])

    def test_whole_tokens_only(self):
        # "Rome" must not match inside "Romeo"
        assert not metrics.answer_match("Romeo and Juliet is set in Verona.", ["Rome"])

    def test_no_golds(self):
        assert not metrics.answer_match("anything", [])

    def test_yes_no_judged_on_first_sentence(self):
        assert metrics.answer_match("Yes, both were American [1]. No other...", ["yes"])
        assert not metrics.answer_match("No, Derrickson is American [1]. Yes, Wood was too.", ["yes"])

    def test_yes_no_hedge_fails(self):
        assert not metrics.answer_match("Yes and no: it depends.", ["no"])


class TestF1:
    def test_first_sentence_only(self):
        answer = "Christopher Nolan directed Inception [1]. Nolan also wrote it."
        f1 = metrics.answer_f1(answer, ["Christopher Nolan"])
        assert 0.5 < f1 < 1.0

    def test_no_overlap(self):
        assert metrics.token_f1("Paris", "Berlin") == 0.0


class TestRecall:
    def test_any(self):
        assert metrics.title_recall(['A', 'B', 'C'], ['C', 'X'], 2, 'any') == 0.0
        assert metrics.title_recall(['A', 'B', 'C'], ['C', 'X'], 3, 'any') == 1.0

    def test_all_is_fraction(self):
        assert metrics.title_recall(['A', 'B', 'C'], ['A', 'X'], None, 'all') == 0.5

    def test_no_golds(self):
        assert metrics.title_recall(['A'], [], None, 'any') == 0.0

    def test_passages_contain_answer_skips_yes_no(self):
        assert metrics.passages_contain_answer(["Canberra is the capital."], ["Canberra"])
        assert not metrics.passages_contain_answer(["yes indeed"], ["yes"])


class TestMcNemar:
    def test_symmetric_and_bounds(self):
        assert metrics.mcnemar_p(0, 0) == 1.0
        assert metrics.mcnemar_p(5, 5) == 1.0
        assert metrics.mcnemar_p(10, 0) == pytest.approx(2 / 1024)
        assert metrics.mcnemar_p(3, 9) == metrics.mcnemar_p(9, 3)


def _question(**kw):
    q = {'id': 'q1', 'source': 'hotpotqa', 'tag': 'bridge', 'question': 'Q?', 'answers': ['Nolan'],
         'gold_titles': ['A', 'B'], 'gold_mode': 'all'}
    q.update(kw)
    return q


class TestRetrievalRow:
    def test_multi_hop(self):
        passages = [{'path': 'A', 'text': 'x'}, {'path': 'A', 'text': 'y'}, {'path': 'C', 'text': 'Nolan'},
                    {'path': 'B', 'text': 'z'}]
        row = retrieval_row(_question(), 'fulltext', ['A', 'B'], passages, 0.1)
        assert row['r@1'] == 0.5
        assert row['r@3'] == 0.5
        assert row['r@5'] == 1.0
        assert row['all@ctx'] == 1.0
        assert row['ans@ctx'] is True
        assert row['articles'] == ['A', 'C', 'B']
        assert row['subset'] == 'hotpotqa-bridge'

    def test_any_mode_all_equals_recall(self):
        q = _question(gold_mode='any', answers=['yes'])
        row = retrieval_row(q, 'hybrid', ['A', 'B'], [{'path': 'B', 'text': 't'}], 0.1)
        assert row['all@ctx'] == row['r@ctx'] == 1.0
        assert row['ans@ctx'] is None          # yes/no answers are not looked for in passages


class TestReport:
    def test_report_tables(self, tmp_path):
        q = _question()
        rows = {
            'fulltext': [retrieval_row(q, 'fulltext', ['A', 'B'], [{'path': 'A', 'text': 't'}], 0.2)],
            'hybrid': [retrieval_row(q, 'hybrid', ['A', 'B'], [{'path': 'A', 'text': 't'}, {'path': 'B', 'text': 'u'}], 0.3)],
        }
        for system, data in rows.items():
            with open(tmp_path / f'retrieval-{system}.jsonl', 'w') as f:
                for row in data:
                    f.write(json.dumps(row) + '\n')
        (tmp_path / 'meta.json').write_text(json.dumps({'zim': 'test.zim'}))
        text = write_report(tmp_path)
        assert '| fulltext | all | 1 | 0.50 |' in text      # one of two golds at rank 1
        assert '| hybrid | all | 1 | 0.50 |' in text
        assert 'hybrid vs fulltext on r@ctx: better on 1 question(s), worse on 0 (McNemar p = 1)' in text
        assert (tmp_path / 'report.md').exists()


class TestQuestionSet:
    @pytest.mark.parametrize('path', [QUESTIONS, HANDWRITTEN])
    def test_schema(self, path):
        rows = list(read_jsonl(path))
        assert rows
        assert len({r['id'] for r in rows}) == len(rows)
        for r in rows:
            assert r['question'].strip()
            assert r['gold_titles'] and r['gold_mode'] in ('any', 'all')
            assert isinstance(r['answers'], list)

    def test_subsets(self):
        names = {subset(r) for r in read_jsonl(QUESTIONS)}
        assert names == {'handwritten', 'hotpotqa-bridge', 'hotpotqa-comparison', 'popqa'}
