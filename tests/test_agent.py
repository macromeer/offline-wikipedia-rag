"""Tests for the chat agent (wikirag/agent.py), its tools and citation handling"""

import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

import ollama
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import ZimStore
from tests.fakes import FakeStore, _article
from wikirag.agent import MAX_CALLS_PER_ROUND, NO_ANSWER, WikiChat
from wikirag.citations import CitationTable, ReferenceSectionFilter, cited_numbers, strip_reference_section
from wikirag.cli import TerminalPrinter
from wikirag.tools import WikiTools


@pytest.fixture
def store():
    articles = {
        'NASA': _article(
            'NASA',
            ('', 'NASA is the United States space agency.'),
            ('History', 'NASA was established in 1958 by the National Aeronautics and Space Act.'),
            ('Strategic plan', 'NASA goals: expand knowledge, explore the Moon and Mars.'),
            ('History > Apollo', 'Apollo 11 landed on the Moon in 1969.'),
        ),
        'Apollo_11': _article(
            'Apollo 11',
            ('', 'Apollo 11 was the first crewed Moon landing.'),
            ('Lunar surface operations', 'Neil Armstrong stepped onto the surface first.'),
        ),
        'Mercury': _article('Mercury', ('', 'Mercury may refer to:'), disambig=True,
                            links=['Mercury_(planet)', 'Mercury_(element)']),
    }
    titles = {'NASA': 'NASA', 'Apollo 11': 'Apollo_11', 'Mercury': 'Mercury'}
    return FakeStore(articles, titles, ['NASA', 'Apollo_11'], dfs={'nasa': 5, 'moon': 50, 'founded': 500})


# ---------------------------------------------------------------- scripted Ollama

def _chunk(content='', tool_calls=None, done=False):
    calls = [ollama.Message.ToolCall(function=ollama.Message.ToolCall.Function(name=n, arguments=a))
             for n, a in (tool_calls or [])]
    return ollama.ChatResponse(model='m', done=done, prompt_eval_count=100 if done else None,
                               message=ollama.Message(role='assistant', content=content, tool_calls=calls or None))


class ScriptedOllama:
    """
    Stands in for ollama.chat(stream=True). Each reply is a string (streamed in
    small pieces), a list of (tool name, arguments), or a (text, calls) tuple
    for text followed by tool calls in the same response. Records every request.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.closed_early = 0

    def __call__(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        return self._stream(reply)

    def _stream(self, reply):
        finished = False
        text, calls = reply if isinstance(reply, tuple) else (reply, None) if isinstance(reply, str) else ('', reply)
        try:
            for i in range(0, len(text), 7):
                yield _chunk(text[i:i + 7])
            finished = True
            yield _chunk(tool_calls=calls, done=True)
        finally:
            if not finished:
                self.closed_early += 1

    def messages(self, i):
        return self.requests[i]['messages']


def _chat(store, script, **kwargs):
    events = []
    chat = WikiChat(store, 'm', on_event=lambda kind, data: events.append((kind, data)), **kwargs)
    return chat, events


def _streamed_text(events):
    return ''.join(d['text'] for k, d in events if k == 'token')


SEARCH_NASA = [('search_wikipedia', {'query': 'NASA founded'})]


class TestToolLoop:
    def test_search_then_answer(self, store):
        script = ScriptedOllama(SEARCH_NASA, 'NASA was established in 1958 [2].')
        chat, events = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            result = chat.ask('When was NASA founded?')
        assert result.answer == 'NASA was established in 1958 [2].'
        assert [c.number for c in result.citations] == [2]
        assert result.invalid_citations == [] and not result.forced_search
        tool_message = script.messages(1)[-1]
        assert tool_message['role'] == 'tool' and tool_message['tool_name'] == 'search_wikipedia'
        assert '[1] NASA\n' in tool_message['content']           # lead of the title hit comes first
        assert f"[2] {result.citations[0].label}\n" in tool_message['content']
        assert result.shown == list(range(1, len(result.shown) + 1))
        assert script.requests[0]['tools'] and script.requests[1]['tools']
        assert script.requests[0]['think'] is False
        assert _streamed_text(events) == result.answer

    def test_answer_without_search_is_discarded_and_search_forced(self, store):
        from_memory = 'NASA was founded in 1958, as everyone knows. ' * 10
        script = ScriptedOllama(from_memory, 'NASA was established in 1958 [2].')
        chat, events = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            result = chat.ask('When was NASA founded?')
        assert result.forced_search
        assert result.calls[0].arguments == {'query': 'When was NASA founded?'}
        assert script.closed_early == 1                 # generation stopped once it was clearly an answer
        assert 'everyone knows' not in _streamed_text(events)
        assert all('everyone knows' not in (m.get('content') or '') for m in script.messages(1))
        assert result.answer == 'NASA was established in 1958 [2].'

    def test_round_cap_then_answer_without_tools(self, store):
        script = ScriptedOllama(SEARCH_NASA, [('search_wikipedia', {'query': 'Apollo 11'})], 'Answer [1].')
        chat, _ = _chat(store, script, max_rounds=2)
        with patch('wikirag.llm.ollama.chat', script):
            result = chat.ask('When was NASA founded?')
        assert len(result.calls) == 2
        assert script.requests[2]['tools'] is None
        assert result.answer == 'Answer [1].'

    def test_preamble_before_tool_call_is_withdrawn(self, store):
        preamble = 'I will look this up in more detail to be sure. ' * 5
        script = ScriptedOllama(SEARCH_NASA, (preamble, [('read_section', {'title': 'Apollo 11'})]),
                                'Apollo 11 landed in 1969 [4].')
        chat, events = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            result = chat.ask('When did Apollo 11 land?')
        kinds = [k for k, _ in events]
        assert 'discard' in kinds and kinds.index('discard') > kinds.index('answer')
        assert result.answer == 'Apollo 11 landed in 1969 [4].'
        assert [c.name for c in result.calls] == ['search_wikipedia', 'read_section']

    def test_short_preamble_is_never_printed(self, store):
        script = ScriptedOllama(SEARCH_NASA, ('Let me check.', [('read_section', {'title': 'Apollo 11'})]),
                                'Apollo 11 landed in 1969 [4].')
        chat, events = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            chat.ask('When did Apollo 11 land?')
        assert 'discard' not in [k for k, _ in events]
        assert _streamed_text(events) == 'Apollo 11 landed in 1969 [4].'

    def test_too_many_calls_in_one_round(self, store):
        calls = [('search_wikipedia', {'query': f'q{i}'}) for i in range(MAX_CALLS_PER_ROUND + 2)]
        script = ScriptedOllama(calls, 'Done [1].')
        chat, _ = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            result = chat.ask('When was NASA founded?')
        assert len(result.calls) == MAX_CALLS_PER_ROUND
        tool_messages = [m for m in script.messages(1) if m['role'] == 'tool']
        assert len(tool_messages) == len(calls)
        assert tool_messages[-1]['content'].startswith('Skipped')

    def test_invalid_citation_reported(self, store):
        script = ScriptedOllama(SEARCH_NASA, 'NASA was established in 1958 [2][9].')
        chat, _ = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            result = chat.ask('When was NASA founded?')
        assert result.invalid_citations == [9]
        assert [c.number for c in result.citations] == [2]

    def test_reference_section_not_streamed(self, store):
        script = ScriptedOllama(SEARCH_NASA, 'NASA was established in 1958 [2].\n\nReferences:\n[2] NASA > History')
        chat, events = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            result = chat.ask('When was NASA founded?')
        assert result.answer == 'NASA was established in 1958 [2].'
        assert 'References' not in _streamed_text(events)

    def test_empty_reply_after_search_retries_without_tools(self, store):
        script = ScriptedOllama(SEARCH_NASA, '', 'NASA dates from 1958 [2].')
        chat, _ = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            result = chat.ask('When was NASA founded?')
        assert script.requests[2]['tools'] is None
        assert result.answer == 'NASA dates from 1958 [2].'

    def test_no_answer_is_reported(self, store):
        script = ScriptedOllama(SEARCH_NASA, '', '')
        chat, events = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            result = chat.ask('When was NASA founded?')
        assert result.answer == NO_ANSWER and _streamed_text(events) == NO_ANSWER

    def test_system_prompt_states_snapshot_date(self, store):
        script = ScriptedOllama(SEARCH_NASA, 'Answer [1].')
        chat, _ = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            chat.ask('When was NASA founded?')
        system = script.messages(0)[0]
        assert system['role'] == 'system' and '2026-06-17' in system['content']
        assert 'at most 2 rounds' in system['content']


class TestHistory:
    def test_follow_up_sees_previous_turn_and_numbers_are_stable(self, store):
        script = ScriptedOllama(
            [('search_wikipedia', {'query': 'NASA'})], 'NASA is the space agency [1].',
            [('search_wikipedia', {'query': 'NASA founded'})], 'It was established in 1958 [2].',
        )
        chat, _ = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            first = chat.ask('What is NASA?')
            second = chat.ask('and when was it founded?')
        messages = script.messages(2)
        assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user']
        assert messages[2]['content'] == first.answer
        assert '[1] NASA' in messages[0]['content']        # cited passages listed, text not carried over
        # NASA lead was shown in turn 1: same number, and its text is shown again in turn 2
        tool_text = script.messages(3)[-1]['content']
        assert '[1] NASA\nNASA is the United States space agency.' in tool_text
        assert [c.number for c in second.citations] == [2]

    def test_follow_up_forced_search_includes_previous_question(self, store):
        script = ScriptedOllama(SEARCH_NASA, 'NASA is the space agency [1].', '', 'In 1958 [2].')
        chat, _ = _chat(store, script)
        with patch('wikirag.llm.ollama.chat', script):
            chat.ask('What is NASA?')
            second = chat.ask('and when was it founded?')
        assert second.calls[0].forced
        assert second.calls[0].arguments['query'] == 'What is NASA? and when was it founded?'

    def test_history_window_and_reset(self, store):
        replies = []
        for i in range(3):
            replies += [SEARCH_NASA, f'Answer {i} [1].']
        script = ScriptedOllama(*replies)
        chat, _ = _chat(store, script, history_turns=1)
        with patch('wikirag.llm.ollama.chat', script):
            for i in range(3):
                chat.ask(f'Question {i}?')
        assert [q for q, _, _ in chat.history] == ['Question 2?']
        roles = [m['role'] for m in script.messages(4)]
        assert roles == ['system', 'user', 'assistant', 'user']      # only one earlier turn
        chat.reset()
        assert chat.history == [] and len(chat.table) == 0


class TestTools:
    @pytest.fixture
    def tools(self, store):
        return WikiTools(store, CitationTable())

    def test_read_lead_and_section(self, tools):
        lead = tools.read_section('Apollo 11')
        assert '[1] Apollo 11\nApollo 11 was the first crewed Moon landing.' in lead.content
        section = tools.read_section('apollo 11', 'lunar surface operations')
        assert section.numbers == [2] and 'Neil Armstrong' in section.content

    def test_read_section_by_substring_and_nested_heading(self, tools):
        result = tools.read_section('NASA', 'apollo')
        assert 'Apollo 11 landed on the Moon in 1969.' in result.content

    def test_missing_section_lists_headings(self, tools):
        result = tools.read_section('NASA', 'Budget')
        assert result.numbers == [] and 'History; Strategic plan; History > Apollo' in result.content

    def test_unknown_title_and_disambiguation(self, tools):
        assert 'No article titled' in tools.read_section('Nonexistent page').content
        content = tools.read_section('Mercury').content
        assert 'disambiguation' in content and 'Mercury (planet)' in content

    def test_repeated_passage_not_resent_within_turn(self, tools):
        tools.search('NASA founded', 4)
        again = tools.read_section('NASA')
        assert again.new == 0 and 'Already shown above: [1] NASA' in again.content
        tools.begin_turn()
        assert 'NASA is the United States space agency.' in tools.read_section('NASA').content

    def test_bad_arguments(self, tools):
        assert tools.run('search_wikipedia', {}, 5).content.startswith('Empty query')
        assert tools.run('delete_everything', {}, 5).content.startswith('Unknown tool')


class TestCitations:
    @pytest.mark.parametrize('text,expected', [
        ('A [1]. B [2][3].', [1, 2, 3]),
        ('A [1, 4] and [2-4].', [1, 4, 2, 3]),
        ('A [2–3]. Not a citation: [x] or [1a].', [2, 3]),
        ('Repeated [1] [1].', [1]),
    ])
    def test_cited_numbers(self, text, expected):
        assert cited_numbers(text) == expected

    def test_strip_reference_section(self):
        assert strip_reference_section('Answer [1].\n\n**Sources:**\n[1] NASA') == 'Answer [1].'
        assert strip_reference_section('Answer about sources of energy [1].') == 'Answer about sources of energy [1].'

    @pytest.mark.parametrize('text,expected', [
        ('Answer [1].\n\nReferences:\n[1] NASA', 'Answer [1].\n\n'),
        ('Answer [1].\n## Sources\n- NASA', 'Answer [1].\n'),
        ('Sources of energy include coal [1].\nMore [2].', 'Sources of energy include coal [1].\nMore [2].'),
        ('Line one.\n\nSecond paragraph [1].', 'Line one.\n\nSecond paragraph [1].'),
        ('Answer.\nSources', 'Answer.\n'),
    ])
    def test_streaming_filter_any_split(self, text, expected):
        rng = random.Random(0)
        for _ in range(20):
            filt, out, i = ReferenceSectionFilter(), [], 0
            while i < len(text):
                step = rng.randint(1, 6)
                out.append(filt.feed(text[i:i + step]))
                i += step
            out.append(filt.flush())
            assert ''.join(out) == expected


class TestTerminalPrinter:
    def test_answer_lines_are_indented(self, capsys):
        printer = TerminalPrinter()
        printer('answer', {})
        printer('token', {'text': 'First li'})
        printer('token', {'text': 'ne.\n\nSecond.'})
        out = capsys.readouterr().out
        assert out.endswith('   First line.\n\n   Second.')


# ---------------------------------------------------------------- real ZIM and Ollama

def _full_zim():
    from wikirag.zimfiles import resolve_zim_path
    try:
        path = resolve_zim_path()
    except (FileNotFoundError, ValueError):
        return None
    return path if path and '_all_' in path.name else None


@pytest.mark.integration
class TestChatOnFullZim:
    """Needs the full English Wikipedia ZIM and Ollama with a tool-calling model (WIKI_TEST_MODEL)"""

    @pytest.fixture(scope='class')
    @classmethod
    def chat(cls):
        from wikirag import llm
        path = _full_zim()
        if not path:
            pytest.skip('no full English Wikipedia ZIM found')
        try:
            available = llm.list_models()
        except Exception:
            available = {}
        if not available:
            pytest.skip('Ollama not running or no models')
        model = os.environ.get('WIKI_TEST_MODEL') or llm.detect_summarization_model(None, available)
        if not llm.supports_tools(model):
            pytest.skip(f'{model} does not support tools')
        return WikiChat(ZimStore(path), model)

    def test_follow_up_uses_history(self, chat):
        first = chat.ask('What is NASA?')
        assert first.citations and not first.invalid_citations
        second = chat.ask('and when was it founded?')
        assert '1958' in second.answer
        assert second.citations and not second.invalid_citations
        assert any('nasa' in (c.arguments.get('query') or '').lower() for c in second.calls)

    def test_simple_question_latency(self, chat):
        chat.reset()
        result = chat.ask('What is the capital of Australia?')
        assert 'Canberra' in result.answer and result.citations
        assert result.time < 15
