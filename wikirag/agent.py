"""
Chat over the ZIM with a short tool loop.

Each turn: the model gets search_wikipedia/read_section and must search at
least once (if it answers without searching, a search for the raw question is
injected and it is asked again). After max_rounds of tool calls it answers
without tools. The answer is streamed; [n] markers refer to a session-wide
citation table. History keeps the last turns' questions and answers only, plus
the labels of the passages they cited; passage text is not carried over.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from retrieval import ZimStore

from . import llm
from .citations import Citation, CitationTable, ReferenceSectionFilter, cited_numbers, strip_reference_section
from .kiwix import KiwixServer
from .questions import passage_budget
from .tools import MIN_K, TOOLS, WikiTools

MAX_CALLS_PER_ROUND = 3
# Text held back before streaming, so a short preamble before a tool call
# ("Let me search for that.") is not printed as if it were the answer
STREAM_HOLDBACK_CHARS = 160

SYSTEM_PROMPT = """You answer questions using an offline copy of English Wikipedia{dated}. You have two tools:
- search_wikipedia(query): returns numbered passages "[n] Article > Section".
- read_section(title, section): reads one section of an article (omit section for the introduction).

Rules:
1. Always call search_wikipedia before answering, even if you think you know the answer. Search with names and specific terms, not the whole question. For a follow-up question, put the subject from the conversation into the query (e.g. "NASA founding", not "when was it founded").
2. If the passages do not contain the answer, search again with different words (the name of the person, event or place involved) or read a section. You have at most {rounds} rounds of tool calls.
3. Answer only from the passages. After every sentence with a fact from a passage, cite it with its number, like [2] or [1][4]. Only cite numbers you have been shown. If the passages do not answer the question, say so plainly.
4. The first sentence answers the question directly. Then add the supporting details that matter, in short paragraphs. No headings, and no References or Sources list at the end.{cutoff}"""

ANSWER_OPTIONS = {
    'num_predict': 1500,
    'temperature': 0.3,   # factual, citation-heavy answers; also steadier tool use
    'top_p': 0.9,
}

NO_ANSWER = "No answer was generated. Please try again."

Event = Callable[[str, Dict], None]


@dataclass
class ToolCallRecord:
    name: str
    arguments: Dict
    numbers: List[int]
    new: int
    forced: bool = False


@dataclass
class TurnResult:
    question: str
    answer: str
    citations: List[Citation]                  # passages cited in the answer, by number
    shown: List[int]                           # every passage number shown to the model this turn
    invalid_citations: List[int]               # [n] in the answer that match no passage
    calls: List[ToolCallRecord] = field(default_factory=list)
    model: str = ''
    time: float = 0.0
    first_token: Optional[float] = None        # seconds until the first answer text was streamed
    prompt_tokens: List[int] = field(default_factory=list)   # per model call

    @property
    def forced_search(self) -> bool:
        return any(c.forced for c in self.calls)


def _arguments(raw) -> Dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return dict(raw or {})


class WikiChat:
    """A chat session over one ZIM with one tool-calling model"""

    def __init__(self, store: ZimStore, model: str, kiwix: Optional[KiwixServer] = None,
                 max_rounds: int = 2, history_turns: int = 4, on_event: Optional[Event] = None):
        self.store = store
        self.model_name = model
        self.max_rounds = max(1, max_rounds)
        self.history_turns = max(0, history_turns)
        self.on_event = on_event or (lambda kind, data: None)
        url_for = (lambda c: kiwix.path_url(c.path, c.anchor)) if kiwix and kiwix.content_base else None
        self.table = CitationTable(url_for)
        self.tools = WikiTools(store, self.table)
        self.history: List[Tuple[str, str, List[int]]] = []   # (question, answer, cited numbers)

    def reset(self) -> None:
        """Forget the conversation (citation numbers start again at 1)"""
        self.history.clear()
        self.table = CitationTable(self.table._url_for)
        self.tools = WikiTools(self.store, self.table)

    # -- prompt

    def _system_prompt(self) -> str:
        date = self.store.date
        prompt = SYSTEM_PROMPT.format(
            dated=f" (snapshot dated {date})" if date else '',
            rounds=self.max_rounds,
            cutoff=f"\n5. Events after {date} are not in this snapshot; say so if asked about them." if date else '',
        )
        cited = sorted({n for _, _, numbers in self._recent_history() for n in numbers if n in self.table})
        if cited:
            labels = '; '.join(f"[{n}] {self.table[n].label}" for n in cited)
            prompt += ("\n\nPassages cited in earlier answers (their text is no longer shown; search again "
                       f"for details): {labels}")
        return prompt

    def _recent_history(self):
        return self.history[-self.history_turns:] if self.history_turns else []

    def _messages(self, question: str) -> List[Dict]:
        messages = [{'role': 'system', 'content': self._system_prompt()}]
        for q, a, _ in self._recent_history():
            messages += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
        messages.append({'role': 'user', 'content': question})
        return messages

    def _forced_query(self, question: str) -> str:
        """
        Query for an injected search. A follow-up ("and when was it founded?")
        usually lacks its subject, so the previous question is prepended.
        """
        recent = self._recent_history()
        return f"{recent[-1][0]} {question}" if recent else question

    # -- one model call

    def _call(self, messages: List[Dict], tools: Optional[Sequence[Dict]], stream_answer: bool,
              stage: str, filt: ReferenceSectionFilter, started: float, timing: Dict):
        """
        Stream one model response. Returns (content, tool_calls, streamed).
        Text is passed to the 'token' event only when stream_answer is set and
        no tool call has appeared within the first STREAM_HOLDBACK_CHARS.
        """
        content, calls, streamed = [], [], False
        # before the first search, a long reply without a tool call is an answer from
        # memory that will be discarded; stop it early instead of generating it all
        abort_unsearched = tools is not None and not stream_answer

        def emit(text):
            text = filt.feed(text)
            if text:
                if timing.get('first_token') is None:
                    timing['first_token'] = time.perf_counter() - started
                self.on_event('token', {'text': text})

        def usage(prompt_tokens, num_ctx):
            timing.setdefault('prompt_tokens', []).append(prompt_tokens)
            if prompt_tokens >= num_ctx - ANSWER_OPTIONS['num_predict']:
                self.on_event('warning', {'text': f"{stage} prompt may have been truncated to fit num_ctx {num_ctx}"})

        stream = llm.chat_stream(self.model_name, messages, ANSWER_OPTIONS, stage, tools=tools, on_usage=usage)
        for chunk in stream:
            message = chunk['message']
            for call in message.get('tool_calls') or []:
                function = call['function']
                calls.append({'name': function['name'], 'arguments': _arguments(function.get('arguments'))})
            piece = message.get('content') or ''
            if not piece:
                continue
            content.append(piece)
            if abort_unsearched and not calls and len(''.join(content).strip()) > STREAM_HOLDBACK_CHARS:
                stream.close()
                break
            if streamed:
                emit(piece)
            elif stream_answer and not calls and len(''.join(content).strip()) > STREAM_HOLDBACK_CHARS:
                streamed = True
                self.on_event('answer', {})
                emit(''.join(content).lstrip())
        return ''.join(content), calls, streamed

    # -- turn

    def _run_tools(self, messages: List[Dict], calls: List[Dict], content: str, budget: int,
                   records: List[ToolCallRecord], forced: bool = False) -> None:
        """Append the assistant tool-call message and one tool message per call"""
        messages.append({'role': 'assistant', 'content': content,
                         'tool_calls': [{'function': {'name': c['name'], 'arguments': c['arguments']}} for c in calls]})
        executed = calls[:MAX_CALLS_PER_ROUND]
        k = max(MIN_K, budget // len(executed))
        for i, call in enumerate(calls):
            if i >= MAX_CALLS_PER_ROUND:
                messages.append({'role': 'tool', 'tool_name': call['name'],
                                 'content': f'Skipped: at most {MAX_CALLS_PER_ROUND} tool calls per round.'})
                continue
            self.on_event('tool', {'name': call['name'], 'arguments': call['arguments'], 'forced': forced})
            result = self.tools.run(call['name'], call['arguments'], k)
            records.append(ToolCallRecord(call['name'], call['arguments'], result.numbers, result.new, forced))
            self.on_event('tool_result', {'name': call['name'], 'shown': len(result.numbers), 'new': result.new})
            messages.append({'role': 'tool', 'tool_name': call['name'], 'content': result.content})

    def ask(self, question: str, k: Optional[int] = None) -> TurnResult:
        started = time.perf_counter()
        budget = k or passage_budget(question)
        messages = self._messages(question)
        records: List[ToolCallRecord] = []
        filt = ReferenceSectionFilter()
        timing: Dict = {}
        self.tools.begin_turn()
        rounds, final_without_tools = 0, False

        while True:
            searched = bool(records)
            use_tools = rounds < self.max_rounds and not final_without_tools
            stage = f"Round {rounds + 1}" if use_tools else "Answer"
            content, calls, streamed = self._call(
                messages, TOOLS if use_tools else None,
                stream_answer=searched, stage=stage, filt=filt, started=started, timing=timing)

            if calls and use_tools:
                if streamed:   # text already printed turned out to be a preamble
                    self.on_event('discard', {})
                    filt = ReferenceSectionFilter()
                    timing.pop('first_token', None)
                rounds += 1
                self._run_tools(messages, calls, content, budget, records)
                continue
            if not searched:
                # answered (or produced nothing) without searching: search for the question and ask again
                rounds += 1
                forced = {'name': 'search_wikipedia', 'arguments': {'query': self._forced_query(question)}}
                self._run_tools(messages, [forced], '', budget, records, forced=True)
                continue
            if not content.strip() and use_tools:
                final_without_tools = True   # empty reply; ask once more without tools
                continue
            if not streamed:
                self.on_event('answer', {})
                text = filt.feed(content.strip() or NO_ANSWER)
                if text:
                    timing.setdefault('first_token', time.perf_counter() - started)
                    self.on_event('token', {'text': text})
            tail = filt.flush()
            if tail:
                self.on_event('token', {'text': tail})
            break

        answer = strip_reference_section(content.strip()) or NO_ANSWER
        numbers = cited_numbers(answer)
        shown = list(dict.fromkeys(n for r in records for n in r.numbers))
        invalid = [n for n in numbers if n not in self.table]
        if self.history_turns:
            self.history.append((question, answer, [n for n in numbers if n in self.table]))
            del self.history[:-self.history_turns]
        return TurnResult(
            question=question, answer=answer, citations=self.table.citations(numbers), shown=shown,
            invalid_citations=invalid, calls=records, model=self.model_name,
            time=time.perf_counter() - started, first_token=timing.get('first_token'),
            prompt_tokens=timing.get('prompt_tokens', []),
        )
