"""Command-line interface: one question (--question) or an interactive chat."""

import argparse
import sys
from typing import Dict, List, Optional

import ollama

from retrieval import ZimStore

from . import llm
from .agent import TurnResult, WikiChat
from .kiwix import KiwixServer, find_kiwix_binary
from .oneshot import OneShotRAG
from .zimfiles import resolve_zim_path

RULE = "=" * 70
INDENT = "   "
QUIT_WORDS = ('quit', 'exit', 'q')
RESET_WORDS = ('/reset', 'reset', '/new')


class TerminalPrinter:
    """Prints agent events: tool calls as they happen, the answer as it streams"""

    def __init__(self):
        self._at_line_start = True

    def __call__(self, kind: str, data: Dict) -> None:
        if kind == 'tool':
            args = data['arguments']
            if data['name'] == 'search_wikipedia':
                what = f'search "{args.get("query", "")}"'
            elif data['name'] == 'read_section':
                section = args.get('section')
                what = f'read "{args.get("title", "")}"' + (f' > "{section}"' if section else '')
            else:
                what = data['name']
            note = " (model answered without searching)" if data.get('forced') else ""
            print(f"  🔎 {what}{note}", end='', flush=True)
        elif kind == 'tool_result':
            print(f" → {data['shown']} passage(s), {data['new']} new")
        elif kind == 'answer':
            print(f"\n{RULE}\n📖 Answer:\n")
            self._at_line_start = True
        elif kind == 'token':
            self._write(data['text'])
        elif kind == 'discard':
            print("\n  (that was not the answer; the model is searching again)")
            self._at_line_start = True
        elif kind == 'warning':
            print(f"\n  ⚠ {data['text']}")

    def _write(self, text: str) -> None:
        out = []
        for ch in text:
            if self._at_line_start and ch != '\n':
                out.append(INDENT)
                self._at_line_start = False
            out.append(ch)
            if ch == '\n':
                self._at_line_start = True
        print(''.join(out), end='', flush=True)


def _print_cited(result: TurnResult) -> None:
    print("\n\n" + "-" * 70)
    citations = result.citations
    if citations:
        linked = any(c.url for c in citations)
        print("📚 Sources (click to open):" if linked else "📚 Sources:")
        for c in citations:
            print(f"   [{c.number}] {c.label}")
            if c.url:
                print(f"       {c.url}")
    else:
        print("⚠ The answer cites no passages.")
    if result.invalid_citations:
        print(f"⚠ Citations that match no retrieved passage: {', '.join(f'[{n}]' for n in result.invalid_citations)}")
    first = f", first answer text after {result.first_token:.1f}s" if result.first_token else ""
    tokens = f", largest prompt {max(result.prompt_tokens)} tokens" if result.prompt_tokens else ""
    print(f"⏱️  {result.time:.1f}s{first}; {len(result.calls)} tool call(s), {len(result.shown)} passage(s) shown{tokens}")
    print(RULE)


def _print_oneshot(result: Dict) -> None:
    print("\n" + RULE)
    print("📖 Answer:\n")
    for line in result['answer'].split('\n'):
        if line.strip():
            if not line.strip().lower().startswith('sources:'):
                print(f"{INDENT}{line}")
        else:
            print()
    print("\n" + "-" * 70)
    sources = result['sources']
    linked = any(s.get('url') for s in sources)
    print("📚 Sources (click to open):" if linked else "📚 Sources:")
    for idx, s in enumerate(sources, 1):
        print(f"   [{idx}] {s.get('label', s['title'])}")
        if s.get('url'):
            print(f"       {s['url']}")
    print(RULE)


def _banner(model: str, source: str, chat: bool) -> None:
    print("\n" + RULE)
    print(" 🌐 Offline Wikipedia AI Assistant")
    print(RULE)
    print(f" 🤖 Model: {model}")
    print(f" 📚 Wikipedia: Local ({source})")
    if chat:
        print(" 💡 Follow-up questions use the conversation; '/reset' starts over, 'quit' exits")
    else:
        print(" 💡 Each question is answered on its own; 'quit' exits")
    print(RULE + "\n")


def interactive(app, source: str, max_results: Optional[int] = None) -> None:
    chat = isinstance(app, WikiChat)
    _banner(app.model_name, source, chat)
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in QUIT_WORDS:
                print("\n👋 Goodbye!")
                break
            if not question:
                continue
            if chat and question.lower() in RESET_WORDS:
                app.reset()
                print("🧹 Conversation cleared.")
                continue
            ask(app, question, max_results)
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def ask(app, question: str, max_results: Optional[int] = None) -> None:
    print(f"\n🔍 {question}")
    if isinstance(app, WikiChat):
        result = app.ask(question, k=max_results)
        _print_cited(result)
    else:
        _print_oneshot(app.query_with_rag(question, max_results=max_results))


def _check_ollama_running() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False


def build_app(args):
    """Set up the ZIM, kiwix-serve and models; returns (app, source description)"""
    zim_path = resolve_zim_path(args.zim)
    available = llm.list_models()
    kiwix = KiwixServer(args.kiwix_url, zim_path)

    if args.retrieval == 'kiwix':
        if not kiwix.connect(not args.no_auto_start):
            raise Exception(f"Could not connect to or start Kiwix server at {kiwix.url}")
        print(f"✓ Wikipedia book: {kiwix.content_base}")
        selection = llm.detect_selection_model(args.selection_model, available)
        print(f"✓ Selection model: {selection}")
        model = llm.detect_summarization_model(args.model, available)
        print(f"✓ Summarization model: {model}")
        return OneShotRAG(model, 'kiwix', kiwix=kiwix, selection_model=selection), kiwix.url

    if zim_path is None:
        raise Exception("No Wikipedia ZIM file found. Download one with scripts/setup_full_offline_wikipedia.sh, "
                        "or pass --zim PATH / set WIKI_ZIM")
    store = ZimStore(zim_path)
    index_note = "full-text index" if store.has_fulltext else "no full-text index, title search only"
    print(f"✓ Wikipedia ZIM: {zim_path.name} (dated {store.date or 'unknown'}, {index_note})")
    # kiwix-serve is optional here: it only makes source links clickable
    if kiwix.connect(not args.no_auto_start and find_kiwix_binary() is not None):
        print(f"✓ Wikipedia book: {kiwix.content_base}")
    else:
        kiwix = None
        print("ℹ kiwix-serve not running; sources are listed as 'Title > Section'")

    model = llm.detect_summarization_model(args.model, available)
    if args.no_tools or not llm.supports_tools(model):
        if not args.no_tools:
            print(f"ℹ {model} does not support tool calling; retrieving once per question")
        print(f"✓ Model: {model} (one-shot)")
        return OneShotRAG(model, 'zim', store=store, kiwix=kiwix), zim_path.name
    print(f"✓ Model: {model} (searches with tools, up to {args.max_rounds} round(s))")
    app = WikiChat(store, model, kiwix=kiwix, max_rounds=args.max_rounds,
                   history_turns=args.history_turns, on_event=TerminalPrinter())
    return app, zim_path.name


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description='Chat with offline English Wikipedia (a Kiwix ZIM file) through local Ollama models',
        epilog='kiwix-serve is started automatically when installed (for clickable source links).'
    )
    parser.add_argument('--question', type=str,
                        help='Single question (otherwise interactive chat)')
    parser.add_argument('--zim', type=str, default=None,
                        help='Wikipedia ZIM file (default: $WIKI_ZIM, else newest complete *.zim in ~/wikipedia-offline)')
    parser.add_argument('--model', type=str, default=None,
                        help='Answer model (default: auto-detect, e.g. gemma4, qwen3.6, qwen3, llama3.1)')
    parser.add_argument('--max-results', type=int, default=None,
                        help='Passages per search (zim) or articles (kiwix); default by question complexity')
    parser.add_argument('--max-rounds', type=int, default=2,
                        help='Rounds of tool calls before the model must answer (default: 2)')
    parser.add_argument('--history-turns', type=int, default=4,
                        help='Earlier questions and answers kept for follow-ups (default: 4; 0 disables)')
    parser.add_argument('--no-tools', action='store_true',
                        help='Retrieve once for the question and answer in one call (no tool loop, no history)')
    parser.add_argument('--retrieval', choices=['zim', 'kiwix'], default='zim',
                        help='zim: read the ZIM directly with libzim (default); kiwix: v1 pipeline over kiwix-serve')
    parser.add_argument('--selection-model', type=str, default=None,
                        help='Article selection model, --retrieval kiwix only (default: auto-detect)')
    parser.add_argument('--kiwix-url', type=str, default='http://localhost:8080',
                        help='Kiwix server URL')
    parser.add_argument('--no-auto-start', action='store_true',
                        help='Do not automatically start Kiwix server')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        print("🔍 Checking dependencies...")
        if not _check_ollama_running():
            print("❌ Ollama is not running")
            print("   Start it with: ollama serve")
            print("   Or install from: https://ollama.com")
            return 1
        print("✓ Ollama is running\n")

        app, source = build_app(args)
        if args.question:
            ask(app, args.question, args.max_results)
        else:
            interactive(app, source, args.max_results)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
