"""Ollama model detection and chat calls."""

import json
import re
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import ollama

# Model auto-detection. Entries without a tag match any tag of that family
# (subject to the eligibility rules below); entries with a tag also match exactly.
SELECTION_MODEL_PREFERENCES = [
    'qwen3.6', 'qwen3.5', 'gemma4', 'qwen3',
    'qwen2.5:32b-instruct', 'qwen2.5:32b', 'qwen2.5:14b-instruct', 'qwen2.5:14b', 'qwen2.5:7b-instruct',
    'mistral-small', 'mistral:7b', 'hermes3:8b', 'hermes3', 'llama3.1:8b', 'phi3:medium',
]
SUMMARIZATION_MODEL_PREFERENCES = [
    'gemma4', 'qwen3.6', 'qwen3.5', 'qwen3',
    'llama3.1:8b-instruct', 'llama3.1:8b', 'gemma2:27b', 'gemma2:9b', 'mistral:7b',
    'granite3.1-dense:8b', 'qwen2.5:7b', 'llama3.3:70b', 'llama3.1:70b-instruct', 'llama3.1:70b',
]
MIN_MODEL_PARAMS_B = 4.0
EXCLUDED_MODEL_SUBSTRINGS = ('coder', 'embed', 'rerank', 'deepseek')
EXCLUDED_MODEL_TOKENS = ('r1',)  # matched as a whole name component, e.g. 'foo-r1:7b'

# Context window. Ollama's default (4096) silently truncates the synthesis prompt.
# Rounded to powers of two so repeated calls reuse the loaded model instead of reloading.
MIN_NUM_CTX = 16384
CHARS_PER_TOKEN = 4  # rough estimate; consistency matters more than precision

# {model name: parameter count in billions, or None if unknown}
ModelList = Dict[str, Optional[float]]


def parse_param_billions(text: str) -> Optional[float]:
    """Parse a parameter count like '26b', '1.5b', '751.63M', '8x7b' into billions"""
    match = re.search(r'(?:(\d+)x)?(\d+(?:\.\d+)?)([bm])(?![a-z])', text.lower())
    if not match:
        return None
    experts, value, unit = match.groups()
    billions = float(value) * (int(experts) if experts else 1)
    return billions / 1000 if unit == 'm' else billions


def is_excluded_model(name: str) -> bool:
    lowered = name.lower()
    if any(s in lowered for s in EXCLUDED_MODEL_SUBSTRINGS):
        return True
    components = re.split(r'[-_:./]', lowered)
    return any(t in components for t in EXCLUDED_MODEL_TOKENS)


def _canonical_model_name(name: str) -> str:
    """'llama3.1' and 'llama3.1:latest' name the same model"""
    return name if ':' in name else f"{name}:latest"


def list_models() -> ModelList:
    """Installed Ollama models with parameter sizes from `ollama list` details (for tags like ':latest')"""
    try:
        response = ollama.list()
    except Exception as e:
        print(f"⚠ Could not list models: {e}")
        return {}
    models = {}
    for m in getattr(response, 'models', None) or []:
        details = getattr(m, 'details', None)
        models[m.model] = parse_param_billions(getattr(details, 'parameter_size', None) or '')
    return models


def _as_model_list(available: Union[ModelList, Sequence[str]]) -> ModelList:
    return dict(available) if isinstance(available, dict) else {name: None for name in available}


def model_param_billions(name: str, available: Union[ModelList, Sequence[str]] = ()) -> Optional[float]:
    """Parameter count in billions, from the tag (e.g. ':26b') or from `ollama list` details"""
    tag = name.split(':', 1)[1] if ':' in name else ''
    size = parse_param_billions(tag)
    return size if size is not None else _as_model_list(available).get(name)


def is_eligible_model(name: str, available: Union[ModelList, Sequence[str]] = ()) -> bool:
    """Reject coder/embedding/reranker/reasoning models and models below MIN_MODEL_PARAMS_B"""
    if is_excluded_model(name):
        return False
    size = model_param_billions(name, available)
    return size is None or size >= MIN_MODEL_PARAMS_B


def pick_model(preferences: List[str], available: Union[ModelList, Sequence[str]], role: str) -> str:
    """
    For each preference in order: exact tag match, else an eligible model of the
    same family with another tag (e.g. 'qwen3.6' -> 'qwen3.6:35b'). Then any
    eligible model, then any non-excluded model.
    """
    available = _as_model_list(available)
    canonical = {_canonical_model_name(a): a for a in available}
    for pref in preferences:
        exact = canonical.get(_canonical_model_name(pref))
        if exact:
            return exact
        family = pref.split(':')[0]
        for avail in available:
            if avail.split(':')[0] == family and is_eligible_model(avail, available):
                return avail

    for avail in available:
        if is_eligible_model(avail, available):
            print(f"⚠ Using fallback {role} model: {avail}")
            return avail
    for avail in available:
        if not is_excluded_model(avail):
            print(f"⚠ Using fallback {role} model: {avail} (below {MIN_MODEL_PARAMS_B:g}B, expect weak results)")
            return avail

    raise Exception(f"No suitable Ollama model for {role}. Pull one, e.g.: ollama pull qwen3:8b")


def detect_selection_model(preferred: Optional[str], available: Union[ModelList, Sequence[str]]) -> str:
    """Model for v1 article selection. Reasoning, coder and sub-4B models are never auto-selected."""
    return preferred or pick_model(SELECTION_MODEL_PREFERENCES, available, 'selection')


def detect_summarization_model(preferred: Optional[str], available: Union[ModelList, Sequence[str]]) -> str:
    """Model that searches and writes the answer"""
    return preferred or pick_model(SUMMARIZATION_MODEL_PREFERENCES, available, 'summarization')


def supports_tools(model: str) -> bool:
    """True if the model's template supports tool calling (`ollama show` capabilities)"""
    try:
        capabilities = ollama.show(model).capabilities
    except Exception:
        return False
    return capabilities is None or 'tools' in capabilities


def num_ctx_for(chars: int, num_predict: int) -> int:
    """Smallest power-of-two context (>= MIN_NUM_CTX) that fits a prompt of `chars` characters plus output"""
    needed = chars // CHARS_PER_TOKEN + num_predict + 512
    num_ctx = MIN_NUM_CTX
    while num_ctx < needed:
        num_ctx *= 2
    return num_ctx


def _prompt_chars(messages: Sequence[Dict], tools: Optional[Sequence[Dict]]) -> int:
    chars = sum(len(m.get('content') or '') + len(json.dumps(m.get('tool_calls') or [], default=str))
                for m in messages)
    return chars + (len(json.dumps(tools)) if tools else 0)


def _report_usage(response, num_ctx: int, num_predict: int, stage: str) -> int:
    prompt_tokens = response.get('prompt_eval_count') or 0
    print(f"  📏 {stage}: {prompt_tokens} prompt tokens (num_ctx {num_ctx})")
    if prompt_tokens >= num_ctx - num_predict:
        print(f"  ⚠ {stage} prompt may have been truncated to fit num_ctx {num_ctx}")
    return prompt_tokens


def chat(model: str, messages: Sequence[Dict], options: Dict, stage: str,
         tools: Optional[Sequence[Dict]] = None):
    """
    ollama.chat with num_ctx sized to the prompt and thinking off (thinking
    models otherwise spend num_predict on hidden reasoning and return empty
    content). Logs prompt_eval_count so context truncation is visible.
    """
    num_predict = options.get('num_predict', 0)
    num_ctx = num_ctx_for(_prompt_chars(messages, tools), num_predict)
    response = ollama.chat(model=model, messages=list(messages), tools=tools,
                           options={**options, 'num_ctx': num_ctx}, think=False)
    _report_usage(response, num_ctx, num_predict, stage)
    return response


def chat_stream(model: str, messages: Sequence[Dict], options: Dict, stage: str,
                tools: Optional[Sequence[Dict]] = None,
                on_usage: Optional[Callable[[int, int], None]] = None) -> Iterator:
    """
    Streaming variant of chat(): yields response chunks. After the last one,
    on_usage(prompt_tokens, num_ctx) is called, or usage is printed if not given.
    """
    num_predict = options.get('num_predict', 0)
    num_ctx = num_ctx_for(_prompt_chars(messages, tools), num_predict)
    chunks: Iterable = ollama.chat(model=model, messages=list(messages), tools=tools,
                                   options={**options, 'num_ctx': num_ctx}, think=False, stream=True)
    try:
        for chunk in chunks:
            yield chunk
            if chunk.get('done'):
                if on_usage:
                    on_usage(chunk.get('prompt_eval_count') or 0, num_ctx)
                else:
                    _report_usage(chunk, num_ctx, num_predict, stage)
    finally:
        # closing early drops the HTTP stream, which makes Ollama stop generating
        close = getattr(chunks, 'close', None)
        if close:
            close()
