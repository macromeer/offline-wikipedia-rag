#!/usr/bin/env python3
"""
Full Offline Wikipedia RAG using Kiwix Server
Automatically starts Kiwix server if not running
"""

import ollama
import requests
from bs4 import BeautifulSoup
import argparse
from typing import List, Dict, Optional
from urllib.parse import quote
import re
import time
import subprocess
import os
import sys
import signal
import atexit
from pathlib import Path


# Global variable to track Kiwix process started by this script
_kiwix_process = None


# Shared language filters for query understanding/keyword extraction
QUESTION_STOPWORDS = {
    'what', 'when', 'where', 'who', 'whom', 'whose', 'why', 'which', 'how',
    'is', 'are', 'was', 'were', 'am', 'been', 'being',
    'does', 'do', 'did', 'done', 'doing',
    'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'its', 'it', 'has', 'have', 'had', 'having',
    'this', 'that', 'these', 'those',
    'me', 'you', 'tell', 'explain', 'describe', 'define',
    'cause', 'causes', 'caused',
    'become', 'became', 'get', 'got', 'make', 'made', 'take', 'took'
}

QUESTION_SKIP_WORDS = {
    # Topic-agnostic filler terms and vague qualifiers
    'them', 'this', 'that', 'these', 'those', 'some', 'many', 'much', 'more', 'most',
    'people', 'person', 'persons', 'anyone', 'anybody', 'everyone', 'everybody',
    'someone', 'somebody', 'nobody', 'others', 'other', 'another',
    'thing', 'things', 'something', 'anything', 'everything', 'nothing', 'stuff',
    'good', 'bad', 'best', 'worst', 'better', 'great', 'awful', 'awesome', 'terrible', 'excellent', 'poor',
    'worth', 'value', 'quality', 'type', 'types', 'kind', 'kinds'
}

KEYWORD_BLACKLIST = QUESTION_STOPWORDS.union(QUESTION_SKIP_WORDS)

def _normalize_for_match(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(tokens)


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


def _parse_param_billions(text: str) -> Optional[float]:
    """Parse a parameter count like '26b', '1.5b', '751.63M', '8x7b' into billions"""
    match = re.search(r'(?:(\d+)x)?(\d+(?:\.\d+)?)([bm])(?![a-z])', text.lower())
    if not match:
        return None
    experts, value, unit = match.groups()
    billions = float(value) * (int(experts) if experts else 1)
    return billions / 1000 if unit == 'm' else billions


def _is_excluded_model(name: str) -> bool:
    lowered = name.lower()
    if any(s in lowered for s in EXCLUDED_MODEL_SUBSTRINGS):
        return True
    components = re.split(r'[-_:./]', lowered)
    return any(t in components for t in EXCLUDED_MODEL_TOKENS)


def _canonical_model_name(name: str) -> str:
    """'llama3.1' and 'llama3.1:latest' name the same model"""
    return name if ':' in name else f"{name}:latest"


def _num_ctx_for(prompt: str, num_predict: int) -> int:
    """Smallest power-of-two context (>= MIN_NUM_CTX) that fits prompt + output"""
    needed = len(prompt) // CHARS_PER_TOKEN + num_predict + 512
    num_ctx = MIN_NUM_CTX
    while num_ctx < needed:
        num_ctx *= 2
    return num_ctx

def _find_kiwix_binary():
    """Find kiwix-serve binary in common locations"""
    locations = [
        Path.home() / ".local/bin/kiwix-serve",
        Path("/usr/local/bin/kiwix-serve"),
        Path("/usr/bin/kiwix-serve"),
    ]
    
    for location in locations:
        if location.exists():
            return str(location)
    
    # Try system PATH
    try:
        result = subprocess.run(["which", "kiwix-serve"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return None


ZIM_SEARCH_PATHS = [
    Path.home() / "wikipedia-offline",
    Path.home() / "Downloads",
    Path("/data/wikipedia"),
    Path("/var/lib/kiwix"),
]


def _is_complete_zim(path: Path) -> bool:
    """
    Check the ZIM header: magic number, and file size equal to the MD5
    checksum position + 16. Rejects partial downloads without reading the file.
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(80)
        if len(header) < 80 or header[:4] != b'ZIM\x04':
            return False
        checksum_pos = int.from_bytes(header[72:80], 'little')
        return path.stat().st_size == checksum_pos + 16
    except OSError:
        return False


def _zim_sort_key(path: Path):
    """Newest dump first: date suffix in the name (e.g. _2026-06), then mtime"""
    match = re.search(r'_(\d{4}-\d{2})\.zim$', path.name)
    return (match.group(1) if match else '', path.stat().st_mtime)


def _find_zim_files(search_paths: List[Path] = None) -> List[Path]:
    """Find complete ZIM files in common locations, newest first"""
    zim_files = []
    for path in search_paths or ZIM_SEARCH_PATHS:
        if not path.exists():
            continue
        for zim in path.glob("*.zim"):
            if _is_complete_zim(zim):
                zim_files.append(zim)
            else:
                print(f"⚠ Skipping incomplete ZIM (still downloading?): {zim}", file=sys.stderr)
    return sorted(zim_files, key=_zim_sort_key, reverse=True)


def resolve_zim_path(cli_value: str = None) -> Optional[Path]:
    """Pick the ZIM to use: --zim flag, then WIKI_ZIM env var, then auto-discovery"""
    explicit = cli_value or os.environ.get('WIKI_ZIM')
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ZIM file not found: {path}")
        if not _is_complete_zim(path):
            raise ValueError(f"Not a complete ZIM file (still downloading?): {path}")
        return path
    zim_files = _find_zim_files()
    return zim_files[0] if zim_files else None


def _is_kiwix_running(port=8080):
    """Check if Kiwix server is already running"""
    try:
        response = requests.get(f"http://localhost:{port}/", timeout=2)
        return response.status_code == 200
    except:
        return False


def _start_kiwix_server(port=8080, zim_path: Path = None):
    """Start Kiwix server serving zim_path if not already running"""
    global _kiwix_process
    
    # Check if already running
    if _is_kiwix_running(port):
        print(f"✓ Kiwix server already running at http://localhost:{port}")
        return True
    
    print("📚 Starting Kiwix server...")
    
    # Find binary
    kiwix_bin = _find_kiwix_binary()
    if not kiwix_bin:
        print("❌ kiwix-serve not found. Install with:")
        print("   wget https://download.kiwix.org/release/kiwix-tools/kiwix-tools_linux-x86_64.tar.gz")
        print("   tar xzf kiwix-tools_linux-x86_64.tar.gz")
        print("   mv kiwix-tools_*/kiwix-serve ~/.local/bin/")
        return False
    
    if zim_path is None:
        print("❌ No Wikipedia ZIM file found. Download with:")
        print("   scripts/setup_full_offline_wikipedia.sh")
        print("   or pass --zim PATH / set WIKI_ZIM")
        return False

    # Start server
    try:
        print(f"   ZIM: {zim_path}")
        cmd = [kiwix_bin, "--port", str(port), "--address", "127.0.0.1", str(zim_path)]
        
        _kiwix_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Wait for server to be ready
        for i in range(10):
            time.sleep(0.5)
            if _is_kiwix_running(port):
                print(f"✓ Kiwix server started at http://localhost:{port}")
                return True
        
        print("⚠ Kiwix server started but not responding")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start Kiwix server: {e}")
        return False


def _cleanup_kiwix():
    """Stop Kiwix server if we started it"""
    global _kiwix_process
    if _kiwix_process:
        try:
            os.killpg(os.getpgid(_kiwix_process.pid), signal.SIGTERM)
            _kiwix_process.wait(timeout=5)
            print("\n✓ Stopped Kiwix server")
        except:
            pass


# Register cleanup on exit
atexit.register(_cleanup_kiwix)


class KiwixWikipediaRAG:
    """RAG system using local Kiwix Wikipedia server with two-stage AI pipeline"""
    
    def __init__(self, model_name: str = None, selection_model: str = None, kiwix_url: str = "http://localhost:8080", auto_start: bool = True, zim_path: Path = None):
        """
        Initialize the Kiwix RAG system with specialized models

        Args:
            model_name: Summarization model name (auto-detects if None)
            selection_model: Article selection model name (auto-detects if None)
            kiwix_url: URL of the Kiwix server
            auto_start: Automatically start Kiwix server if not running
            zim_path: ZIM file to serve when auto-starting (see resolve_zim_path)
        """
        self.kiwix_url = kiwix_url.rstrip('/')
        self.zim_path = zim_path

        # Test Kiwix connection or auto-start
        try:
            response = requests.get(f"{self.kiwix_url}/", timeout=5)
            response.raise_for_status()
            print(f"✓ Connected to Kiwix server at {self.kiwix_url}")
        except Exception as e:
            if auto_start:
                port = int(kiwix_url.split(":")[-1]) if ":" in kiwix_url else 8080
                if not _start_kiwix_server(port, zim_path):
                    raise Exception(f"Could not connect to or start Kiwix server at {self.kiwix_url}")
            else:
                raise Exception(f"Could not connect to Kiwix server at {self.kiwix_url}: {e}")

        # Base URL for direct article lookups, e.g. http://localhost:8080/content/wikipedia_en_all_nopic_2026-06
        self.content_base = self._discover_content_base()
        if self.content_base:
            print(f"✓ Wikipedia book: {self.content_base}")

        # Detect available models
        available_models = self._get_available_models()
        
        # Configure selection model (Stage 1: Classification)
        # Best: Qwen2.5-32B (superior classification), Mistral-Small, Hermes-3-8B
        self.selection_model = self._detect_selection_model(selection_model, available_models)
        
        # Configure summarization model (Stage 2: Synthesis)
        # Best: Llama-3.1-70B (world knowledge + coherent generation), Gemma-2-27B
        self.model_name = self._detect_summarization_model(model_name, available_models)
        
        print(f"✓ Selection model: {self.selection_model}")
        print(f"✓ Summarization model: {self.model_name}")
    
    def _discover_content_base(self) -> Optional[str]:
        """
        Find the book URL prefix from the server's OPDS catalog, preferring the
        book that matches self.zim_path. Falls back to the ZIM file stem.
        """
        stem = self.zim_path.stem if self.zim_path else None
        try:
            response = requests.get(f"{self.kiwix_url}/catalog/v2/entries", params={'count': 100}, timeout=5)
            response.raise_for_status()
            hrefs = re.findall(r'<link[^>]*type="text/html"[^>]*href="([^"]*/content/[^"]+)"', response.text)
        except Exception:
            hrefs = []
        if hrefs:
            matching = [h for h in hrefs if stem and h.rstrip('/').endswith(f"/{stem}")]
            wikipedia = [h for h in hrefs if '/wikipedia_' in h]
            href = (matching or wikipedia or hrefs)[0].rstrip('/')
            return href if href.startswith('http') else f"{self.kiwix_url}{href}"
        if stem:
            return f"{self.kiwix_url}/content/{stem}"
        return None

    def _article_url(self, title: str) -> Optional[str]:
        """Direct URL for an article title (titles are case-sensitive, spaces become underscores)"""
        if not getattr(self, 'content_base', None):
            return None
        return f"{self.content_base}/{quote(title.replace(' ', '_'))}"

    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models (also records parameter sizes for tags like ':latest')"""
        self._model_param_sizes = {}
        try:
            response = ollama.list()
            if hasattr(response, 'models'):
                for m in response.models:
                    size = _parse_param_billions(getattr(getattr(m, 'details', None), 'parameter_size', None) or '')
                    if size is not None:
                        self._model_param_sizes[m.model] = size
                return [m.model for m in response.models]
            elif isinstance(response, dict) and 'models' in response:
                return [m['name'] if isinstance(m, dict) else m.model for m in response['models']]
            return []
        except Exception as e:
            print(f"⚠ Could not list models: {e}")
            return []

    def _model_param_billions(self, name: str) -> Optional[float]:
        """Parameter count in billions, from the tag (e.g. ':26b') or from `ollama list` details"""
        tag = name.split(':', 1)[1] if ':' in name else ''
        size = _parse_param_billions(tag)
        if size is None:
            size = getattr(self, '_model_param_sizes', {}).get(name)
        return size

    def _is_eligible_model(self, name: str) -> bool:
        """Reject coder/embedding/reranker/reasoning models and models below MIN_MODEL_PARAMS_B"""
        if _is_excluded_model(name):
            return False
        size = self._model_param_billions(name)
        return size is None or size >= MIN_MODEL_PARAMS_B

    def _pick_model(self, preferences: List[str], available: List[str], role: str) -> str:
        """
        For each preference in order: exact tag match, else an eligible model of the
        same family with another tag (e.g. 'qwen3.6' -> 'qwen3.6:35b'). Then any
        eligible model, then any non-excluded model.
        """
        canonical = {_canonical_model_name(a): a for a in available}
        for pref in preferences:
            exact = canonical.get(_canonical_model_name(pref))
            if exact:
                return exact
            family = pref.split(':')[0]
            for avail in available:
                if avail.split(':')[0] == family and self._is_eligible_model(avail):
                    return avail

        for avail in available:
            if self._is_eligible_model(avail):
                print(f"⚠ Using fallback {role} model: {avail}")
                return avail
        for avail in available:
            if not _is_excluded_model(avail):
                print(f"⚠ Using fallback {role} model: {avail} (below {MIN_MODEL_PARAMS_B:g}B, expect weak results)")
                return avail

        raise Exception(f"No suitable Ollama model for {role}. Pull one, e.g.: ollama pull qwen3:8b")

    def _detect_selection_model(self, preferred: str, available: List[str]) -> str:
        """
        Detect best model for article selection (classification task).
        Reasoning, coder and sub-4B models are never auto-selected.
        """
        if preferred:
            return preferred
        return self._pick_model(SELECTION_MODEL_PREFERENCES, available, 'selection')

    def _detect_summarization_model(self, preferred: str, available: List[str]) -> str:
        """Detect best model for summarization (synthesis task)"""
        if preferred:
            return preferred
        return self._pick_model(SUMMARIZATION_MODEL_PREFERENCES, available, 'summarization')
    
    def extract_search_terms(self, question: str) -> List[str]:
        """
        Extract Wikipedia article title candidates following Wikipedia naming conventions
        
        Wikipedia titles follow specific conventions that Kiwix can leverage:
        - Sentence case (first word capitalized, rest lowercase unless proper nouns)
        - Singular form preferred ("Cat" not "Cats")
        - Common names over official ("Bill Clinton" not "William Jefferson Clinton")
        - No leading articles ("French Revolution" not "The French Revolution")
        
        Kiwix search is case-insensitive and prefix-based for title matching.
        
        Args:
            question: User's natural language question
            
        Returns:
            List of 3-5 Wikipedia article title candidates
        """
        q_lower = question.lower()
        terms = []
        
        # Strategy 0: Extract quoted terms (e.g., "The Expanse")
        quoted_terms = re.findall(r'["\']([^"\'\']+)["\']', question)
        for term in quoted_terms:
            if term.strip() and len(term.strip()) > 2:
                terms.append(term.strip())
        
        # Remove question words and common stopwords
        stopwords = QUESTION_STOPWORDS
        
        # Extract proper nouns (capitalized words in original question)
        words_original = question.replace('?', '').replace(',', '').replace('.', '').split()
        proper_nouns = []
        i = 0
        while i < len(words_original):
            word = words_original[i]
            # Check if capitalized and not a stopword
            if word and word[0].isupper() and word.lower() not in stopwords:
                # Check for multi-word proper nouns (consecutive capitalized words)
                phrase = [word]
                j = i + 1
                while j < len(words_original) and words_original[j] and words_original[j][0].isupper():
                    phrase.append(words_original[j])
                    j += 1
                proper_nouns.append(' '.join(phrase))
                i = j
            else:
                i += 1
        
        # Extract content words (lowercase, filtered)
        words_lower = q_lower.replace('?', '').replace(',', '').replace('.', '').split()
        content_words = [w.strip('?.,!:;\'"') for w in words_lower 
                        if w.strip('?.,!:;\'"') not in stopwords and len(w) > 3]
        
        # Strategy 1: Use proper nouns as-is (e.g., "Donald Trump")
        for noun in proper_nouns[:3]:
            if noun not in terms:
                terms.append(noun)
        
        # Strategy 2: Use important single content words (capitalized for Wikipedia)
        # Skip very short words and common pronouns
        skip_words = QUESTION_SKIP_WORDS
        for word in content_words[:5]:  # Look at more words
            if word not in skip_words and len(word) >= 4:  # At least 4 chars to catch words like "mars", "love", etc.
                title = word.capitalize()
                if title not in terms and title not in proper_nouns:
                    terms.append(title)
        
        # Strategy 3: Try consecutive word pairs from content words
        for i in range(min(2, len(content_words) - 1)):
            if content_words[i] in skip_words or content_words[i+1] in skip_words:
                continue
            phrase = f"{content_words[i].capitalize()} {content_words[i+1]}"
            if phrase not in terms:
                terms.append(phrase)
        
        return terms[:5] if terms else [question]

    def extract_primary_keywords(self, question: str) -> List[str]:
        """Derive primary topical keywords (lowercase) from the question text"""
        normalized_tokens = re.findall(r"[a-z0-9']+", question.lower())
        base_tokens: List[str] = []
        for token in normalized_tokens:
            if len(token) < 3:
                continue
            if token in KEYWORD_BLACKLIST:
                continue
            if token not in base_tokens:
                base_tokens.append(token)
        try:
            search_terms = self.extract_search_terms(question)
        except Exception:
            search_terms = []
        for term in search_terms:
            for token in re.split(r"[\s\-_/()]+", term.lower()):
                token = token.strip()
                if len(token) < 3 or token in KEYWORD_BLACKLIST:
                    continue
                if token not in base_tokens:
                    base_tokens.append(token)
        keywords: List[str] = []
        def _add_keyword(value: str):
            value = value.strip()
            if value and value not in keywords:
                keywords.append(value)
        for token in base_tokens:
            _add_keyword(token)
        for i in range(len(base_tokens) - 1):
            first_token = base_tokens[i]
            second_token = base_tokens[i + 1]
            if first_token in KEYWORD_BLACKLIST or second_token in KEYWORD_BLACKLIST:
                continue
            pair = f"{first_token} {second_token}"
            _add_keyword(pair)
        if not keywords:
            fallback = [w for w in normalized_tokens if len(w) >= 4]
            if fallback:
                _add_keyword(fallback[0])
        return keywords[:6]

    def extract_focus_phrases(self, question: str) -> List[str]:
        """Return multi-word phrases that should be treated as primary topics"""
        phrases: List[str] = []

        def _add_phrase(raw_value: str):
            raw_value = raw_value.strip()
            normalized = _normalize_for_match(raw_value)
            if not raw_value or not normalized:
                return
            if len(normalized.split()) < 2:
                return
            if raw_value not in phrases:
                phrases.append(raw_value)

        # Strategy 1: quoted spans
        for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', question):
            candidate = match[0] or match[1]
            _add_phrase(candidate)

        # Strategy 2: proper nouns / extracted search terms with spaces
        try:
            search_terms = self.extract_search_terms(question)
        except Exception:
            search_terms = []
        for term in search_terms:
            if ' ' in term and term not in phrases:
                _add_phrase(term)

        return phrases[:4]

    def _title_matches_keywords(self, title: str, keywords: List[str]) -> bool:
        """Check if title contains enough keyword overlap"""
        if not keywords:
            return True
        normalized_title = _normalize_for_match(title)
        matches = 0
        for keyword in keywords:
            normalized_keyword = _normalize_for_match(keyword)
            if not normalized_keyword:
                continue
            if normalized_keyword in normalized_title:
                matches += 1
        if matches == 0:
            return False
        if len(keywords) >= 3:
            return matches >= 2
        return matches >= 1

    def _title_matches_focus_phrase(self, title: str, phrases: List[str]) -> bool:
        if not phrases:
            return False
        title_tokens = _normalize_for_match(title).split()
        if not title_tokens:
            return False
        for phrase in phrases:
            phrase_tokens = _normalize_for_match(phrase).split()
            if not phrase_tokens:
                continue
            pos = 0
            matched_all = True
            for token in phrase_tokens:
                while pos < len(title_tokens) and title_tokens[pos] != token:
                    pos += 1
                if pos == len(title_tokens):
                    matched_all = False
                    break
                pos += 1
            if matched_all:
                return True
        return False
    
    def search_kiwix(self, query: str, max_results: int = 25, primary_keywords: List[str] = None, focus_phrases: List[str] = None) -> List[Dict]:
        """
        Search local Wikipedia via Kiwix using Wikipedia title conventions
        
        Retrieves more results and uses multiple search strategies to find main articles
        (e.g., "Earthquake" article comes after "List of earthquakes..." in alphabetical order)
        
        Args:
            query: Search query (user's question)
            max_results: Maximum number of results to retrieve per search term
            primary_keywords: Optional keywords to prioritize when ordering results
            focus_phrases: Multi-word phrases that should be prioritized
            
        Returns:
            List of search results with titles and URLs
        """
        try:
            # Extract Wikipedia-style article titles
            search_terms = self.extract_search_terms(query)
            
            all_results = []
            seen_titles = set()
            
            # Strategy 1: Search each extracted term
            for term in search_terms:
                if len(all_results) >= 100:  # Increased cap for better selection
                    break
                results = self._do_search(term, max_results)
                for r in results:
                    # Case-insensitive duplicate detection
                    title_lower = r['title'].lower()
                    if title_lower not in seen_titles:
                        all_results.append(r)
                        seen_titles.add(title_lower)
            
            # Strategy 2: Try TV show/movie/media format (common Wikipedia pattern)
            # e.g., "The Expanse" -> "The Expanse (TV series)"
            for term in search_terms[:3]:
                if len(all_results) >= 100:
                    break
                for suffix in [" (TV series)", " (film)", " (TV show)", " (television)"]:
                    media_title = f"{term}{suffix}"
                    media_url = self._article_url(media_title)
                    if not media_url:
                        break
                    try:
                        response = requests.head(media_url, timeout=2, allow_redirects=True)
                        if response.status_code == 200:
                            title_lower = media_title.lower()
                            if title_lower not in seen_titles:
                                all_results.insert(0, {'title': media_title, 'url': media_url})
                                seen_titles.add(title_lower)
                                break  # Found it, move to next term
                    except:
                        pass
            
            # Strategy 3: Direct lookup for main article (singular form)  
            # This helps find "Earthquake" even when lists come first alphabetically
            for term in search_terms[:3]:  # Try first 3 terms as direct lookups
                if len(all_results) >= 100:
                    break
                # Try exact match by requesting the article directly  
                direct_url = self._article_url(term)
                if not direct_url:
                    break
                try:
                    response = requests.head(direct_url, timeout=2, allow_redirects=True)
                    if response.status_code == 200:
                        title = term
                        title_lower = title.lower()
                        if title_lower not in seen_titles:
                            all_results.insert(0, {'title': title, 'url': direct_url})  # Insert at beginning
                            seen_titles.add(title_lower)
                except:
                    pass  # Article doesn't exist or failed to load
            
            if primary_keywords:
                prioritized, others = [], []
                for result in all_results:
                    if self._title_matches_keywords(result['title'], primary_keywords):
                        prioritized.append(result)
                    else:
                        others.append(result)
                if prioritized:
                    all_results = prioritized + others
            if focus_phrases:
                phrase_hits, remainder = [], []
                for result in all_results:
                    if self._title_matches_focus_phrase(result['title'], focus_phrases):
                        phrase_hits.append(result)
                    else:
                        remainder.append(result)
                if phrase_hits:
                    all_results = phrase_hits + remainder
            print(f"  ✓ Retrieved {len(all_results)} unique candidates")
            return all_results
            
        except Exception as e:
            print(f"⚠ Search error: {e}")
            return []
    
    def _do_search(self, pattern: str, limit: int = 15) -> List[Dict]:
        """Helper to perform a single Kiwix search"""
        try:
            search_url = f"{self.kiwix_url}/search"
            params = {'pattern': pattern, 'pageSize': limit}
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            results_div = soup.find('div', class_='results')
            
            if results_div:
                for li in results_div.find_all('li')[:limit]:
                    link = li.find('a')
                    if link and link.get('href'):
                        title = link.get_text(strip=True)
                        url = link['href']
                        if not url.startswith('http'):
                            url = f"{self.kiwix_url}{url}"
                        results.append({'title': title, 'url': url})
            
            return results
        except:
            return []
    
    def fetch_article_abstract(self, url: str) -> str:
        """Fetch just the first paragraph (abstract) of an article"""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            content = soup.find('div', {'id': 'mw-content-text'})
            if not content:
                content = soup.find('div', {'class': 'mw-parser-output'})
            
            if content:
                # Get first meaningful paragraph
                for p in content.find_all('p'):
                    text = p.get_text(strip=True)
                    if len(text) > 100:  # Skip short paragraphs
                        return text[:500]  # First 500 chars
            return ""
        except:
            return ""
    
    def select_relevant_articles(self, question: str, search_results: List[Dict], target_count: int, primary_keywords: List[str] = None, focus_phrases: List[str] = None) -> List[Dict]:
        """
        Stage 1: Use specialized classification model for article selection
        
        Args:
            question: User's question
            search_results: List of article titles, URLs, and abstracts from search
            target_count: Number of articles to select
            primary_keywords: Keyword hints extracted from the user question
            focus_phrases: Multi-word phrases extracted from the user question
        """
        if len(search_results) <= target_count:
            return search_results
        keywords = primary_keywords or []
        phrases = focus_phrases or []
        has_phrase_candidate = bool(phrases and any(self._title_matches_focus_phrase(r['title'], phrases) for r in search_results))

        def relevance_score(result):
            title = result['title'].lower()
            score = 0
            if any(suffix in title for suffix in [' (tv series)', ' (film)', ' (tv show)', ' (television)']):
                score += 100
            if title.startswith('list of') or title.startswith('lists of'):
                score -= 50
            if 'disambiguation' in title or 'index of' in title:
                score -= 40
            phrase_match = self._title_matches_focus_phrase(result['title'], phrases)
            if has_phrase_candidate:
                if phrase_match:
                    score += 200
                else:
                    score -= 90
            abstract = result.get('abstract', '')
            if len(abstract) > 200:
                score += 20
            elif len(abstract) > 100:
                score += 10
            if keywords and abstract:
                abstract_lower = abstract.lower()
                if any(keyword in abstract_lower for keyword in keywords):
                    score += 25
            if len(title) < 30:
                score += 5
            if keywords:
                if self._title_matches_keywords(title, keywords):
                    score += 80
                else:
                    score -= 60
            if phrases and not has_phrase_candidate and phrase_match:
                score += 60
            return score

        search_results = sorted(search_results, key=relevance_score, reverse=True)

        articles_text = ""
        article_index_map: Dict[int, int] = {}
        display_num = 1
        for i, result in enumerate(search_results[:30]):
            title = result['title']
            abstract = result.get('abstract', '')
            if abstract and len(abstract) > 30:
                article_index_map[display_num] = i
                abstract_preview = abstract[:200] + "..." if len(abstract) > 200 else abstract
                articles_text += f"{display_num}. **{title}**\n   {abstract_preview}\n\n"
                display_num += 1
            elif not title.lower().startswith('list of') and not title.lower().startswith('lists of'):
                article_index_map[display_num] = i
                articles_text += f"{display_num}. **{title}**\n   (Main article)\n\n"
                display_num += 1

        if not articles_text:
            for i, result in enumerate(search_results[:15]):
                article_index_map[i + 1] = i
                title = result['title']
                articles_text += f"{i+1}. **{title}**\n\n"

        print(f"  🤖 Selecting with {self.selection_model} (using article abstracts)...")
        keyword_note = ""
        if keywords:
            keyword_note = "Primary topic keywords: " + ', '.join(f'"{kw}"' for kw in keywords[:4]) + "\n"
        phrase_note = ""
        if phrases:
            phrase_note = "Focus phrases: " + ', '.join(f'"{ph}"' for ph in phrases[:2]) + "\n"
        header_note = (keyword_note + phrase_note + "\n") if (keyword_note or phrase_note) else ""

        selection_prompt = f"""You are selecting Wikipedia articles to answer this question:

Question: "{question}"

{header_note}Available articles:
{keyword_note if keyword_note else ''}Available articles:
{articles_text}

Task: Select the {target_count} MOST RELEVANT articles.

RULES:
1. ALWAYS select the main article about the question's primary topic
   - "The Expanse TV show" → select "The Expanse (TV series)"
   - "Albert Einstein" → select "Albert Einstein" biography
   - "earthquakes" → select "Earthquake" main article

2. When keywords are provided above, every selected article MUST contain those keywords (or obvious singular/plural variants) in the title or abstract

3. When focus phrases are provided above, prioritize articles whose titles contain that exact phrase (punctuation differences are OK)

4. For TV shows, movies, books:
   - Select the main article about that specific work
   - REJECT: songs, unrelated topics with similar names
   - Example: "The Expanse" show ≠ "Expanse" (geography term)

5. Match the question's intent:
   - "Is X good?" → select main article about X
   - "Who is X?" → select biographical article
   - "What causes X?" → select article explaining X

6. REJECT:
   - Articles about different topics that share a word
   - Lists, episodes, songs, year pages
   - Tangentially related topics

Examples:
Q: "Is The Expanse a good show?"
→ Select: "The Expanse (TV series)" NOT "Expanse" or "Good Mythical Morning"

Q: "Tell me about earthquakes"
→ Select: "Earthquake" main article NOT "List of earthquakes"

Output ONLY comma-separated numbers (example: 2,5,8):
"""

        try:
            response = self._chat(
                self.selection_model,
                selection_prompt,
                options={
                    'num_predict': 200,
                    'temperature': 0.2,
                    'top_p': 0.9,
                },
                stage='Selection',
            )
            answer = response['message']['content'].strip()
            if answer:
                numbers = re.findall(r'\d+', answer)
                seen_indices = set()
                indices = []
                for n in numbers:
                    num = int(n)
                    if num in article_index_map:
                        actual_idx = article_index_map[num]
                        if actual_idx not in seen_indices:
                            indices.append(actual_idx)
                            seen_indices.add(actual_idx)
                indices = indices[:target_count]
                if indices:
                    selected = [search_results[i] for i in indices]
                    if selected:
                        return selected
            print(f"  ⚠ Selection returned no valid results, using fallback")
        except Exception as e:
            print(f"  ⚠ Selection error: {e}")

        filtered = [r for r in search_results if not (
            r['title'].lower().startswith('list of') or
            r['title'].lower().startswith('lists of') or
            re.search(r'\b(19|20)\d{2}\b', r['title']) or
            'disambiguation' in r['title'].lower()
        )]
        if keywords:
            keyword_filtered = [r for r in filtered if self._title_matches_keywords(r['title'], keywords)]
            if keyword_filtered:
                filtered = keyword_filtered
            else:
                keyword_filtered = [r for r in search_results if self._title_matches_keywords(r['title'], keywords)]
                if keyword_filtered:
                    filtered = keyword_filtered
        if phrases:
            phrase_filtered = [r for r in filtered if self._title_matches_focus_phrase(r['title'], phrases)]
            if phrase_filtered:
                filtered = phrase_filtered
        return (filtered or search_results)[:target_count]

    def fetch_article(self, url: str, max_paragraphs: int = None) -> str:
        """
        Fetch article content from Kiwix
        
        Args:
            url: Article URL
            max_paragraphs: Maximum paragraphs to read (None = all)
            
        Returns:
            Article text content
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find main content (Wikipedia structure)
            content = soup.find('div', {'id': 'mw-content-text'})
            if not content:
                content = soup.find('div', {'class': 'mw-parser-output'})
            if not content:
                content = soup.find('body')
            
            if content:
                # Extract paragraphs (balanced by article count)
                paragraphs = content.find_all('p')
                
                # Filter out empty paragraphs and get text
                texts = []
                para_limit = max_paragraphs if max_paragraphs else len(paragraphs)
                total_chars = 0
                max_chars_per_article = 8000  # Hard limit per article
                
                for p in paragraphs[:para_limit]:
                    text = p.get_text(strip=True)
                    if len(text) > 50:  # Only meaningful paragraphs
                        # Stop if we've gathered enough content
                        if total_chars + len(text) > max_chars_per_article:
                            break
                        texts.append(text)
                        total_chars += len(text)
                
                combined = '\n\n'.join(texts)
                
                # Clean up text
                combined = re.sub(r'\[\d+\]', '', combined)  # Remove citation numbers
                combined = re.sub(r'\s+', ' ', combined)  # Normalize whitespace
                
                return combined
            
            return ""
            
        except Exception as e:
            print(f"⚠ Fetch error for {url}: {e}")
            return ""
    
    def _chat(self, model: str, prompt: str, options: Dict, stage: str):
        """
        Single-prompt ollama.chat with num_ctx sized to the prompt and thinking off
        (thinking models otherwise spend num_predict on hidden reasoning and return
        empty content). Logs prompt_eval_count so context truncation is visible.
        """
        num_ctx = _num_ctx_for(prompt, options.get('num_predict', 0))
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={**options, 'num_ctx': num_ctx},
            think=False,
        )
        prompt_tokens = response.get('prompt_eval_count') or 0
        print(f"  📏 {stage}: {prompt_tokens} prompt tokens (num_ctx {num_ctx})")
        if prompt_tokens >= num_ctx - options.get('num_predict', 0):
            print(f"  ⚠ {stage} prompt may have been truncated to fit num_ctx {num_ctx}")
        return response

    def estimate_question_complexity(self, question: str) -> int:
        """
        Estimate question complexity to determine how many articles to retrieve
        
        Args:
            question: User's question
            
        Returns:
            Number of articles to retrieve (3-7)
        """
        question_lower = question.lower()
        
        # Complex question indicators
        complexity_score = 0
        
        # Multi-part questions (need multiple perspectives)
        if ' and ' in question_lower:
            complexity_score += 2
        if ' vs ' in question_lower or ' versus ' in question_lower:
            complexity_score += 3  # Comparisons need both sides
        
        # Comparison/relationship questions (need context from multiple articles)
        if any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs']):
            complexity_score += 3
        if any(word in question_lower for word in ['relationship', 'connect', 'relate', 'impact', 'affect', 'influence', 'cause']):
            complexity_score += 2
        
        # Deep/analytical questions (need comprehensive context)
        if any(word in question_lower for word in ['how does', 'how do', 'why', 'explain']):
            complexity_score += 2
        if any(word in question_lower for word in ['history', 'evolution', 'development', 'origin']):
            complexity_score += 2
        
        # Broad conceptual questions
        if any(word in question_lower for word in ['overview', 'summary', 'introduction', 'basics']):
            complexity_score += 1
        
        # Future/prediction questions (need current state + theories)
        if any(word in question_lower for word in ['future', 'prediction', 'will', 'going to']):
            complexity_score += 2
        
        # Long questions often need more context
        if len(question.split()) > 12:
            complexity_score += 1
        
        # Map complexity to number of articles
        # With better selection AI, we can retrieve more targeted articles
        if complexity_score >= 6:
            return 6  # Very complex - retrieve 6 articles
        elif complexity_score >= 4:
            return 5  # Complex - retrieve 5 articles
        elif complexity_score >= 3:
            return 4  # Moderate-complex - retrieve 4 articles
        elif complexity_score >= 2:
            return 4  # Moderate - retrieve 4 articles
        else:
            return 3  # Simple - retrieve 3 articles (minimum)
    
    def query_with_rag(self, question: str, max_results: int = None) -> Dict:
        """
        Answer question using RAG with local Wikipedia
        
        Args:
            question: User's question
            max_results: Number of articles to retrieve (auto-detected if None)
            
        Returns:
            Dictionary with answer and sources
        """
        start_time = time.time()
        
        # Auto-detect complexity if not specified
        if max_results is None:
            max_results = self.estimate_question_complexity(question)
        
        print(f"\n🔍 Searching local Wikipedia for: {question}")
        primary_keywords = self.extract_primary_keywords(question)
        focus_phrases = self.extract_focus_phrases(question)
        if primary_keywords:
            print(f"  🔑 Focus keywords: {', '.join(primary_keywords[:4])}")
        if focus_phrases:
            print(f"  🧭 Focus phrases: {', '.join(focus_phrases[:2])}")
        
        # Step 1: Search Kiwix (retrieves 3x more results)
        search_results = self.search_kiwix(question, max_results=max_results, primary_keywords=primary_keywords, focus_phrases=focus_phrases)
        
        if not search_results:
            return {
                'question': question,
                'answer': "No relevant Wikipedia articles found in local database.",
                'sources': [],
                'model': self.model_name
            }
        
        print(f"✓ Found {len(search_results)} candidate article(s)")
        
        # Step 1.5: Fetch abstracts for better selection (first paragraph only)
        print(f"  📄 Fetching article abstracts for AI selection...")
        for i, result in enumerate(search_results):
            if i >= 30:  # Limit abstract fetching to first 30 for speed
                break
            abstract = self.fetch_article_abstract(result['url'])
            result['abstract'] = abstract
        
        # Step 2: Use AI to select most relevant articles with context
        selected_results = self.select_relevant_articles(question, search_results, max_results, primary_keywords=primary_keywords, focus_phrases=focus_phrases)
        
        selected_titles = [r['title'] for r in selected_results]
        print(f"✓ AI selected {len(selected_results)} article(s): {', '.join(selected_titles)}")
        
        # Balance content depth with article count for consistent speed
        # Target: Keep total context under 40-50k chars for <15s response time
        paragraphs_per_article = {
            3: 20,   # 3 articles: ~20 paragraphs each (~24k chars total)
            4: 15,   # 4 articles: ~15 paragraphs each (~24k chars total)
            5: 12,   # 5 articles: ~12 paragraphs each (~24k chars total)
            6: 10,   # 6 articles: ~10 paragraphs each (~24k chars total)
            7: 8,    # 7 articles: ~8 paragraphs each (~22k chars total)
        }
        max_paragraphs = paragraphs_per_article.get(len(selected_results), 15)
        print(f"  📊 Reading ~{max_paragraphs} paragraphs per article (max 8k chars each)")
        
        # Fetch article contents
        contents = []
        for result in selected_results:
            print(f"  📄 Fetching: {result['title']}")
            content = self.fetch_article(result['url'], max_paragraphs=max_paragraphs)
            if content:
                contents.append({
                    'title': result['title'],
                    'content': content,
                    'url': result['url']
                })
        
        if not contents:
            # Check if question contains abbreviations/acronyms
            words = question.replace('?', '').replace('.', '').replace(',', '').split()
            abbreviations = [w.strip() for w in words if w.strip().isupper() and len(w.strip()) >= 2 and len(w.strip()) <= 5]
            
            if abbreviations:
                abbrev_list = ', '.join(f"'{a}'" for a in abbreviations[:3])  # Show max 3
                suggestion = f"Could not find article content. Your question contains abbreviation(s): {abbrev_list}.\n\nTip: Try spelling out the full term (e.g., 'What is an exchange-traded fund?' instead of 'What is an ETF?')"
            else:
                suggestion = "Could not retrieve article content. Try rephrasing your question or using different search terms."
            
            return {
                'question': question,
                'answer': suggestion,
                'sources': [],
                'model': self.model_name
            }
        
        # Build context with article numbers for citation
        context_parts = []
        for idx, item in enumerate(contents, 1):
            context_parts.append(f"[Article {idx}] **{item['title']}**:\n{item['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Build source list for reference
        source_list = "\n".join([f"[{idx}] {item['title']}" for idx, item in enumerate(contents, 1)])
        
        # Create synthesis-optimized prompt for Stage 2
        # Llama-3.1-70B excels at world knowledge + coherent long-form generation
        prompt = f"""You are an expert research analyst synthesizing information from multiple Wikipedia articles.

TASK: Answer the question by synthesizing information from ALL provided articles.

Question: "{question}"

Available Articles:
{source_list}

Article Contents:
{context}

SYNTHESIS INSTRUCTIONS:
1. **Direct Verdict**: The first sentence must explicitly answer the question (e.g., "Yes, the film earned overwhelmingly positive reviews for... [1]"). Make the stance clear (yes/no/mixed) before adding context.
2. **Stay On-Task**: Only include details that help judge quality/relevance of the topic. Omit long cast lists or plot summaries unless they support the verdict.
3. **Comprehensiveness**: Integrate information from ALL articles to support the verdict.
4. **Coherence**: Create a logical narrative that links supporting evidence.
5. **Evidence**: Use concrete facts (awards, box office, critical reception) with citations.
6. **Perspectives**: Note differing viewpoints if present, and explain them.
7. **Structure**: Write in clear paragraphs; use lists only when essential.
8. **Accuracy**: Stay within the provided articles; do not invent data.
9. **Citations**: Add inline citations [1], [2], [3] after EVERY fact drawn from the articles.

CRITICAL - INLINE CITATIONS:
- Add [1], [2], or [3] immediately after each fact, quote, or claim from that article
- Multiple sources: use [1][2] or [1,2] if information appears in multiple articles
- Example: "Bill Murray was born in 1950 [1] and starred in Ghostbusters [1][3]."
- Every paragraph should have multiple citations showing source of information

FORMAT:
- Write natural paragraphs with inline citations only.
- Do NOT repeat the question.
- Do NOT add headings such as "References", "Sources", or "Bibliography"—inline citations are sufficient.
- End the answer immediately after the final paragraph (no trailing lists or sections).

Your synthesized answer with inline citations (stop after final paragraph):"""
        
        print(f"🤖 Generating synthesis with {self.model_name}...")
        
        # Query summarization model with optimized settings
        # Llama-3.1-70B: 3x faster inference, excellent coherent generation
        try:
            response = self._chat(
                self.model_name,
                prompt,
                options={
                    'num_predict': 1500,   # Allow comprehensive answers
                    'temperature': 0.7,    # Balance factual accuracy with coherence
                    'top_p': 0.9,
                    'repeat_penalty': 1.1, # Reduce repetition in synthesis
                },
                stage='Synthesis',
            )
            
            answer = response['message']['content']
            
            # Remove redundant references/sources section at the end
            # LLMs often add this despite instructions - we show sources separately
            import re
            # Match "References:", "Sources:", "Bibliography:" followed by citation list
            pattern = r'\n\s*\[?(References?|Sources?|Bibliography)\]?[:\-]?\s*(\n.*)?$'
            answer = re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE).rstrip()
            
        except Exception as e:
            print(f"  ⚠ Generation error: {e}")
            answer = "Error generating answer. Please try again."
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  Total time: {elapsed_time:.1f}s")
        
        return {
            'question': question,
            'answer': answer,
            'sources': contents,
            'model': self.model_name,
            'time': elapsed_time
        }
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*70)
        print(" 🌐 Offline Wikipedia AI Assistant")
        print("="*70)
        print(f" 🤖 Model: {self.model_name}")
        print(f" 📚 Wikipedia: Local ({self.kiwix_url})")
        print(f" 💡 Tip: Ask any question, type 'quit' to exit")
        print("="*70 + "\n")
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Get answer
                result = self.query_with_rag(question)
                
                print("\n" + "="*70)
                print("📖 Answer:\n")
                # Format answer with proper line wrapping
                answer_lines = result['answer'].split('\n')
                for line in answer_lines:
                    if line.strip():
                        # Skip old-style "Sources:" line if present
                        if not line.strip().lower().startswith('sources:'):
                            print(f"   {line}")
                    else:
                        print()
                
                print("\n" + "-"*70)
                print("📚 Source Articles (click to open):")
                for idx, s in enumerate(result['sources'], 1):
                    print(f"   [{idx}] {s['title']}")
                    print(f"       {s['url']}")
                print("="*70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def _check_ollama_running():
    """Check if Ollama is running"""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def _check_dependencies():
    """Check all required dependencies"""
    print("🔍 Checking dependencies...")
    
    # Check Ollama
    if not _check_ollama_running():
        print("❌ Ollama is not running")
        print("   Start it with: ollama serve")
        print("   Or install from: https://ollama.ai")
        return False
    print("✓ Ollama is running")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Offline Wikipedia RAG with Kiwix - Automatically handles setup',
        epilog='The script will automatically start Kiwix server if needed.'
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Summarization model (auto-detects: llama3.1:8b, gemma2:27b)')
    parser.add_argument('--selection-model', type=str, default=None,
                        help='Selection model (auto-detects: mistral:7b, qwen2.5)')
    parser.add_argument('--kiwix-url', type=str, default='http://localhost:8080',
                        help='Kiwix server URL')
    parser.add_argument('--question', type=str,
                        help='Single question (otherwise interactive mode)')
    parser.add_argument('--max-results', type=int, default=None,
                        help='Number of Wikipedia articles to retrieve (auto-detects by complexity)')
    parser.add_argument('--no-auto-start', action='store_true',
                        help='Do not automatically start Kiwix server')
    parser.add_argument('--zim', type=str, default=None,
                        help='Wikipedia ZIM file (default: $WIKI_ZIM, else newest complete *.zim in ~/wikipedia-offline)')

    args = parser.parse_args()

    try:
        # Check dependencies
        if not _check_dependencies():
            return 1

        print()

        zim_path = resolve_zim_path(args.zim)

        # Initialize RAG with two-stage pipeline (auto-starts Kiwix if needed)
        rag = KiwixWikipediaRAG(
            model_name=args.model,
            selection_model=args.selection_model,
            kiwix_url=args.kiwix_url,
            auto_start=not args.no_auto_start,
            zim_path=zim_path
        )
        
        if args.question:
            # Single question mode
            result = rag.query_with_rag(args.question, max_results=args.max_results)
            
            print("\n" + "="*70)
            print(f"❓ Question: {result['question']}\n")
            print("="*70)
            print("\n📖 Answer:\n")
            
            # Format answer with proper line wrapping
            answer_lines = result['answer'].split('\n')
            for line in answer_lines:
                if line.strip():
                    print(f"   {line}")
                else:
                    print()
            
            print("\n" + "-"*70)
            print("📚 Source Articles (click to open):")
            for idx, s in enumerate(result['sources'], 1):
                print(f"   [{idx}] {s['title']}")
                print(f"       {s['url']}")
            print("="*70 + "\n")
        else:
            # Interactive mode
            rag.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
