"""
One-shot pipelines: retrieve once, then one synthesis call.

- retrieval='zim': passages from retrieval.ZimStore (Phase 1); used by
  --no-tools and for models without tool calling.
- retrieval='kiwix': the v1 pipeline over kiwix-serve HTTP (search, LLM
  selection over abstracts, first paragraphs). Kept unchanged as the
  evaluation baseline.
"""

import re
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from retrieval import ZimStore

from . import llm
from .kiwix import KiwixServer
from .questions import estimate_question_complexity, passage_budget


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


class OneShotRAG:
    """Retrieve once, answer with one chat call over numbered sources"""

    def __init__(self, model_name: str, retrieval: str = 'zim', store: Optional[ZimStore] = None,
                 kiwix: Optional[KiwixServer] = None, selection_model: Optional[str] = None):
        """
        Args:
            model_name: model that writes the answer
            retrieval: 'zim' (needs store) or 'kiwix' (needs a connected kiwix server and selection_model)
            store: ZIM to retrieve passages from
            kiwix: kiwix-serve for v1 search and for source links (optional with 'zim')
            selection_model: model that picks articles from abstracts ('kiwix' only)
        """
        if retrieval not in ('zim', 'kiwix'):
            raise ValueError(f"Unknown retrieval mode: {retrieval}")
        if retrieval == 'zim' and store is None:
            raise ValueError("zim retrieval needs a ZimStore")
        if retrieval == 'kiwix' and (kiwix is None or selection_model is None):
            raise ValueError("kiwix retrieval needs a kiwix server and a selection model")
        self.model_name = model_name
        self.retrieval = retrieval
        self.store = store
        self.kiwix = kiwix
        self.kiwix_url = kiwix.url if kiwix else None
        self.selection_model = selection_model

    def _article_url(self, title: str) -> Optional[str]:
        return self.kiwix.article_url(title) if self.kiwix else None

    def _chat(self, model: str, prompt: str, options: Dict, stage: str):
        """Single-prompt chat (see llm.chat)"""
        return llm.chat(model, [{'role': 'user', 'content': prompt}], options, stage)

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

    def query_with_rag(self, question: str, max_results: int = None) -> Dict:
        """
        Answer question using RAG with local Wikipedia

        Args:
            question: User's question
            max_results: Passages (zim retrieval) or articles (kiwix retrieval) to
                use; auto-detected from question complexity if None

        Returns:
            Dictionary with answer and sources
        """
        start_time = time.time()

        print(f"\n🔍 Searching local Wikipedia for: {question}")
        if self.retrieval == 'zim':
            contents = self._retrieve_passages(question, max_results)
        else:
            contents = self._retrieve_articles(question, max_results)
        if isinstance(contents, dict):  # nothing usable found; already an answer
            return contents

        answer = self._synthesize(question, contents)

        elapsed_time = time.time() - start_time
        print(f"⏱️  Total time: {elapsed_time:.1f}s")

        return {
            'question': question,
            'answer': answer,
            'sources': contents,
            'model': self.model_name,
            'time': elapsed_time
        }

    def _no_answer(self, question: str, answer: str) -> Dict:
        return {'question': question, 'answer': answer, 'sources': [], 'model': self.model_name}

    def _retrieve_passages(self, question: str, max_results: int = None):
        """Section-level chunks straight from the ZIM (see retrieval.ZimStore.retrieve)"""
        if max_results is None:
            max_results = passage_budget(question)
        result = self.store.retrieve(question, k=max_results)
        if not result.chunks:
            return self._no_answer(question, "No relevant Wikipedia articles found in the local ZIM. "
                                             "Try rephrasing your question or using different search terms.")
        if result.title_hits:
            print(f"  🎯 Title matches: {', '.join(title for _, title in result.title_hits)}")
        articles = len({sc.chunk.path for sc in result.chunks})
        print(f"✓ Retrieved {len(result.chunks)} passage(s) from {articles} article(s) "
              f"in {result.timings['total']:.2f}s ({len(result.candidates)} articles scored)")
        contents = []
        for sc in result.chunks:
            chunk = sc.chunk
            print(f"  📄 {chunk.label}")
            contents.append({
                'title': chunk.title,
                'label': chunk.label,
                'content': chunk.text.split('\n', 1)[-1],
                'url': self.kiwix.path_url(chunk.path, chunk.anchor) if self.kiwix else None,
                'chunk_id': chunk.chunk_id,
            })
        return contents

    def _retrieve_articles(self, question: str, max_results: int = None):
        """v1 pipeline: kiwix-serve search, LLM selection over abstracts, first paragraphs of each article"""
        # Auto-detect complexity if not specified
        if max_results is None:
            max_results = estimate_question_complexity(question)

        primary_keywords = self.extract_primary_keywords(question)
        focus_phrases = self.extract_focus_phrases(question)
        if primary_keywords:
            print(f"  🔑 Focus keywords: {', '.join(primary_keywords[:4])}")
        if focus_phrases:
            print(f"  🧭 Focus phrases: {', '.join(focus_phrases[:2])}")

        # Step 1: Search Kiwix (retrieves 3x more results)
        search_results = self.search_kiwix(question, max_results=max_results, primary_keywords=primary_keywords, focus_phrases=focus_phrases)

        if not search_results:
            return self._no_answer(question, "No relevant Wikipedia articles found in local database.")

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

            return self._no_answer(question, suggestion)
        return contents

    def _synthesize(self, question: str, contents: List[Dict]) -> str:
        """Stage 2: one chat call over the numbered sources, with inline [n] citations"""
        passages = self.retrieval == 'zim'
        unit = 'passage' if passages else 'article'

        # Build context with source numbers for citation
        context = "\n\n".join(
            f"[{'Source' if passages else 'Article'} {idx}] **{item.get('label', item['title'])}**:\n{item['content']}"
            for idx, item in enumerate(contents, 1)
        )

        # Build source list for reference
        source_list = "\n".join(f"[{idx}] {item.get('label', item['title'])}" for idx, item in enumerate(contents, 1))

        if passages:
            coverage = ("3. **Relevance**: Passages were retrieved by keyword search; some may be off-topic. "
                        "Use every passage that helps answer the question and ignore the rest.")
        else:
            coverage = "3. **Comprehensiveness**: Integrate information from ALL articles to support the verdict."

        # Create synthesis-optimized prompt for Stage 2
        prompt = f"""You are an expert research analyst synthesizing information from multiple Wikipedia {unit}s.

TASK: Answer the question by synthesizing information from the provided {unit}s.

Question: "{question}"

Available {unit.capitalize()}s:
{source_list}

{unit.capitalize()} Contents:
{context}

SYNTHESIS INSTRUCTIONS:
1. **Direct Verdict**: The first sentence must explicitly answer the question (e.g., "Yes, the film earned overwhelmingly positive reviews for... [1]"). Make the stance clear (yes/no/mixed) before adding context.
2. **Stay On-Task**: Only include details that help judge quality/relevance of the topic. Omit long cast lists or plot summaries unless they support the verdict.
{coverage}
4. **Coherence**: Create a logical narrative that links supporting evidence.
5. **Evidence**: Use concrete facts (awards, box office, critical reception) with citations.
6. **Perspectives**: Note differing viewpoints if present, and explain them.
7. **Structure**: Write in clear paragraphs; use lists only when essential.
8. **Accuracy**: Stay within the provided {unit}s; do not invent data.
9. **Citations**: Add inline citations [1], [2], [3] after EVERY fact drawn from the {unit}s.

CRITICAL - INLINE CITATIONS:
- Add [1], [2], or [3] immediately after each fact, quote, or claim from that {unit}
- Multiple sources: use [1][2] or [1,2] if information appears in multiple {unit}s
- Example: "Bill Murray was born in 1950 [1] and starred in Ghostbusters [1][3]."
- Every paragraph should have multiple citations showing source of information

FORMAT:
- Write natural paragraphs with inline citations only.
- Do NOT repeat the question.
- Do NOT add headings such as "References", "Sources", or "Bibliography"—inline citations are sufficient.
- End the answer immediately after the final paragraph (no trailing lists or sections).

Your synthesized answer with inline citations (stop after final paragraph):"""

        print(f"🤖 Generating synthesis with {self.model_name}...")

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
            pattern = r'\n\s*\[?(References?|Sources?|Bibliography)\]?[:\-]?\s*(\n.*)?$'
            return re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE).rstrip()

        except Exception as e:
            print(f"  ⚠ Generation error: {e}")
            return "Error generating answer. Please try again."
