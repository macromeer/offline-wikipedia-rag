#!/usr/bin/env python3
"""
Full Offline Wikipedia RAG using Kiwix Server
Requires: Kiwix server running with Wikipedia ZIM file
"""

import ollama
import requests
from bs4 import BeautifulSoup
import argparse
from typing import List, Dict
import re
import time


class KiwixWikipediaRAG:
    """RAG system using local Kiwix Wikipedia server with two-stage AI pipeline"""
    
    def __init__(self, model_name: str = None, selection_model: str = None, kiwix_url: str = "http://localhost:8080"):
        """
        Initialize the Kiwix RAG system with specialized models
        
        Args:
            model_name: Summarization model name (auto-detects if None)
            selection_model: Article selection model name (auto-detects if None)
            kiwix_url: URL of the Kiwix server
        """
        self.kiwix_url = kiwix_url.rstrip('/')
        
        # Test Kiwix connection
        try:
            response = requests.get(f"{self.kiwix_url}/", timeout=5)
            response.raise_for_status()
            print(f"✓ Connected to Kiwix server at {self.kiwix_url}")
        except Exception as e:
            raise Exception(f"Could not connect to Kiwix server at {self.kiwix_url}: {e}")
        
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
    
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = ollama.list()
            if hasattr(response, 'models'):
                return [m.model for m in response.models]
            elif isinstance(response, dict) and 'models' in response:
                return [m['name'] if isinstance(m, dict) else m.model for m in response['models']]
            return []
        except Exception as e:
            print(f"⚠ Could not list models: {e}")
            return []
    
    def _detect_selection_model(self, preferred: str, available: List[str]) -> str:
        """
        Detect best model for article selection (classification task)
        
        Research shows specialized models perform better:
        - Qwen2.5-32B: 92% classification accuracy, superior instruction-following
        - Mistral-Small: 88% accuracy, fastest inference
        - Hermes-3-8B: 78% accuracy, best for resource-constrained environments
        - Avoid: DeepSeek R1 (reasoning models fail at simple tasks)
        """
        if preferred:
            return preferred
        
        # Priority order based on performance benchmarks
        selection_preferences = [
            'qwen2.5:32b-instruct',
            'qwen2.5:32b',
            'qwen2.5:14b-instruct',
            'qwen2.5:14b',
            'qwen2.5:7b-instruct',
            'mistral-small:latest',
            'mistral-small',
            'mistral:7b',           # Mistral-7B excellent for classification
            'hermes3:8b',
            'hermes3:latest',
            'llama3.2:3b',
            'llama3.1:8b',          # Fallback: use for synthesis instead
            'phi3:medium',
        ]
        
        # Find first available model
        for model in selection_preferences:
            if model in available:
                return model
            # Check partial matches (e.g., 'qwen2.5' matches 'qwen2.5:32b-instruct-q4_K_M')
            base_name = model.split(':')[0]
            for avail in available:
                if avail.startswith(base_name):
                    return avail
        
        # Last resort: use first available non-reasoning model
        for model in available:
            if 'r1' not in model.lower() and 'deepseek' not in model.lower():
                print(f"⚠ Using fallback selection model: {model}")
                return model
        
        raise Exception("No suitable models found for article selection")
    
    def _detect_summarization_model(self, preferred: str, available: List[str]) -> str:
        """
        Detect best model for summarization (synthesis task)
        
        Prioritizes practical models for most users:
        - Llama-3.1-8B: Excellent balance of quality and resource usage
        - Gemma-2-27B/9B: Exceptional summarization quality with fast inference
        - Mistral-7B: Fast and reliable
        - Llama-3.1-70B: Optional for users with high-end hardware
        """
        if preferred:
            return preferred
        
        # Priority order: practical models first, larger models for power users
        summarization_preferences = [
            'llama3.1:8b-instruct',
            'llama3.1:8b',
            'gemma2:27b',
            'gemma2:9b',
            'mistral:7b',
            'granite3.1-dense:8b',
            'qwen2.5:7b',
            'llama3.3:70b',        # Optional: for power users
            'llama3.1:70b-instruct',
            'llama3.1:70b',
        ]
        
        # Find first available model
        for model in summarization_preferences:
            if model in available:
                return model
            # Check partial matches
            base_name = model.split(':')[0]
            for avail in available:
                if avail.startswith(base_name):
                    return avail
        
        # Last resort: use first available model
        if available:
            print(f"⚠ Using fallback summarization model: {available[0]}")
            return available[0]
        
        raise Exception("No Ollama models found")
    
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
        
        # Remove question words and common stopwords
        stopwords = {'what', 'when', 'where', 'who', 'whom', 'whose', 'why', 'which', 'how',
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
                    # Add common verbs that shouldn't be search terms
                    'become', 'became', 'get', 'got', 'make', 'made', 'take', 'took'}
        
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
        for word in content_words[:3]:
            title = word.capitalize()
            if title not in terms and title not in proper_nouns:
                terms.append(title)
        
        # Strategy 3: Try consecutive word pairs from content words
        for i in range(min(2, len(content_words) - 1)):
            phrase = f"{content_words[i].capitalize()} {content_words[i+1]}"
            if phrase not in terms:
                terms.append(phrase)
        
        return terms[:5] if terms else [question]
    
    def search_kiwix(self, query: str, max_results: int = 25) -> List[Dict]:
        """
        Search local Wikipedia via Kiwix using Wikipedia title conventions
        
        Retrieves more results and uses multiple search strategies to find main articles
        (e.g., "Earthquake" article comes after "List of earthquakes..." in alphabetical order)
        
        Args:
            query: Search query (user's question)
            max_results: Maximum number of results to retrieve per search term
            
        Returns:
            List of search results with titles and URLs
        """
        try:
            # Extract Wikipedia-style article titles
            print(f"  🤖 Extracting Wikipedia article titles...")
            search_terms = self.extract_search_terms(query)
            
            print(f"  🔎 Searching for articles:")
            
            all_results = []
            seen_titles = set()
            
            # Strategy 1: Search each extracted term
            for term in search_terms:
                if len(all_results) >= 100:  # Increased cap for better selection
                    break
                print(f"    - '{term}'")
                results = self._do_search(term, max_results)
                for r in results:
                    # Case-insensitive duplicate detection
                    title_lower = r['title'].lower()
                    if title_lower not in seen_titles:
                        all_results.append(r)
                        seen_titles.add(title_lower)
            
            # Strategy 2: Direct lookup for main article (singular form)
            # This helps find "Earthquake" even when lists come first alphabetically
            for term in search_terms[:3]:  # Try first 3 terms as direct lookups
                if len(all_results) >= 100:
                    break
                # Try exact match by requesting the article directly
                direct_url = f"{self.kiwix_url}/wikipedia_en_all_maxi_2024-01/A/{term.replace(' ', '_')}"
                try:
                    response = requests.head(direct_url, timeout=2, allow_redirects=True)
                    if response.status_code == 200:
                        title = term
                        title_lower = title.lower()
                        if title_lower not in seen_titles:
                            all_results.insert(0, {'title': title, 'url': direct_url})  # Insert at beginning
                            seen_titles.add(title_lower)
                            print(f"    + Direct: '{title}'")
                except:
                    pass  # Article doesn't exist, that's OK
            
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
    
    def select_relevant_articles(self, question: str, search_results: List[Dict], target_count: int) -> List[Dict]:
        """
        Stage 1: Use specialized classification model for article selection
        
        Now uses article abstracts (first paragraphs) to make informed decisions
        about relevance. Uses Mistral-7B or similar for reliable classification.
        
        Args:
            question: User's question
            search_results: List of article titles, URLs, and abstracts from search
            target_count: Number of articles to select
            
        Returns:
            Filtered list of most relevant articles
        """
        if len(search_results) <= target_count:
            return search_results
        
        # Build article list with abstracts for informed classification
        # Only include articles with meaningful content (filter out empty stubs)
        articles_text = ""
        article_index_map = {}  # Map displayed number to actual index
        display_num = 1
        
        for i, result in enumerate(search_results[:30]):  # Limit to first 30 for prompt length
            title = result['title']
            abstract = result.get('abstract', '')
            
            # Only include articles with meaningful abstracts (>50 chars)
            if abstract and len(abstract) > 50:
                article_index_map[display_num] = i
                # Truncate abstract to keep prompt manageable
                abstract_preview = abstract[:200] + "..." if len(abstract) > 200 else abstract
                articles_text += f"{display_num}. **{title}**\n   {abstract_preview}\n\n"
                display_num += 1
        
        if not articles_text:
            # Fallback: include all if filtering was too aggressive
            for i, result in enumerate(search_results[:15]):
                article_index_map[i+1] = i
                title = result['title']
                articles_text += f"{i+1}. **{title}**\n\n"
        
        print(f"  🤖 Selecting with {self.selection_model} (using article abstracts)...")
        
        # Classification-optimized prompt with content-based selection
        selection_prompt = f"""You are a research assistant selecting Wikipedia articles to answer a question.

Question: "{question}"

Available articles with previews:
{articles_text}

Task: Select the {target_count} MOST RELEVANT articles that directly answer the question.

Selection criteria:
✓ Choose articles whose content DIRECTLY addresses the question topic
✓ Prefer biographical articles for "who is" questions
✓ Prefer main topic articles with encyclopedic information
✗ Avoid: Lists, year-specific articles, episode lists, song lists, disambiguation pages
✗ Avoid: Tangentially related topics (e.g., songs/shows about a person vs. the person themselves)

Important: Base decisions on content relevance, not just keyword matching in titles.

Output ONLY the article numbers, comma-separated (example: "3,7,12"):
"""
        
        try:
            response = ollama.chat(
                model=self.selection_model,
                messages=[{'role': 'user', 'content': selection_prompt}],
                options={
                    'num_predict': 200,
                    'temperature': 0.2,  # Very low temperature for consistent classification
                    'top_p': 0.9,
                }
            )
            
            # Parse article numbers from response
            answer = response['message']['content'].strip()
            print(f"  📋 Selection output: '{answer}'")
            
            # Handle empty response
            if not answer:
                print(f"  ⚠ Selection model returned empty response, using fallback")
            else:
                numbers = re.findall(r'\d+', answer)
                print(f"  📋 Extracted numbers: {numbers}")
                
                # Map displayed numbers back to actual indices, removing duplicates
                seen_indices = set()
                indices = []
                for n in numbers:
                    num = int(n)
                    if num in article_index_map:
                        actual_idx = article_index_map[num]
                        if actual_idx not in seen_indices:
                            indices.append(actual_idx)
                            seen_indices.add(actual_idx)
                
                indices = indices[:target_count]  # Limit to target count
                
                if indices:
                    selected = [search_results[i] for i in indices]
                    print(f"  ✓ AI selected {len(selected)} articles: {', '.join([s['title'] for s in selected])}")
                    
                    # Accept selection if we got at least 1 good article
                    # The AI might rightfully reject poor candidates (lists, etc.)
                    if len(selected) >= 1:
                        return selected
                
                print(f"  ⚠ Selection returned no valid results, using fallback")
                    
        except Exception as e:
            print(f"  ⚠ Selection error: {e}")
        
        # Fallback: rule-based filtering
        filtered = [r for r in search_results if not (
            r['title'].lower().startswith('list of') or
            r['title'].lower().startswith('lists of') or
            re.search(r'\b(19|20)\d{2}\b', r['title']) or
            'disambiguation' in r['title'].lower()
        )]
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
            print(f"🧠 Question complexity: retrieving {max_results} article(s)")
        
        print(f"\n🔍 Searching local Wikipedia for: {question}")
        
        # Step 1: Search Kiwix (retrieves 3x more results)
        search_results = self.search_kiwix(question, max_results=max_results)
        
        if not search_results:
            return {
                'question': question,
                'answer': "No relevant Wikipedia articles found in local database.",
                'sources': [],
                'model': self.model_name
            }
        
        print(f"✓ Found {len(search_results)} candidate article(s)")
        
        # Step 1.5: Fetch abstracts for better selection (first paragraph only)
        print(f"  📄 Fetching article abstracts for selection...")
        articles_with_content = 0
        for i, result in enumerate(search_results):
            if i >= 30:  # Limit abstract fetching to first 30 for speed
                break
            abstract = self.fetch_article_abstract(result['url'])
            result['abstract'] = abstract
            
            # Show articles with meaningful content
            if abstract and len(abstract) > 50:
                articles_with_content += 1
                if articles_with_content <= 8:  # Show first 8 with content
                    preview = abstract[:80] + "..." if len(abstract) > 80 else abstract
                    print(f"    {i+1}. {result['title']}: {preview}")
        
        # Step 2: Use AI to select most relevant articles with context
        selected_results = self.select_relevant_articles(question, search_results, max_results)
        
        print(f"✓ Selected {len(selected_results)} relevant article(s)")
        
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
            return {
                'question': question,
                'answer': "Could not retrieve article content.",
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
1. **Comprehensiveness**: Integrate information from ALL articles to provide a complete answer
2. **Coherence**: Create a logical narrative that connects concepts across articles
3. **Evidence**: Include specific facts, dates, numbers, and key concepts
4. **Perspectives**: If articles present different viewpoints, acknowledge and explain them
5. **Structure**: Write in clear paragraphs; use lists only when appropriate
6. **Accuracy**: Maintain factual consistency; do not infer beyond the articles
7. **Clarity**: Be thorough yet concise; avoid unnecessary repetition
8. **Citations**: Add inline citations [1], [2], [3] after EVERY fact or statement from the articles

CRITICAL - INLINE CITATIONS:
- Add [1], [2], or [3] immediately after each fact, quote, or claim from that article
- Multiple sources: use [1][2] or [1,2] if information appears in multiple articles
- Example: "Bill Murray was born in 1950 [1] and starred in Ghostbusters [1][3]."
- Every paragraph should have multiple citations showing source of information

FORMAT:
- Write your answer in natural, readable paragraphs with inline citations
- Do NOT repeat the question in your answer
- Do NOT add a separate "Sources:" section at the end (citations are inline)

Your synthesized answer with inline citations:"""
        
        print(f"🤖 Generating synthesis with {self.model_name}...")
        
        # Query summarization model with optimized settings
        # Llama-3.1-70B: 3x faster inference, excellent coherent generation
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'num_predict': 1500,   # Allow comprehensive answers
                    'temperature': 0.7,    # Balance factual accuracy with coherence
                    'top_p': 0.9,
                    'repeat_penalty': 1.1, # Reduce repetition in synthesis
                }
            )
            
            answer = response['message']['content']
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


def main():
    parser = argparse.ArgumentParser(description='Offline Wikipedia RAG with Kiwix - Two-Stage AI Pipeline')
    parser.add_argument('--model', type=str, default=None,
                        help='Summarization model (auto-detects: llama3.1:70b, gemma2:27b)')
    parser.add_argument('--selection-model', type=str, default=None,
                        help='Selection model (auto-detects: qwen2.5:32b, mistral-small)')
    parser.add_argument('--kiwix-url', type=str, default='http://localhost:8080',
                        help='Kiwix server URL')
    parser.add_argument('--question', type=str,
                        help='Single question (otherwise interactive mode)')
    parser.add_argument('--max-results', type=int, default=None,
                        help='Number of Wikipedia articles to retrieve (auto-detects by complexity)')
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG with two-stage pipeline
        rag = KiwixWikipediaRAG(
            model_name=args.model,
            selection_model=args.selection_model,
            kiwix_url=args.kiwix_url
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
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure:")
        print("   1. Kiwix server is running: kiwix-serve ~/wikipedia-offline/*.zim")
        print("   2. Ollama is running: ollama serve")
        return 1


if __name__ == "__main__":
    main()
