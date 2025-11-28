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


class KiwixWikipediaRAG:
    """RAG system using local Kiwix Wikipedia server"""
    
    def __init__(self, model_name: str = None, kiwix_url: str = "http://localhost:8080"):
        """
        Initialize the Kiwix RAG system
        
        Args:
            model_name: Ollama model name (auto-detects if None)
            kiwix_url: URL of the Kiwix server
        """
        # Auto-detect model if not specified
        if not model_name:
            try:
                response = ollama.list()
                models = response.models if hasattr(response, 'models') else []
                # Filter out qwen models
                models = [m for m in models if 'qwen' not in m.model.lower()]
                
                if models:
                    model_name = models[0].model
                    print(f"🤖 Auto-detected model: {model_name}")
                else:
                    raise Exception("No suitable Ollama models found")
            except Exception as e:
                raise Exception(f"Could not detect Ollama model: {e}")
        
        self.model_name = model_name
        self.kiwix_url = kiwix_url.rstrip('/')
        
        # Test Kiwix connection
        try:
            response = requests.get(f"{self.kiwix_url}/", timeout=5)
            response.raise_for_status()
            print(f"✓ Connected to Kiwix server at {self.kiwix_url}")
        except Exception as e:
            raise Exception(f"Could not connect to Kiwix server at {self.kiwix_url}: {e}")
        
        print(f"✓ Initialized with model: {self.model_name}")
    def extract_search_terms(self, question: str) -> str:
        """
        Extract key search terms from a natural language question
        
        Args:
            question: User's natural language question
            
        Returns:
            Optimized search query
        """
        # Remove question words and common phrases
        stopwords = ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were', 
                    'does', 'do', 'did', 'can', 'could', 'would', 'should', 'will', 'the', 'a', 'an',
                    'and', 'or', 'but', 'its', 'it', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                    'tell', 'me', 'explain', 'describe', 'about', 'be', 'have', 'has', 'had', 'old']
        
        # Important words to prioritize (keep them even if they might seem like stopwords)
        important_terms = {
            'age': 'age',
            'old': 'age',
            'future': 'future',
            'past': 'history',
            'origin': 'origin',
            'beginning': 'origin',
            'end': 'future',
        }
        
        # Clean and split
        words = question.lower().split()
        
        # Filter out stopwords and punctuation, apply mappings
        keywords = []
        for w in words:
            # Remove punctuation
            w_clean = w.strip('?.,!:;')
            
            # Check if it's an important term to map
            if w_clean in important_terms:
                keywords.append(important_terms[w_clean])
            # Keep if not a stopword and length > 2
            elif w_clean not in stopwords and len(w_clean) > 2:
                keywords.append(w_clean)
        
        # Join remaining keywords (limit to 4 most important words)
        search_query = ' '.join(keywords[:4])
        
        return search_query if search_query else question
    
    def search_kiwix(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Search local Wikipedia via Kiwix
        
        Args:
            query: Search query
            max_results: Maximum number of results to retrieve initially (will be filtered by AI)
            
        Returns:
            List of search results with titles and URLs
        """
        try:
            # Extract key search terms for better results
            search_query = self.extract_search_terms(query)
            print(f"  🔎 Search terms: {search_query}")
            
            # Retrieve MORE results than needed (will filter with AI)
            initial_results = max_results * 3  # 3x overfetch
            
            # Kiwix search endpoint
            search_url = f"{self.kiwix_url}/search"
            params = {
                'pattern': search_query,
                'pageSize': initial_results
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse search results - Kiwix uses <ul class="results">
            results = []
            results_div = soup.find('div', class_='results')
            if results_div:
                for li in results_div.find_all('li')[:initial_results]:
                    link = li.find('a')
                    if link and link.get('href'):
                        title = link.get_text(strip=True)
                        url = link['href']
                        if not url.startswith('http'):
                            full_url = f"{self.kiwix_url}{url}"
                        else:
                            full_url = url
                        results.append({'title': title, 'url': full_url})
            
            return results
            
        except Exception as e:
            print(f"⚠ Search error: {e}")
            return []
    
    def select_relevant_articles(self, question: str, search_results: List[Dict], target_count: int) -> List[Dict]:
        """
        Use AI to select the most relevant articles from search results
        
        Args:
            question: User's question
            search_results: List of article titles and URLs from search
            target_count: Number of articles to select
            
        Returns:
            Filtered list of most relevant articles
        """
        if len(search_results) <= target_count:
            return search_results
        
        # Build list of article titles
        titles_list = "\n".join([f"{i+1}. {r['title']}" for i, r in enumerate(search_results)])
        
        # Ask AI to select most relevant
        selection_prompt = f"""You are helping select the most relevant Wikipedia articles to answer a question.

Question: {question}

Available articles:
{titles_list}

Task: Select the {target_count} MOST relevant articles that would help answer this question.
Return ONLY the numbers (e.g., "1, 5, 7, 12") - no explanations, just comma-separated numbers.

Selected article numbers:"""
        
        try:
            print(f"  🤖 AI selecting {target_count} most relevant from {len(search_results)} articles...")
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': selection_prompt
                }]
            )
            
            # Parse response to get article indices
            answer = response['message']['content'].strip()
            # Extract numbers from response
            import re
            numbers = re.findall(r'\d+', answer)
            selected_indices = [int(n) - 1 for n in numbers[:target_count]]  # Convert to 0-based
            
            # Filter to valid indices
            selected_indices = [i for i in selected_indices if 0 <= i < len(search_results)]
            
            # Return selected articles
            selected = [search_results[i] for i in selected_indices]
            
            if selected:
                print(f"  ✓ AI selected: {', '.join([s['title'] for s in selected])}")
                return selected
            else:
                # Fallback to first N if parsing failed
                print(f"  ⚠ AI selection failed, using first {target_count} results")
                return search_results[:target_count]
                
        except Exception as e:
            print(f"  ⚠ AI selection error: {e}, using first {target_count} results")
            return search_results[:target_count]
    
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
                for p in paragraphs[:para_limit]:
                    text = p.get_text(strip=True)
                    if len(text) > 50:  # Only meaningful paragraphs
                        texts.append(text)
                
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
        
        # Map complexity to number of articles (increased range)
        if complexity_score >= 6:
            return 7  # Very complex - retrieve 7 articles
        elif complexity_score >= 4:
            return 6  # Complex - retrieve 6 articles
        elif complexity_score >= 3:
            return 5  # Moderate-complex - retrieve 5 articles
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
        
        # Step 2: Use AI to select most relevant articles
        selected_results = self.select_relevant_articles(question, search_results, max_results)
        
        print(f"✓ Selected {len(selected_results)} relevant article(s)")
        
        # Balance content depth with article count
        # More articles = less content per article (to keep total context reasonable)
        paragraphs_per_article = {
            3: 30,   # 3 articles: read ~30 paragraphs each
            4: 25,   # 4 articles: read ~25 paragraphs each
            5: 20,   # 5 articles: read ~20 paragraphs each
            6: 15,   # 6 articles: read ~15 paragraphs each
            7: 12,   # 7 articles: read ~12 paragraphs each
        }
        max_paragraphs = paragraphs_per_article.get(len(selected_results), 20)
        print(f"  📊 Reading ~{max_paragraphs} paragraphs per article")
        
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
        
        # Create enhanced prompt
        prompt = f"""You are an expert researcher with access to multiple Wikipedia articles. Your task is to synthesize information from ALL the provided articles to give a comprehensive, accurate answer.

INSTRUCTIONS:
1. Read and understand ALL the Wikipedia articles provided below
2. Identify connections, patterns, and relationships across the articles
3. Synthesize the information into a coherent, well-structured answer
4. Include specific facts, dates, numbers, and key concepts from the articles
5. If articles present different perspectives, acknowledge and explain them
6. Write in clear, natural paragraphs (avoid bullet points unless listing items)
7. Be thorough but avoid unnecessary repetition
8. Do NOT repeat the question in your answer

IMPORTANT - CITING SOURCES:
After your answer, add a "Sources:" section listing which articles you used.
Format: "Sources: [1], [2], [3]" (list the article numbers you referenced)

Available Articles:
{source_list}

Wikipedia Articles:
{context}

Question: {question}

Provide a comprehensive answer followed by your sources:"""
        
        print(f"🤖 Generating answer with {self.model_name}...")
        
        # Query Ollama
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )
        
        answer = response['message']['content']
        
        return {
            'question': question,
            'answer': answer,
            'sources': contents,
            'model': self.model_name
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
                        print(f"   {line}")
                    else:
                        print()
                
                print("\n" + "-"*70)
                print("📚 Retrieved Articles:")
                for idx, s in enumerate(result['sources'], 1):
                    print(f"   [{idx}] {s['title']}")
                print("="*70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Offline Wikipedia RAG with Kiwix')
    parser.add_argument('--model', type=str, default=None,
                        help='Ollama model name (auto-detects, excludes qwen)')
    parser.add_argument('--kiwix-url', type=str, default='http://localhost:8080',
                        help='Kiwix server URL')
    parser.add_argument('--question', type=str,
                        help='Single question (otherwise interactive mode)')
    parser.add_argument('--max-results', type=int, default=3,
                        help='Number of Wikipedia articles to retrieve')
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG
        rag = KiwixWikipediaRAG(model_name=args.model, kiwix_url=args.kiwix_url)
        
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
            print("📚 Retrieved Articles:")
            for idx, s in enumerate(result['sources'], 1):
                print(f"   [{idx}] {s['title']}")
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
