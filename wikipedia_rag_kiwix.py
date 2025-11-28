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
    
    def search_kiwix(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Search local Wikipedia via Kiwix
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results with titles and URLs
        """
        try:
            # Kiwix search endpoint
            search_url = f"{self.kiwix_url}/search"
            params = {
                'pattern': query,
                'pageSize': max_results
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse search results - Kiwix uses <ul class="results">
            results = []
            results_div = soup.find('div', class_='results')
            if results_div:
                for li in results_div.find_all('li')[:max_results]:
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
    
    def fetch_article(self, url: str) -> str:
        """
        Fetch article content from Kiwix
        
        Args:
            url: Article URL
            
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
                # Extract all paragraphs (not just direct children)
                paragraphs = content.find_all('p')
                
                # Filter out empty paragraphs and get text
                texts = []
                for p in paragraphs[:10]:  # First 10 paragraphs
                    text = p.get_text(strip=True)
                    if len(text) > 50:  # Only meaningful paragraphs
                        texts.append(text)
                
                combined = '\n\n'.join(texts)
                
                # Clean up text
                combined = re.sub(r'\[\d+\]', '', combined)  # Remove citation numbers
                combined = re.sub(r'\s+', ' ', combined)  # Normalize whitespace
                
                return combined[:3000]  # Limit to 3000 chars
            
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
            Number of articles to retrieve (1-5)
        """
        question_lower = question.lower()
        
        # Complex question indicators
        complexity_score = 0
        
        # Multi-part questions
        if ' and ' in question_lower or ' vs ' in question_lower or ' versus ' in question_lower:
            complexity_score += 2
        
        # Comparison/relationship questions
        if any(word in question_lower for word in ['compare', 'difference', 'relationship', 'connect', 'relate', 'impact', 'affect', 'influence']):
            complexity_score += 2
        
        # Broad/conceptual questions
        if any(word in question_lower for word in ['how does', 'how do', 'why', 'explain', 'history of', 'overview']):
            complexity_score += 1
        
        # Long questions often need more context
        if len(question.split()) > 10:
            complexity_score += 1
        
        # Map complexity to number of articles
        if complexity_score >= 4:
            return 5  # Very complex - retrieve 5 articles
        elif complexity_score >= 2:
            return 4  # Complex - retrieve 4 articles
        elif complexity_score >= 1:
            return 3  # Moderate - retrieve 3 articles
        else:
            return 2  # Simple - retrieve 2 articles
    
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
        
        # Search Kiwix
        search_results = self.search_kiwix(question, max_results=max_results)
        
        if not search_results:
            return {
                'question': question,
                'answer': "No relevant Wikipedia articles found in local database.",
                'sources': [],
                'model': self.model_name
            }
        
        print(f"✓ Found {len(search_results)} article(s)")
        
        # Fetch article contents
        contents = []
        for result in search_results:
            print(f"  📄 Fetching: {result['title']}")
            content = self.fetch_article(result['url'])
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
        
        # Build context
        context_parts = []
        for item in contents:
            context_parts.append(f"**{item['title']}**:\n{item['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a knowledgeable assistant. Using the Wikipedia content below, provide a clear, well-structured answer.

FORMAT YOUR RESPONSE LIKE THIS:
- Use clear paragraphs (not bullet points unless listing items)
- Write in a natural, conversational style
- Include key facts and context
- Be concise but complete
- Do NOT include citations like [1] or footnotes
- Do NOT repeat the question

Wikipedia Content:
{context}

Question: {question}

Provide a clear, well-written answer:"""
        
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
                print("📚 Sources: " + ", ".join([s['title'] for s in result['sources']]))
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
            
            print("\n" + "="*70)
            print(f"📚 Sources: {', '.join([s['title'] for s in result['sources']])}")
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
