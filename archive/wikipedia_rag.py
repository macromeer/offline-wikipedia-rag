#!/usr/bin/env python3
"""
Wikipedia RAG System with Ollama
Retrieves relevant Wikipedia content and uses Ollama for generation
"""

import ollama
import wikipedia
import chromadb
from typing import List, Dict
import argparse


class WikipediaRAG:
    """RAG system that connects Ollama with Wikipedia"""
    
    def __init__(self, model_name: str = "llama2", use_cache: bool = True):
        """
        Initialize the RAG system
        
        Args:
            model_name: Name of the Ollama model to use
            use_cache: Whether to cache Wikipedia queries
        """
        self.model_name = model_name
        self.use_cache = use_cache
        
        # Initialize ChromaDB for caching if enabled
        if use_cache:
            self.chroma_client = chromadb.Client()
            try:
                self.collection = self.chroma_client.get_collection("wikipedia_cache")
            except:
                self.collection = self.chroma_client.create_collection("wikipedia_cache")
        
        print(f"✓ Initialized Wikipedia RAG with model: {model_name}")
    
    def search_wikipedia(self, query: str, max_results: int = 3) -> List[str]:
        """
        Search Wikipedia and return relevant content
        
        Args:
            query: Search query
            max_results: Maximum number of articles to retrieve
            
        Returns:
            List of article summaries
        """
        try:
            # Search for relevant articles
            search_results = wikipedia.search(query, results=max_results)
            
            if not search_results:
                return [f"No Wikipedia articles found for query: {query}"]
            
            contents = []
            for title in search_results[:max_results]:
                try:
                    # Get article summary
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = page.summary[:1000]  # Limit to 1000 chars
                    contents.append(f"**{title}**:\n{summary}")
                    
                    # Cache the content if enabled
                    if self.use_cache:
                        self.collection.add(
                            documents=[summary],
                            metadatas=[{"title": title, "url": page.url}],
                            ids=[f"wiki_{title}"]
                        )
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    contents.append(f"**{title}**: Multiple articles found. Suggestions: {', '.join(e.options[:3])}")
                except wikipedia.exceptions.PageError:
                    continue
                except Exception as e:
                    print(f"Warning: Could not retrieve {title}: {e}")
                    continue
            
            return contents if contents else [f"Could not retrieve content for: {query}"]
            
        except Exception as e:
            return [f"Error searching Wikipedia: {str(e)}"]
    
    def query_with_rag(self, question: str, max_wiki_results: int = 3) -> Dict:
        """
        Answer a question using RAG with Wikipedia
        
        Args:
            question: User's question
            max_wiki_results: Number of Wikipedia articles to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        print(f"\n🔍 Searching Wikipedia for: {question}")
        
        # Retrieve relevant Wikipedia content
        wiki_contents = self.search_wikipedia(question, max_results=max_wiki_results)
        
        # Build context from Wikipedia
        context = "\n\n".join(wiki_contents)
        
        # Create prompt with context
        prompt = f"""You are a helpful assistant. Use the following Wikipedia content to answer the question accurately. If the content doesn't contain the answer, say so.

Wikipedia Content:
{context}

Question: {question}

Answer based on the Wikipedia content above:"""
        
        print(f"✓ Retrieved {len(wiki_contents)} Wikipedia article(s)")
        print(f"🤖 Generating answer with {self.model_name}...")
        
        # Query Ollama with context
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
            'sources': wiki_contents,
            'model': self.model_name
        }
    
    def interactive_mode(self):
        """Run an interactive Q&A session"""
        print("\n" + "="*60)
        print("Wikipedia RAG with Ollama - Interactive Mode")
        print("="*60)
        print(f"Model: {self.model_name}")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Get answer with RAG
                result = self.query_with_rag(question)
                
                print("\n" + "="*60)
                print(f"📖 Answer:\n{result['answer']}")
                print("\n" + "-"*60)
                print(f"📚 Sources: {len(result['sources'])} Wikipedia article(s)")
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Wikipedia RAG with Ollama')
    parser.add_argument('--model', type=str, default=None, 
                        help='Ollama model name (default: auto-detect first available model)')
    parser.add_argument('--question', type=str, 
                        help='Single question to ask (otherwise starts interactive mode)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable Wikipedia caching')
    
    args = parser.parse_args()
    
    # Auto-detect model if not specified
    model_name = args.model
    if not model_name:
        try:
            response = ollama.list()
            models = response.models if hasattr(response, 'models') else []
            if models:
                model_name = models[0].model
                print(f"🤖 Auto-detected model: {model_name}")
            else:
                print("❌ No Ollama models found. Install one with: ollama pull llama2")
                return
        except Exception as e:
            print(f"❌ Could not connect to Ollama: {e}")
            print("💡 Make sure Ollama is running: ollama serve")
            return
    
    # Initialize RAG system
    rag = WikipediaRAG(model_name=model_name, use_cache=not args.no_cache)
    
    if args.question:
        # Single question mode
        result = rag.query_with_rag(args.question)
        print("\n" + "="*60)
        print(f"Question: {result['question']}")
        print(f"\nAnswer:\n{result['answer']}")
        print("\n" + "-"*60)
        print(f"Sources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source[:200]}...")
        print("="*60)
    else:
        # Interactive mode
        rag.interactive_mode()


if __name__ == "__main__":
    main()
