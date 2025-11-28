#!/usr/bin/env python3
"""
Advanced Wikipedia RAG with local vector database
Uses pre-indexed Wikipedia content for fully offline operation
"""

import ollama
import chromadb
from chromadb.utils import embedding_functions
import argparse
from typing import List, Dict


class OfflineWikipediaRAG:
    """RAG system using local Wikipedia index"""
    
    def __init__(self, model_name: str = "llama2", db_path: str = "./wikipedia_db"):
        """
        Initialize the offline RAG system
        
        Args:
            model_name: Ollama model name
            db_path: Path to Wikipedia database
        """
        self.model_name = model_name
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Use sentence transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        try:
            self.collection = self.chroma_client.get_collection(
                "wikipedia",
                embedding_function=self.embedding_function
            )
            print(f"✓ Loaded Wikipedia index from: {db_path}")
            print(f"✓ Total chunks in database: {self.collection.count()}")
        except Exception as e:
            print(f"❌ Error: Could not load Wikipedia database from {db_path}")
            print(f"   {e}")
            print(f"\n💡 Tip: Run wikipedia_indexer.py first to build the index")
            raise
        
        print(f"✓ Initialized with model: {model_name}")
    
    def search_local_wikipedia(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search local Wikipedia index
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant content chunks with metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            formatted_results.append({
                'content': doc,
                'title': meta.get('title', 'Unknown'),
                'url': meta.get('url', ''),
                'relevance': 1 - dist  # Convert distance to relevance score
            })
        
        return formatted_results
    
    def query_with_rag(self, question: str, n_results: int = 5) -> Dict:
        """
        Answer question using RAG with local Wikipedia
        
        Args:
            question: User's question
            n_results: Number of Wikipedia chunks to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        print(f"\n🔍 Searching local Wikipedia for: {question}")
        
        # Retrieve relevant content
        results = self.search_local_wikipedia(question, n_results=n_results)
        
        if not results:
            return {
                'question': question,
                'answer': "No relevant Wikipedia content found in local database.",
                'sources': [],
                'model': self.model_name
            }
        
        # Build context
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result['title']} (relevance: {result['relevance']:.2f})]\n"
                f"{result['content']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant. Use the following Wikipedia content to answer the question accurately. If the content doesn't fully answer the question, provide what information is available and mention what's missing.

Wikipedia Content:
{context}

Question: {question}

Answer based on the Wikipedia content above:"""
        
        print(f"✓ Retrieved {len(results)} relevant chunk(s)")
        print(f"   Top source: {results[0]['title']} (relevance: {results[0]['relevance']:.2f})")
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
            'sources': results,
            'model': self.model_name
        }
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*60)
        print("Offline Wikipedia RAG with Ollama")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Database: {self.collection.count()} chunks indexed")
        print("Type 'quit' or 'exit' to stop\n")
        
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
                
                print("\n" + "="*60)
                print(f"📖 Answer:\n{result['answer']}")
                print("\n" + "-"*60)
                print(f"📚 Sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"   {i}. {source['title']} (relevance: {source['relevance']:.2f})")
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Offline Wikipedia RAG with Ollama')
    parser.add_argument('--model', type=str, default=None,
                        help='Ollama model name (default: auto-detect first available model)')
    parser.add_argument('--db-path', type=str, default='./wikipedia_db',
                        help='Path to Wikipedia database')
    parser.add_argument('--question', type=str,
                        help='Single question (otherwise interactive mode)')
    parser.add_argument('--n-results', type=int, default=5,
                        help='Number of Wikipedia chunks to retrieve')
    
    args = parser.parse_args()
    
    # Auto-detect model if not specified
    model_name = args.model
    if not model_name:
        try:
            import ollama
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
    
    # Initialize RAG
    rag = OfflineWikipediaRAG(model_name=model_name, db_path=args.db_path)
    
    if args.question:
        # Single question mode
        result = rag.query_with_rag(args.question, n_results=args.n_results)
        print("\n" + "="*60)
        print(f"Question: {result['question']}")
        print(f"\nAnswer:\n{result['answer']}")
        print("\n" + "-"*60)
        print(f"Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['title']} (relevance: {source['relevance']:.2f})")
            print(f"   {source['content'][:200]}...")
        print("="*60)
    else:
        # Interactive mode
        rag.interactive_mode()


if __name__ == "__main__":
    main()
