#!/usr/bin/env python3
"""
Build a local Wikipedia index for fully offline RAG
Downloads and indexes Wikipedia articles for faster offline access
"""

import wikipedia
import chromadb
from chromadb.utils import embedding_functions
import argparse
from tqdm import tqdm
import json
import os


class WikipediaIndexer:
    """Build and manage a local Wikipedia index"""
    
    def __init__(self, db_path: str = "./wikipedia_db"):
        """
        Initialize the indexer
        
        Args:
            db_path: Path to store the ChromaDB database
        """
        self.db_path = db_path
        
        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Use sentence transformers for embeddings (runs locally)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                "wikipedia",
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.chroma_client.create_collection(
                "wikipedia",
                embedding_function=self.embedding_function
            )
        
        print(f"✓ Initialized Wikipedia indexer at: {db_path}")
    
    def index_articles(self, titles: list, batch_size: int = 10):
        """
        Index Wikipedia articles by title
        
        Args:
            titles: List of Wikipedia article titles to index
            batch_size: Number of articles to process at once
        """
        print(f"\n📥 Indexing {len(titles)} Wikipedia articles...")
        
        successful = 0
        failed = []
        
        for i in tqdm(range(0, len(titles), batch_size)):
            batch_titles = titles[i:i+batch_size]
            
            for title in batch_titles:
                try:
                    # Fetch article
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    # Split content into chunks (for better retrieval)
                    content_chunks = self._split_content(page.content, chunk_size=500)
                    
                    # Add to database
                    for idx, chunk in enumerate(content_chunks):
                        self.collection.add(
                            documents=[chunk],
                            metadatas=[{
                                "title": title,
                                "url": page.url,
                                "chunk_id": idx,
                                "total_chunks": len(content_chunks)
                            }],
                            ids=[f"{title}_chunk_{idx}"]
                        )
                    
                    successful += 1
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    failed.append(f"{title} (disambiguation)")
                except wikipedia.exceptions.PageError:
                    failed.append(f"{title} (not found)")
                except Exception as e:
                    failed.append(f"{title} ({str(e)})")
        
        print(f"\n✓ Successfully indexed: {successful}")
        if failed:
            print(f"⚠ Failed: {len(failed)}")
            print(f"  First few failures: {', '.join(failed[:5])}")
    
    def index_category(self, category: str, max_articles: int = 100):
        """
        Index articles from a Wikipedia category
        
        Args:
            category: Wikipedia category name
            max_articles: Maximum number of articles to index
        """
        print(f"\n🔍 Searching category: {category}")
        
        # Search for articles in this category
        search_results = wikipedia.search(category, results=max_articles)
        
        print(f"Found {len(search_results)} articles")
        self.index_articles(search_results)
    
    def index_from_file(self, filepath: str):
        """
        Index articles from a file containing titles (one per line)
        
        Args:
            filepath: Path to file with article titles
        """
        with open(filepath, 'r') as f:
            titles = [line.strip() for line in f if line.strip()]
        
        self.index_articles(titles)
    
    def search(self, query: str, n_results: int = 5):
        """
        Search the local Wikipedia index
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results with content and metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def get_stats(self):
        """Get statistics about the index"""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'db_path': self.db_path
        }
    
    @staticmethod
    def _split_content(content: str, chunk_size: int = 500, overlap: int = 50):
        """Split content into overlapping chunks"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i+chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks


def main():
    parser = argparse.ArgumentParser(description='Build local Wikipedia index')
    parser.add_argument('--db-path', type=str, default='./wikipedia_db',
                        help='Path to store the database')
    parser.add_argument('--titles', nargs='+',
                        help='List of article titles to index')
    parser.add_argument('--file', type=str,
                        help='File containing article titles (one per line)')
    parser.add_argument('--category', type=str,
                        help='Wikipedia category to index')
    parser.add_argument('--max-articles', type=int, default=100,
                        help='Maximum articles to index from category')
    parser.add_argument('--search', type=str,
                        help='Test search query')
    parser.add_argument('--stats', action='store_true',
                        help='Show index statistics')
    
    args = parser.parse_args()
    
    # Initialize indexer
    indexer = WikipediaIndexer(db_path=args.db_path)
    
    # Index content
    if args.titles:
        indexer.index_articles(args.titles)
    elif args.file:
        indexer.index_from_file(args.file)
    elif args.category:
        indexer.index_category(args.category, max_articles=args.max_articles)
    
    # Test search
    if args.search:
        print(f"\n🔍 Testing search: {args.search}")
        results = indexer.search(args.search)
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ), 1):
            print(f"\n{i}. {meta['title']} (relevance: {1-dist:.3f})")
            print(f"   {doc[:200]}...")
    
    # Show stats
    if args.stats or not any([args.titles, args.file, args.category, args.search]):
        stats = indexer.get_stats()
        print(f"\n📊 Index Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Database path: {stats['db_path']}")


if __name__ == "__main__":
    main()
