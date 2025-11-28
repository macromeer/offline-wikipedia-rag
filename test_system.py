#!/usr/bin/env python3
"""
Test script for the Wikipedia RAG system
"""

import ollama
import sys


def test_ollama_connection():
    """Test if Ollama is accessible"""
    print("🧪 Testing Ollama connection...")
    try:
        response = ollama.list()
        models = response.models if hasattr(response, 'models') else []
        print(f"✓ Ollama is running")
        if models:
            print(f"✓ Available models: {', '.join([m.model for m in models])}")
        else:
            print(f"⚠ No models installed. Install a model with: ollama pull llama2")
        return True, models
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print(f"💡 Make sure Ollama is running: ollama serve")
        return False, []


def test_wikipedia_api():
    """Test Wikipedia API access"""
    print("\n🧪 Testing Wikipedia API...")
    try:
        import wikipedia
        result = wikipedia.search("Python programming", results=1)
        print(f"✓ Wikipedia API working")
        print(f"✓ Test search found: {result}")
        return True
    except Exception as e:
        print(f"❌ Wikipedia API failed: {e}")
        return False


def test_basic_rag():
    """Test basic RAG functionality"""
    print("\n🧪 Testing basic RAG system...")
    try:
        from wikipedia_rag import WikipediaRAG
        
        # Get available models
        _, models = test_ollama_connection()
        if not models:
            print("❌ No Ollama models available")
            return False
        
        model_name = models[0].model
        print(f"Using model: {model_name}")
        
        rag = WikipediaRAG(model_name=model_name)
        
        # Test query
        test_question = "What is Python programming language?"
        print(f"\n📝 Test question: {test_question}")
        
        result = rag.query_with_rag(test_question, max_wiki_results=2)
        
        print(f"\n✓ RAG system working!")
        print(f"✓ Answer length: {len(result['answer'])} characters")
        print(f"✓ Sources retrieved: {len(result['sources'])}")
        
        print(f"\n📖 Answer preview:")
        print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"💡 Install dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False


def test_offline_rag():
    """Test offline RAG with local database"""
    print("\n🧪 Testing offline RAG system...")
    try:
        from wikipedia_rag_offline import OfflineWikipediaRAG
        
        # Get available models
        _, models = test_ollama_connection()
        if not models:
            print("❌ No Ollama models available")
            return False
        
        model_name = models[0].model
        
        try:
            rag = OfflineWikipediaRAG(model_name=model_name)
            print(f"✓ Offline RAG system initialized")
            return True
        except Exception as e:
            print(f"⚠ Offline RAG not available: {e}")
            print(f"💡 This is normal if you haven't built the index yet")
            print(f"💡 Run: python wikipedia_indexer.py --titles 'Python' 'Machine learning' 'Artificial intelligence'")
            return None  # Not a failure, just not set up yet
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Offline RAG test failed: {e}")
        return False


def main():
    print("="*60)
    print("Wikipedia RAG System - Test Suite")
    print("="*60 + "\n")
    
    # Run tests
    results = {
        'ollama': test_ollama_connection()[0],
        'wikipedia': test_wikipedia_api(),
        'basic_rag': test_basic_rag(),
        'offline_rag': test_offline_rag()
    }
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"{status:10} {test_name}")
    
    print("="*60)
    
    # Check if ready to use
    if results['ollama'] and results['wikipedia'] and results['basic_rag']:
        print("\n🎉 System is ready to use!")
        print("💡 Run: python wikipedia_rag.py")
    else:
        print("\n⚠ System needs setup. Check failed tests above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
