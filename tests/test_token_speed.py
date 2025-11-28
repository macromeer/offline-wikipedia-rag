#!/usr/bin/env python3
"""Test token generation speed for different models"""

import ollama
import time

def test_token_speed(model_name, prompt="Explain photosynthesis in detail.", max_tokens=500):
    """Test token generation speed"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    tokens_generated = 0
    
    try:
        stream = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
            options={'num_predict': max_tokens}
        )
        
        first_token_time = None
        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time()
                print(f"Time to first token: {first_token_time - start_time:.2f}s")
            
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                # Rough token count (approximate)
                tokens_generated += len(content.split())
        
        end_time = time.time()
        total_time = end_time - start_time
        generation_time = end_time - (first_token_time or start_time)
        
        tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
        
        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Generation time: {generation_time:.2f}s")
        print(f"  Approximate tokens: {tokens_generated}")
        print(f"  Tokens/sec: {tokens_per_sec:.1f}")
        
        return {
            'model': model_name,
            'total_time': total_time,
            'tokens': tokens_generated,
            'tokens_per_sec': tokens_per_sec
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    models = ['mistral:7b', 'llama3.1:8b']
    
    results = []
    for model in models:
        result = test_token_speed(model)
        if result:
            results.append(result)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        print(f"{r['model']:20} - {r['tokens_per_sec']:.1f} tokens/sec ({r['total_time']:.1f}s total)")
