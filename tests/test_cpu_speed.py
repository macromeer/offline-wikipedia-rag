#!/usr/bin/env python3
"""Quick CPU token speed test"""

import ollama
import time

model = 'llama3.1:8b'
prompt = "What is photosynthesis? Answer in 2 sentences."

print(f"Testing {model} in CPU mode...")
print("Generating 100 tokens...")

start = time.time()
tokens = 0

stream = ollama.chat(
    model=model,
    messages=[{'role': 'user', 'content': prompt}],
    stream=True,
    options={'num_predict': 100}
)

first_token = None
for chunk in stream:
    if first_token is None:
        first_token = time.time()
        print(f"Time to first token: {first_token - start:.2f}s")
    if 'message' in chunk and 'content' in chunk['message']:
        content = chunk['message']['content']
        tokens += len(content.split())
        print(content, end='', flush=True)

end = time.time()
print(f"\n\nTotal: {end - start:.1f}s, ~{tokens} tokens, ~{tokens/(end-first_token):.1f} tok/s")
