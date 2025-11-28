# 📚 Usage Examples

## Interactive Mode

The easiest way to use the system:

```bash
$ python wikipedia_rag_kiwix.py

======================================================================
 🌐 Offline Wikipedia AI Assistant
======================================================================
 🤖 Model: deepseek-r1:latest
 📚 Wikipedia: Local (http://localhost:8080)
 💡 Tip: Ask any question, type 'quit' to exit
======================================================================

❓ Your question: What is photosynthesis?

🔍 Searching local Wikipedia...
✓ Found 3 article(s)
  📄 Fetching: Photosynthesis
  📄 Fetching: Light-dependent reactions
  📄 Fetching: Calvin cycle
🤖 Generating answer...

======================================================================
📖 Answer:

   Photosynthesis is the biological process by which plants, algae, and 
   certain bacteria convert light energy (usually from the sun) into 
   chemical energy stored in glucose molecules...

======================================================================
📚 Sources: Photosynthesis, Light-dependent reactions, Calvin cycle
======================================================================
```

## Single Question Mode

Quick one-off questions:

```bash
$ python wikipedia_rag_kiwix.py --question "Who invented the telephone?"

======================================================================
❓ Question: Who invented the telephone?
======================================================================

📖 Answer:

   Alexander Graham Bell is credited with inventing the telephone in 1876.
   He was a Scottish-born scientist and inventor who developed the first
   practical telephone device...

======================================================================
📚 Sources: Alexander Graham Bell, Invention of the telephone
======================================================================
```

## Use Specific Model

```bash
# Use a specific Ollama model
python wikipedia_rag_kiwix.py --model llama2 --question "Explain relativity"

# Use smaller model for faster responses
python wikipedia_rag_kiwix.py --model deepseek-r1:7b
```

## Adjust Number of Sources

```bash
# Use only 1 Wikipedia article (faster, less context)
python wikipedia_rag_kiwix.py --max-results 1 --question "What is DNA?"

# Use 5 articles (slower, more comprehensive)
python wikipedia_rag_kiwix.py --max-results 5 --question "Explain quantum physics"
```

## Example Questions to Try

### Science & Technology
```bash
python wikipedia_rag_kiwix.py --question "How does a computer processor work?"
python wikipedia_rag_kiwix.py --question "What is CRISPR gene editing?"
python wikipedia_rag_kiwix.py --question "Explain black holes"
```

### History
```bash
python wikipedia_rag_kiwix.py --question "What caused the French Revolution?"
python wikipedia_rag_kiwix.py --question "Who was Cleopatra?"
python wikipedia_rag_kiwix.py --question "Explain the Industrial Revolution"
```

### Arts & Culture
```bash
python wikipedia_rag_kiwix.py --question "Who painted the Mona Lisa?"
python wikipedia_rag_kiwix.py --question "What is Renaissance art?"
python wikipedia_rag_kiwix.py --question "Explain jazz music"
```

### Philosophy & Ideas
```bash
python wikipedia_rag_kiwix.py --question "What is existentialism?"
python wikipedia_rag_kiwix.py --question "Explain Plato's theory of forms"
python wikipedia_rag_kiwix.py --question "What is the scientific method?"
```

### Current Events & Geography
```bash
python wikipedia_rag_kiwix.py --question "Where is Mount Everest?"
python wikipedia_rag_kiwix.py --question "What is climate change?"
python wikipedia_rag_kiwix.py --question "Explain the European Union"
```

## Advanced Usage

### Custom Kiwix Server

If running Kiwix on different port or machine:

```bash
python wikipedia_rag_kiwix.py \
  --kiwix-url http://192.168.1.100:8090 \
  --question "Your question"
```

### Scripting

Use in shell scripts:

```bash
#!/bin/bash
# Ask multiple questions

questions=(
  "What is AI?"
  "What is machine learning?"
  "What is deep learning?"
)

for q in "${questions[@]}"; do
  echo "Asking: $q"
  python wikipedia_rag_kiwix.py --question "$q" > "answer_${q//[^a-zA-Z]/_}.txt"
done
```

### Python Integration

Use as a module:

```python
from wikipedia_rag_kiwix import KiwixWikipediaRAG

# Initialize
rag = KiwixWikipediaRAG(model_name='deepseek-r1:latest')

# Ask question
result = rag.query_with_rag("What is Python programming?")

print(result['answer'])
print("Sources:", [s['title'] for s in result['sources']])
```

## Tips for Best Results

1. **Be specific**: "Python programming language" > "Python"
2. **Use complete sentences**: "What is..." or "Explain..." 
3. **One topic at a time**: Better answers with focused questions
4. **Technical terms**: Use proper names and technical terminology
5. **Follow-up**: Ask clarifying questions based on previous answers

## Performance Tips

### Faster responses:
```bash
# Reduce sources
--max-results 1

# Use smaller model
--model deepseek-r1:7b
```

### Better quality:
```bash
# More sources
--max-results 5

# Use full model
--model deepseek-r1:latest
```

## Troubleshooting

If you get poor results:

1. **Check sources** - Are they relevant?
2. **Rephrase question** - Try different wording
3. **Adjust sources** - Use `--max-results`
4. **Verify Kiwix** - Open http://localhost:8080 in browser

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more help.
