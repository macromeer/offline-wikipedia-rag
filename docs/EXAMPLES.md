# 📚 Usage Examples

## 🧠 Adaptive Intelligence

The system automatically adjusts how many Wikipedia articles to retrieve based on question complexity:

- **Simple questions** (3 articles): "What is Python?", "Who is Einstein?"
- **Moderate questions** (4-5 articles): "How does photosynthesis work?", "Explain quantum mechanics"
- **Complex questions** (6 articles): "Compare socialism and capitalism", "How old is the universe and what is its future?"
- **Very complex questions** (7 articles): "Explain the history and evolution of democracy", "What caused the fall of the Roman Empire?"

**Smart detection looks for:**
- Multi-part questions ("and", "vs")
- Comparisons ("compare", "difference", "versus")
- Relationships ("impact", "affect", "cause")
- Deep explanations ("how does", "why", "explain")
- Historical context ("history", "evolution", "origin")
- Future predictions ("future", "will", "prediction")

This ensures:
- ✅ Fast responses for simple questions (3 articles)
- ✅ Comprehensive context for complex topics (up to 7 articles)
- ✅ Better synthesis across multiple related articles
- ✅ More thorough content per article (5000 chars vs 3000)
- ✅ Automatic - no configuration needed

## Interactive Mode

The easiest way to use the system:

```bash
$ ./run.sh

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
$ ./run.sh --question "Who invented the telephone?"

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
./run.sh --model llama2 --question "Explain relativity"

# Use smaller model for faster responses
./run.sh --model deepseek-r1:7b
```

## Adjust Number of Sources

```bash
# Use only 1 Wikipedia article (faster, less context)
./run.sh --max-results 1 --question "What is DNA?"

# Use 5 articles (slower, more comprehensive)
./run.sh --max-results 5 --question "Explain quantum physics"
```

## Example Questions to Try

### Science & Technology
```bash
./run.sh --question "How does a computer processor work?"
./run.sh --question "What is CRISPR gene editing?"
./run.sh --question "Explain black holes"
```

### History
```bash
./run.sh --question "What caused the French Revolution?"
./run.sh --question "Who was Cleopatra?"
./run.sh --question "Explain the Industrial Revolution"
```

### Arts & Culture
```bash
./run.sh --question "Who painted the Mona Lisa?"
./run.sh --question "What is Renaissance art?"
./run.sh --question "Explain jazz music"
```

### Philosophy & Ideas
```bash
./run.sh --question "What is existentialism?"
./run.sh --question "Explain Plato's theory of forms"
./run.sh --question "What is the scientific method?"
```

### Current Events & Geography
```bash
./run.sh --question "Where is Mount Everest?"
./run.sh --question "What is climate change?"
./run.sh --question "Explain the European Union"
```

## Advanced Usage

### Custom Kiwix Server

If running Kiwix on different port or machine:

```bash
./run.sh \
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
  ./run.sh --question "$q" > "answer_${q//[^a-zA-Z]/_}.txt"
done
```

### Python Integration

Use as a module:

```python
from retrieval import ZimStore
from wikirag import WikiChat, resolve_zim_path

chat = WikiChat(ZimStore(resolve_zim_path()), model='gemma4:26b')

result = chat.ask("What is Python programming?")
print(result.answer)
print("Sources:", [c.label for c in result.citations])

# follow-ups use the conversation
print(chat.ask("Who created it?").answer)
```

`WikiChat(..., on_event=callback)` receives the tool calls and answer text as they happen (see `wikirag/cli.py` for a printer).

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
