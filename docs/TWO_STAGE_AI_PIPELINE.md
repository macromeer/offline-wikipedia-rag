# Two-Stage AI Pipeline for Article Selection and Summarization

> **v1 pipeline.** Since v2 the default (`--retrieval zim`) reads passages straight from the ZIM with libzim and ranks them with BM25; no selection model is used (see the README, "How It Works"). This page describes the older pipeline, still available with `--retrieval kiwix` as a baseline for evaluation.

## Overview

The Wikipedia RAG system now uses a **two-stage specialized AI pipeline** based on research showing that different models excel at different tasks:

### Stage 1: Article Selection (Classification)
- **Purpose**: Accurately identify the most relevant Wikipedia articles
- **Models**: auto-detected (e.g. qwen3.6, Qwen2.5, Mistral-Small, Hermes-3)
- **Why**: Specialized classification models excel at structured output and instruction-following

### Stage 2: Summarization (Synthesis)
- **Purpose**: Synthesize information from selected articles into coherent answers
- **Best Models**: Llama-3.1-8B (practical), Gemma-2-27B/9B (efficient), Mistral-7B (fast)
- **Optional Upgrade**: Llama-3.1-70B for users with high-end hardware (64GB+ RAM)
- **Why**: These models excel at coherent long-form generation and multi-article synthesis

## Why Two Models?

The rationale was:
1. **Reasoning models are a poor fit for selection**: DeepSeek R1 and similar models spend their budget thinking on a simple classification task
2. **Optimal resource usage**: smaller efficient models for selection, larger models for synthesis

Earlier versions of this page quoted selection accuracies (78-92%) and a 15-20% gain from specialised models. Those numbers were never measured and have been removed. For measured numbers see [Measured performance](#measured-performance).

## Recommended Model Combinations

### Recommended Setup (Most Users: 32GB RAM + 16GB VRAM)
```bash
# Selection: Qwen2.5-14B (excellent classification, practical)
ollama pull qwen2.5:14b-instruct

# Summarization: Llama-3.1-8B (great quality, efficient)
ollama pull llama3.1:8b-instruct

# Run
./run.sh \
  --selection-model qwen2.5:14b-instruct \
  --model llama3.1:8b-instruct
```

### High-Performance Setup (64GB RAM + 24GB VRAM)
# Selection: Qwen2.5-32B (best classification accuracy)
ollama pull qwen2.5:32b-instruct

# Summarization: Gemma-2-27B (excellent quality, practical)
ollama pull gemma2:27b

# Run
./run.sh \
  --selection-model qwen2.5:32b-instruct \
  --model gemma2:27b
```

### Power User Setup (Optional: 128GB RAM + 48GB VRAM)
```bash
# Selection: Qwen2.5-32B
ollama pull qwen2.5:32b-instruct

# Summarization: Llama-3.1-70B (maximum quality)
ollama pull llama3.1:70b-instruct

# Run
./run.sh \
  --selection-model qwen2.5:32b-instruct \
  --model llama3.1:70b-instruct
```

### Budget Setup (32GB RAM + 12GB VRAM)
```bash
# Selection: Hermes-3-8B (efficient, reliable)
ollama pull hermes3:8b

# Summarization: Gemma-2-9B (compact, good quality)
ollama pull gemma2:9b

# Run
./run.sh \
  --selection-model hermes3:8b \
  --model gemma2:9b
```

### Single Model Compromise (If two models are impractical)
```bash
# Mistral-Small: Best balance for both tasks (2-3x faster)
ollama pull mistral-small:latest

./run.sh --model mistral-small:latest
```

## Model Performance Comparison

| Model | Summarization Quality | Speed | RAM Required | Best Use Case |
|-------|----------------------|-------|--------------|---------------|
| **Qwen2.5-14B** | Good | Fast | 16GB | **Recommended selection** |
| **Llama-3.1-8B** | **Very Good** | **Fast** | 8GB | **Recommended summarization** |
| **Qwen2.5-32B** | Good | Medium | 32GB | High-performance selection |
| **Gemma-2-27B** | Excellent | Medium | 28GB | High-performance summarization |
| **Mistral-Small** | Good | Very Fast | 24GB | Balanced single-model option |
| **Hermes-3-8B** | Good | Very Fast | 8GB | Budget selection |
| **Llama-3.1-70B** | Excellent | Medium | 64GB | Optional: power users only |

## Usage

### Auto-Detection (Recommended)
The system automatically detects the best available models:
```bash
./run.sh
```

### Manual Configuration
Specify both models explicitly:
```bash
./run.sh \
  --selection-model qwen2.5:32b-instruct \
  --model llama3.1:70b-instruct \
  --question "What causes earthquakes?"
```

### Check Available Models
```bash
ollama list
```

## How It Works

### Stage 1: Classification (Article Selection)
```
User Question → Selection Model (Qwen2.5-32B) → Relevant Articles
```

The selection model:
- Receives article titles from search results
- Classifies each as RELEVANT or IRRELEVANT
- Uses low temperature (0.3) for consistent classification
- Returns article numbers in structured format

**Prompt Strategy**: Classification-focused with clear rules
```
"You are a classification expert. Classify these Wikipedia articles..."
- RELEVANT: Contains factual information directly answering the question
- IRRELEVANT: Lists, year-specific, fiction, sports, entertainment
```

### Stage 2: Synthesis (Summarization)
```
Selected Articles → Summarization Model (Llama-3.1-70B) → Coherent Answer
```

The summarization model:
- Receives full text of selected articles
- Synthesizes information across all articles
- Uses moderate temperature (0.7) for coherent generation
- Produces comprehensive, well-structured answers

**Prompt Strategy**: Synthesis-focused with integration emphasis
```
"You are an expert research analyst synthesizing information..."
- Comprehensiveness: Integrate information from ALL articles
- Coherence: Create logical narrative connecting concepts
- Evidence: Include specific facts, dates, numbers
```

## Models to Avoid

Based on research and testing:

❌ **DeepSeek-R1**: Reasoning models fail at simple classification tasks
❌ **Mixtral 8x7B**: Complex MoE architecture provides no advantage
❌ **Phi-3-mini**: Insufficient capacity for nuanced relevance determination
❌ **Models <8B parameters**: Consistently fail at classification accuracy

## Memory Optimization

### Quantization Options
Reduce memory by ~60% with minimal quality loss:
```bash
# Q4_K_M quantization (recommended balance)
ollama pull qwen2.5:32b-instruct-q4_K_M
ollama pull llama3.1:70b-instruct-q4_K_M

# Q4_0 quantization (maximum compression)
ollama pull qwen2.5:32b-instruct-q4_0
ollama pull llama3.1:70b-instruct-q4_0
```

### Ollama Configuration
```bash
# Enable parallel model loading
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2

# Enable NUMA for better performance
ollama serve --numa
```

## Measured performance

Measured with the eval harness ([EVALUATION.md](EVALUATION.md)) on 226 questions (hand-written, HotpotQA, PopQA), full English nopic ZIM, selection `qwen3.6:35b`, synthesis `gemma4:26b`, 2x RTX PRO 4000:

| | v1 (this pipeline) | v2 one-shot, hybrid | v2 chat, hybrid |
| --- | --- | --- | --- |
| gold article in context | 0.59 | 0.77 | 0.76 |
| answer contains the gold answer | 0.55 | 0.71 | 0.68 |
| HotpotQA multi-hop answers | 0.25 | 0.40 | 0.43 |
| seconds per question, p50 | 5.4 | 2.9 | 2.9 |

The time and memory figures below are the original estimates for the listed models, not measurements.

## Expected Performance

### Recommended Setup (Qwen2.5-14B + Llama3.1-8B)
- **Synthesis Quality**: Very Good
- **Total Time**: 10-18 seconds (depending on article count)
- **Memory Usage**: 16-24GB RAM

### High-Performance Setup (Qwen2.5-32B + Gemma2-27B)
- **Synthesis Quality**: Excellent
- **Total Time**: 15-22 seconds
- **Memory Usage**: 48-60GB RAM

### Single Model (Mistral-Small)
- **Synthesis Quality**: Good
- **Total Time**: 10-15 seconds
- **Memory Usage**: 24GB RAM

## Troubleshooting

### Selection Model Not Found
The system automatically falls back to available models. To check:
```bash
ollama list | grep -E 'qwen|mistral|hermes'
```

### Out of Memory
1. Use quantized models (Q4_K_M)
2. Reduce article count: `--max-results 3`
3. Use smaller models (Hermes-3-8B + Gemma-2-9B)

### Poor Selection Quality
1. Ensure using non-reasoning model (not DeepSeek R1)
2. Try Qwen2.5-32B if available
3. Check model is properly loaded: `ollama ps`

### Slow Performance
1. Use quantized versions (q4_K_M)
2. Enable NUMA: `ollama serve --numa`
3. Consider Mistral-Small for faster inference

## Example Usage

### Interactive Mode
```bash
./run.sh
```

### Single Question
```bash
./run.sh \
  --question "What is the relationship between plate tectonics and earthquakes?" \
  --max-results 5
```

### Custom Models
```bash
./run.sh \
  --selection-model qwen2.5:14b-instruct \
  --model gemma2:27b \
  --question "What causes volcanoes?"
```

## References

The model choices above came from an unsourced analysis (November 2025); its accuracy claims are not reproduced here. Measured results: [EVALUATION.md](EVALUATION.md).
