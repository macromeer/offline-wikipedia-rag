# Evaluation

How retrieval and answers are measured, and the latest numbers. The harness lives in `eval/`.

## Running it

```bash
# retrieval only (fast: ~0.1-0.3 s per question per system)
uv run python -m eval.run retrieval --systems fulltext,hybrid

# end to end with the answer model (several seconds per question)
uv run python -m eval.run answer --systems chat-hybrid,oneshot-hybrid

# the v1 baseline (needs kiwix-serve and the selection model; slow)
uv run python -m eval.run retrieval --systems v1
uv run python -m eval.run answer --systems v1

# regenerate the report of a results directory
uv run python -m eval.run report eval/results/<dir>
```

Results go to `eval/results/<date-time>/` (git-ignored): one `<stage>-<system>.jsonl` per system with a row per question (retrieved articles, answer, scores, timings) and `report.md`. Pass `--out DIR` to resume an interrupted run or add systems to an existing one; questions already in a file are skipped. `--source handwritten,hotpotqa,popqa` and `--limit N` select questions.

| system | what it runs |
| --- | --- |
| `fulltext` | `ZimStore.retrieve`, Xapian full-text + title hits, no dense index |
| `hybrid` | the same with the dense index attached (what the app uses when the index exists) |
| `v1` | the v1 pipeline: kiwix-serve search, LLM picks articles from abstracts, first paragraphs of each |
| `chat-fulltext`, `chat-hybrid` | `WikiChat` tool loop, a fresh session per question (no history) |
| `oneshot-fulltext`, `oneshot-hybrid` | retrieve once, answer in one call (`--no-tools`) |

Before timing, the harness fills the on-disk term statistics cache for every question and loads the models, so latency is the steady state, not a cold start.

## Question set

`eval/questions.jsonl`, 226 questions, built by `uv run --group eval python -m eval.prepare` (seeded; the sources are downloaded once to `~/.cache/offline-wikipedia-rag/eval-sources`):

| subset | n | gold | what it tests |
| --- | --- | --- | --- |
| `handwritten` | 26 | per question | v1 failure modes and cases found while building v2: media titles ("The Expanse"), lowercase proper nouns, abbreviations ("ETF", "NASA mission and goals"), comparisons, "first person to walk on the moon", "how do earthquakes cause tsunamis". Source: `eval/handwritten.jsonl` |
| `hotpotqa-bridge` | 60 | both supporting articles | multi-hop: the second article is only reachable through the first |
| `hotpotqa-comparison` | 40 | both compared articles | two named entities, compared |
| `popqa` | 100 | the subject's article | single-fact questions about long-tail entities, stratified over the 16 relations and three popularity bands (`band`: rare/mid/popular) |

HotpotQA (CC BY-SA 4.0) questions come from the distractor validation set; PopQA (MIT) from its test set. A sampled question is kept only if its gold articles exist in the ZIM (redirects followed) and are not disambiguation pages: 6 HotpotQA and 2 PopQA samples were dropped for the 2026-06 dump. HotpotQA questions containing "current", "now" and similar words are skipped because their 2017 answers may have changed.

Gold titles are stored as titles and resolved against the ZIM at run time, so the set works with later dumps as long as renamed articles keep redirects.

## Metrics

Retrieval (per question, then averaged):

- **R@k**: gold articles among the first k passages. For `gold_mode: all` (HotpotQA, comparisons) the fraction of golds found; for `any` (alternatives, e.g. *The Expanse* TV series or novels) 1 if any is found.
- **R@ctx**: the same over everything returned, i.e. what the answer model sees. v1 returns 1-7 whole articles (up to 8,000 characters each), the ZIM systems ~10 section passages, so compare v1 on R@ctx, not R@k.
- **all@ctx**: every gold article returned (the multi-hop "complete" rate).
- **ans@ctx**: a gold answer string occurs in the returned text: an upper bound on what the model can answer from the passages. Skipped for yes/no answers.

Answers:

- **answer**: a gold alias occurs in the answer as whole tokens after SQuAD normalisation (case, accents, punctuation, articles, `[n]` markers, thousands separators). This is PopQA's accuracy criterion; exact match is meaningless for multi-sentence answers. Yes/no golds are judged on the first sentence, which fails if it contains both words. Questions without a short answer (`answers: []`) are not scored here.
- **F1**: token F1 of the first sentence against the best gold alias. Low by construction (the first sentence restates the question), useful only for comparing systems.
- **cites ok**: at least one `[n]` and every `[n]` refers to a passage that was shown.
- **cites gold**: a cited passage comes from a gold article.
- **shown R**: gold recall over every passage the model saw (all tool calls for the chat systems).

Known limits: string match misses paraphrases ("the U.S." vs "United States" when no alias covers it) and can be fooled by an answer listing several candidates. PopQA has questions whose subject name is ambiguous ("Who is the author of Images?"); the gold is the article PopQA meant. Spot-check `answer` with the jsonl files before trusting small differences.

## Results (2026-10-02)

Full ZIM `wikipedia_en_all_nopic_2026-06.zim`; dense index over article introductions (`--scope leads`, 6.8M passages); answer model `gemma4:26b`; v1 selection model `qwen3.6:35b`; `max_rounds` 2; 2x RTX PRO 4000 Blackwell. p-values are exact McNemar tests on the questions where the two systems disagree.

### Retrieval

| system | R@1 | R@5 | R@10 | R@ctx | all@ctx | ans@ctx | p50 / p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | 0.50 | 0.59 | 0.59 | 0.59 | 0.48 | 0.40 | 2.45 / 4.38 |
| fulltext | 0.46 | 0.59 | 0.64 | 0.64 | 0.57 | 0.66 | 0.09 / 0.34 |
| hybrid | 0.50 | 0.72 | 0.78 | **0.79** | **0.70** | **0.78** | 0.21 / 0.40 |

| R@ctx by subset | handwritten | hotpotqa-bridge | hotpotqa-comparison | popqa |
| --- | --- | --- | --- | --- |
| v1 | 0.73 | 0.33 (all: 0.03) | 0.82 | 0.61 |
| fulltext | 0.73 | 0.40 (all: 0.17) | 0.89 | 0.67 |
| hybrid | 0.88 | 0.50 (all: 0.22) | 0.94 | 0.87 |

- Hybrid vs fulltext: better on 46 questions, worse on 7 (p = 4e-8). The dense index meets Phase 3's acceptance criterion (recall@10 on the multi-hop subset: bridge 0.40 → 0.50, comparison 0.89 → 0.94), and helps most on long-tail PopQA entities (0.67 → 0.87).
- v1 finds the right article about as often as fulltext at rank 1, but returns 2-3 articles and reads only their first paragraphs, so the answer is in its context for 40% of questions (hybrid 78%). It almost never gets both articles of a bridge question (3%).

### Answers

| system | answer | cites ok | cites gold | shown R | p50 / p95 s |
| --- | --- | --- | --- | --- | --- |
| v1 | 0.55 | 1.00 | 0.69 | 0.59 | 5.44 / 7.73 |
| chat-fulltext | 0.61 | 0.93 | 0.73 | 0.67 | 2.88 / 5.44 |
| chat-hybrid | 0.68 | 0.95 | 0.80 | 0.76 | 2.92 / 5.63 |
| oneshot-hybrid | **0.71** | **1.00** | **0.85** | 0.77 | 2.89 / 4.46 |

| answer by subset | handwritten | hotpotqa-bridge | hotpotqa-comparison | popqa |
| --- | --- | --- | --- | --- |
| v1 | 0.79 | 0.25 | 0.78 | 0.59 |
| chat-fulltext | 0.95 | 0.35 | 0.72 | 0.66 |
| chat-hybrid | 0.95 | 0.43 | 0.75 | 0.75 |
| oneshot-hybrid | 1.00 | 0.40 | 0.80 | 0.81 |

- Every v2 system beats v1: chat-hybrid +41/-12 questions (p = 8e-5), oneshot-hybrid +43/-7 (p = 2e-7), at half v1's latency.
- The dense index helps answers too: chat-hybrid vs chat-fulltext +22/-7 (p = 0.008).
- **The tool loop does not beat one-shot retrieval.** oneshot-hybrid vs chat-hybrid: +20/-13 (p = 0.30, not significant), with fewer citation errors. In the chat runs, 163 of 226 turns made a single search; on bridge questions the model typically searches only the first entity ("Julien Kang", "Robinsons-May") and answers without the second hop, while the raw question also names the second. 6 chat-hybrid answers (3 chat-fulltext) were empty ("No answer was generated") despite the empty-reply retry; one-shot had none.
- Latency is the same for chat and one-shot (~2.9 s p50; first chat text after 2.2 s p50), so the loop's extra round costs little when it is not used.
- Answer F1 (0.14-0.17 for all systems) mostly measures how much the first sentence restates the question; it is not useful for ranking these systems.

### Open problems found by the eval

- **Empty answers in the chat loop** (above): the no-tools retry also comes back empty.
- **Portal pages in the dense index**: `Portal:` pages make up 1.7% of hybrid passages (in 26 of 226 questions); the dense builder should skip non-article namespaces, or retrieval should drop them.
- "NASA mission and goals" still retrieves *List of NASA missions* and no *NASA* passage (all systems); "first person to walk on the moon" never retrieves *Neil Armstrong*, *Apollo 11* or *Moon landing* (the answer comes from other articles' passages).
- String matching undercounts: e.g. "the 2016 *Ghostbusters* reboot" is right but does not contain the gold alias "Ghostbusters: Answer the Call". Differences of a few questions between systems are within this noise.
