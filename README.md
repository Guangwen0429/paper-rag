# Paper-RAG: Citation-Grounded Academic QA

End-to-end RAG over a curated corpus of 8 NLP papers (Transformer, BERT,
GPT-3, InstructGPT, RAG, DPR, Chain-of-Thought, LLaMA 2). Built from
scratch as a methodology-focused study of which retrieval components
actually move the needle in chunked academic-paper RAG, evaluated under
an empirically-measured noise floor.

## Status

Day 13 / 21 — core experiments complete + Day 13 audit done.

## Headline Result

After bug-fix in the evaluator, **only one method on the 30-question
evaluation set exceeds the empirically-measured noise floor**: single-passage
HyDE rerank.

| Config (post bug-fix)       | Keyword Hit | 3-run std    |
|-----------------------------|-------------|--------------|
| hybrid k=5                  | 18/30       | not measured |
| **rerank k=5**              | **20/30**   | **0**        |
| rerank_weighted α=0.7       | 17/30       | not measured |
| **hyde_rerank**             | **21/30**   | **0**        |
| hyde_ensemble (N=5)         | 21/30       | 0            |

The +1 question gain (21 vs 20) is reproducible across 3 runs with std=0,
exceeding the project-level noise floor of ±1 question on a 30q set
(established in Day 10 reproducibility audit).

## Methodology Highlights

- **Noise-floor-aware comparison**: every method runs at least 3 times. A
  single-run difference of 1 question is not treated as signal until std=0
  across runs is verified. Day 10 audit (`rerank_weighted α=0.0` running
  16 / 17 / 16) exposed that prior 15q "improvements" were within noise.
- **Evaluator self-audit**: Day 12 chunk-level inspection revealed an
  AND/OR bug in `check_keyword_hit` that had silently corrupted 11 days of
  results. Q23/Q24 with synonym keyword lists (e.g. `["dot product",
  "inner product"]`) had been judged ✗ even when the answer correctly
  contained one synonym. Fixed by per-question `keyword_match_mode` field.
- **Chunk-level failure attribution**: every method's win/loss diff vs
  baseline is investigated at chunk level. 13 distinct failure modes
  documented in `experiment_log.md`.
- **Mechanism verification**: HyDE's actual mechanism on this corpus
  decomposes into chunk-recall (HyDE pulls answer-bearing chunks into
  top-20) and chunk-selection (cross-encoder rerank within top-20). Q08
  succeeds at both. Q17 succeeds at recall but fails at selection because
  the cross-encoder treats the "d=768" tail-clause as background, not
  answer.

## Project Structure

    paper-rag/
    ├── papers/                    # 8 source PDFs
    ├── src/
    │   ├── loader.py              # PDF loading + chunking
    │   ├── retriever.py           # 7 retrieval modes
    │   ├── generator.py           # answer generation with citations
    │   └── pipeline.py            # end-to-end RAGPipeline class
    ├── scripts/
    │   ├── run_eval.py            # batch evaluation
    │   ├── run_demo.py            # interactive demo
    │   ├── inspect_question.py    # per-question chunk-level inspector
    │   ├── inspect_chunks.py      # chunk-text grepper
    │   ├── analyze_results.py     # produce comparison tables
    │   ├── benchmark_gpu.py       # CPU vs GPU rerank benchmark
    │   └── verify_gpu.py          # CUDA sanity check
    ├── evaluation/
    │   ├── eval_questions.json    # 30-question test set
    │   ├── eval_results_*.json    # per-config detailed results
    │   └── eval_summary_*.json    # per-day rolled-up summaries
    ├── experiment_log.md          # day-by-day log + 32 lessons
    └── error_analysis_report.md   # cross-cutting failure-mode taxonomy

## Pipeline

    PDF
      -> page-level text
      -> chunking (size=500, overlap=50)
      -> vector embedding (text-embedding-3-small) -> Chroma
      -> BM25 index (rank-bm25)
      -> 7 retrieval modes (vector / BM25 / hybrid / rerank /
                            rerank_weighted / hyde_rerank / hyde_ensemble)
      -> cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
      -> answer generation (gpt-4o-mini) with [Source N] citations

## How to Run

### Setup

    git clone https://github.com/Guangwen0429/paper-rag.git
    cd paper-rag
    python -m venv .venv
    source .venv/bin/activate          # Linux/Mac
    # .venv\Scripts\activate            # Windows PowerShell
    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."

### Demo (interactive)

    python scripts/run_demo.py

### Full evaluation suite

    python scripts/run_eval.py
    # Edit EXPERIMENTS list in run_eval.py to choose which configs to run.

### Per-question chunk inspection

    python scripts/inspect_question.py Q08 Q17

### Debug HyDE candidate pool (Day 13 added)

    DEBUG_TOP20=1 python scripts/run_eval.py
    # Prints the 20-chunk pre-rerank pool for every hyde_rerank query.

## Limitations and Future Work

- **Noise floor is a (config, eval-set, evaluator) triple property**:
  cannot be reused across changes to any of the three. Re-audit required
  after any change.
- **Compound queries unsolved**: Q15, Q28, Q30 fail under all 7 retrieval
  modes. Mechanism: query embedding is a mixture of two independent
  topics; retrieval is dominated by bridging chunks (abstracts,
  comparisons) that contain neither answer. Required intervention: query
  decomposition (not implemented).
- **Cross-encoder selection bottleneck**: Q17 case shows that even when
  HyDE delivers the answer chunk to top-20, cross-encoder may demote it.
  Candidate fixes: weighted fusion of RRF and cross-encoder scores at the
  HyDE pipeline (currently only at the hybrid pipeline as
  `rerank_weighted`); LLM-based reranker.
- **Embedding model and chunking strategy**: only OpenAI
  `text-embedding-3-small` with fixed-length chunking tested. BGE-style
  embeddings and sentence-boundary semantic chunking are unexplored.
- **Eval set size**: 30 questions is small. Mechanism claims
  (e.g. "HyDE helps vocabulary-mismatch class") rest on N=1 evidence
  (Q08 alone).

## Reading Order

To retrace the project's reasoning:

1. `experiment_log.md` Day 1-9 — methodology baseline buildup.
2. `experiment_log.md` Day 10 — noise floor discovery (the project's
   methodological turning point).
3. `experiment_log.md` Day 11-12 — HyDE evaluation + evaluator bug fix.
4. `experiment_log.md` Day 13 — audit corrections (current state).
5. `error_analysis_report.md` — cross-cutting failure-mode taxonomy.

## Industry Mapping

This project is a small-scale academic exploration, but each component maps to a real industrial RAG / search-ranking concern:

| Project Component | Industrial Counterpart |
|---|---|
| Hybrid retrieval (BM25 + vector + RRF) | Multi-channel recall: text + dense retrieval fusion |
| Cross-encoder reranking | Fine-ranking layer: DNN ranker / cross-encoder |
| HyDE query rewriting | Query understanding / rewriting module |
| Reproducibility audit + noise floor (Lesson 21) | A/B test variance analysis before claiming improvement |
| Topic-match over answer-match (Lesson 26) | Reranker domain shift in vertical search (medical, legal, finance) |
| Negative result on score fusion (α-sweep, Lesson 22) | Avoiding redundant signal stacking across recall and ranking layers |
| Stratified evaluation (single_fact / multi_chunk / cross_paper) | Per-segment metric reporting to avoid average-masking weaknesses |

The methodology lessons (noise floor, stratified evaluation, evaluator unit-testing — Lessons 21, 23, 32) generalize to any LLM-based ranking system facing distribution shift and metric-noise issues.

## Roadmap

- **Exp O — LLM-as-Judge replacing keyword_hit**: replace character-level keyword matching with LLM judge to handle synonyms and semantic equivalence; expected to remove the false-negative class identified in Lesson 31.
- **Exp P — LLM intent classifier + conditional routing**: classify queries into single_fact / multi_chunk / cross_paper, dispatch to retrieval configs tuned per category (Lesson 16's query-conditional retrieval, finally implementable now that the failure-mode taxonomy is stable).
- **Exp Q — LLM listwise reranker**: replace pointwise cross-encoder with LLM-driven listwise rerank (RankGPT-style) for cross-paper queries that need set-level coverage rather than per-chunk relevance.
- **Exp R — Domain LoRA on cross-encoder**: hard-negative mining from hybrid recall + GPT-4-synthesized queries (~3000 triplets) → LoRA fine-tuning of `ms-marco-MiniLM-L-6-v2` to address topic-match over answer-match (Lesson 26).

## Author

Guangwen Xiong