# Paper-RAG: Citation-Grounded Academic QA

A RAG (Retrieval-Augmented Generation) system for academic paper question answering, with hybrid retrieval, reranking, and citation-grounded answer generation.

## Current Status

🚧 Work in progress — Day 9 of 21

## Features (so far)

- [x] PDF loading and chunking
- [x] OpenAI embedding + Chroma vector store
- [x] Semantic retrieval
- [x] Citation-grounded answer generation
- [x] Multi-paper support (8 NLP research papers)
- [x] Cross-paper routing via semantic similarity
- [x] Modular architecture (loader / retriever / generator / pipeline)
- [x] Evaluation set (15 questions: single-fact, multi-chunk, cross-paper categories) with automatic metrics
- [x] Controlled 4-config experiments (vector/hybrid × k=3/5) with chunk-level error analysis
- [x] Chunking ablation (chunk_size 250 vs 500) with context-fragmentation analysis
- [x] Cross-encoder reranker with answer-match vs topic-match failure analysis
- [x] Weighted fusion of hybrid + cross-encoder scores (α-sweep revealing per-query-optimal α)
- [x] Hybrid retrieval (BM25 + dense)
- [x] Full evaluation set (15 questions, target: 30)
- [ ] Streamlit UI


## Tech Stack

- Python 3.12
- LangChain
- OpenAI (gpt-4o-mini, text-embedding-3-small)
- Chroma vector database

## Project Structure

    paper-rag/
    ├── src/                       # Core modules
    │   ├── loader.py              # PDF loading and chunking
    │   ├── retriever.py           # Embedding and vector retrieval
    │   ├── generator.py           # Prompt construction and LLM generation
    │   └── pipeline.py            # End-to-end RAG pipeline
    ├── scripts/
    │   ├── run_demo.py              # Demo script
    │   ├── run_eval.py              # Multi-config evaluation (vector/hybrid × k=3/5)
    │   ├── analyze_results.py       # Automated failure classification
    │   ├── inspect_question.py      # Per-question chunk inspection tool
    │   └── inspect_chunks.py        # Full chunk index search tool
    ├── evaluation/
    │   ├── eval_questions.json                                  # 15-question curated QA set
    │   ├── eval_results_{mode}_k{k}_15q.json                    # Day 6 results (cs=500, no cs suffix)
    │   ├── eval_results_{mode}_k{k}_cs{size}_15q.json           # Day 7/8 results
    │   ├── eval_results_rerank_weighted_k5_cs500_a{alpha}_15q.json  # Day 9 α-sweep results
    │   ├── eval_summary_all.json                                # Day 6 summary
    │   ├── eval_summary_all_day7.json                           # Day 7 summary
    │   ├── eval_summary_all_day8.json                           # Day 8 summary
    │   ├── eval_summary_all_day9.json                           # Day 9 summary
    │   ├── error_analysis_report.md                             # Chunk-level analysis report
    │   └── experiment_log.md                                    # Experiment notes (Days 1-9)                   # Experiment notes (Days 1-7)
    ├── requirements.txt
    └── README.md

## Setup

1. Install dependencies:

        pip install -r requirements.txt

2. Set your OpenAI API key as an environment variable:

        Windows: setx OPENAI_API_KEY "sk-your-key-here"
        Mac/Linux: export OPENAI_API_KEY="sk-your-key-here"

3. Place PDF papers in the `papers/` directory.

## Quick Start

Run the demo (asks four sample questions with citations):

    python scripts/run_demo.py

Run automatic evaluation on the curated QA set:

    python scripts/run_eval.py

## Usage

    from src.pipeline import RAGPipeline

    rag = RAGPipeline(papers_dir="papers", k=3)
    result = rag.ask("What is retrieval-augmented generation?")
    rag.pretty_print(result)

## Current Evaluation Snapshot (Day 9, 15 questions, 4 experimental stages)

**Day 6 baseline (chunk_size=500, vector vs hybrid retrieval)**:

| Config          | Keyword Hit      | Source Hit | Routing Precision |
| --------------- | ---------------- | ---------- | ----------------- |
| vector k=3      | 6/15 (40.0%)     | 100%       | 86.7%             |
| vector k=5      | 8/15 (53.3%)     | 100%       | 86.7%             |
| hybrid k=3      | 5/15 (33.3%)     | 100%       | 84.4%             |
| hybrid k=5      | 9/15 (60.0%)     | 100%       | 82.7%             |

**Day 7 ablation (chunk_size=250, same 4 configs)** — all regressed 2–3 questions (full table in `experiment_log.md`).

**Day 8 reranker (chunk_size=500, cross-encoder on top of hybrid)**:

| Config              | Keyword Hit      | Routing Precision |
| ------------------- | ---------------- | ----------------- |
| rerank k=3          | 8/15 (53.3%)     | 82.2%             |
| rerank k=5          | 9/15 (60.0%)     | 88.0%             |

**Day 9 weighted fusion (chunk_size=500, k=5, α-sweep)**:

| Config              | Keyword Hit      | Routing Precision | Δ vs Day 6 hybrid k=5 |
| ------------------- | ---------------- | ----------------- | --------------------- |
| α=0.0 (= hybrid)    | 9/15 (60.0%)     | 82.7%             | —                     |
| α=0.3               | 8/15 (53.3%)     | 85.3%             | −1                    |
| α=0.5               | 9/15 (60.0%)     | 85.3%             | 0                     |
| **α=0.7**           | **10/15 (66.7%)** | 85.3%            | **+1**                |
| α=1.0 (= rerank)    | 9/15 (60.0%)     | 88.0%             | 0                     |

**Best config: rerank_weighted k=5 @ α=0.7 (66.7% answer accuracy)** — project-wide high. Net +1 question over both hybrid k=5 (Day 6) and rerank k=5 (Day 8). Chunk-level analysis across 5 α values reveals **each failure mode has a different optimal α** — Q06 needs α=1.0, Q08 needs α=0.0, Q11/Q14 want α≥0.5. The α=0.7 peak is specific to this eval set's failure-mode mixture, not a transferable optimum. Next step: per-query-adaptive α via query classification.

See `evaluation/experiment_log.md` for the full Day 6–9 analysis (17 verified lessons), and `evaluation/error_analysis_report.md` for chunk-level audit.

## Author

Guangwen Xiong