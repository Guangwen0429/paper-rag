# Paper-RAG: Citation-Grounded Academic QA

A RAG (Retrieval-Augmented Generation) system for academic paper question answering, with hybrid retrieval, reranking, and citation-grounded answer generation.

## Current Status

🚧 Work in progress — Day 7 of 21

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
- [x] Hybrid retrieval (BM25 + dense)
- [ ] Reranking
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
    │   ├── eval_questions.json                          # 15-question curated QA set
    │   ├── eval_results_{mode}_k{k}_15q.json            # Day 6 results (cs=500)
    │   ├── eval_results_{mode}_k{k}_cs250_15q.json      # Day 7 results (cs=250)
    │   ├── eval_summary_all.json                        # Day 6 summary
    │   ├── eval_summary_all_day7.json                   # Day 7 summary
    │   ├── error_analysis_report.md                     # Chunk-level analysis report
    │   └── experiment_log.md                            # Experiment notes (Days 1-7)
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

## Current Evaluation Snapshot (Day 7, 15 questions, 4 configs × 2 chunk sizes)

**Day 6 baseline (chunk_size=500)**:

| Config          | Keyword Hit      | Source Hit | Routing Precision |
| --------------- | ---------------- | ---------- | ----------------- |
| vector k=3      | 6/15 (40.0%)     | 100%       | 86.7%             |
| vector k=5      | 8/15 (53.3%)     | 100%       | 86.7%             |
| hybrid k=3      | 5/15 (33.3%)     | 100%       | 84.4%             |
| **hybrid k=5**  | **9/15 (60.0%)** | **100%**   | 82.7%             |

**Day 7 ablation (chunk_size=250)**:

| Config          | Keyword Hit      | Source Hit | Routing Precision | Δ vs Day 6 |
| --------------- | ---------------- | ---------- | ----------------- | ---------- |
| vector k=3      | 4/15 (26.7%)     | 100%       | 86.7%             | −2         |
| vector k=5      | 6/15 (40.0%)     | 100%       | 88.0%             | −2         |
| hybrid k=3      | 3/15 (20.0%)     | 100%       | 75.6%             | −2         |
| hybrid k=5      | 6/15 (40.0%)     | 100%       | 73.3%             | −3         |

**Best config remains hybrid k=5 at chunk_size=500 (60% answer accuracy)**. The cs=500 → cs=250 ablation regressed all four configs by 2–3 questions, revealing that reducing chunk_size trades "embedding purity on short key sentences" for "context completeness + BM25 stability" — a net-negative trade on academic-paper QA.

See `evaluation/experiment_log.md` for the full Day 6–7 analysis (12 verified retrieval failure modes, including 4 new ones introduced by cs=250), and `evaluation/error_analysis_report.md` for chunk-level audit.

## Author

Guangwen Xiong