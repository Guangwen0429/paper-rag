# Paper-RAG: Citation-Grounded Academic QA

A RAG (Retrieval-Augmented Generation) system for academic paper question answering, with hybrid retrieval, cross-encoder reranking, weighted score fusion, HyDE query rewriting, and citation-grounded answer generation.

## Current Status

🚧 Work in progress — Day 11 of 21

## Features (so far)

- [x] PDF loading and chunking (chunk_size=500, overlap=50)
- [x] OpenAI embedding (text-embedding-3-small) + Chroma vector store
- [x] BM25 keyword retrieval
- [x] Hybrid retrieval (BM25 + dense, RRF fusion)
- [x] Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`), GPU-accelerated
- [x] Weighted score fusion (α-sweep over RRF and cross-encoder scores)
- [x] **HyDE query rewriting** (Day 11) — hypothetical answer passages for vector search
- [x] Citation-grounded answer generation (gpt-4o-mini, T=0)
- [x] Multi-paper support (8 NLP research papers)
- [x] Cross-paper routing via semantic similarity
- [x] Modular architecture (loader / retriever / generator / pipeline)
- [x] Curated evaluation set: 30 questions across single-fact / multi-chunk / cross-paper categories
- [x] Multi-config experiments with chunk-level error analysis
- [x] Reproducibility audit framework (multi-run mean ± std for noise floor calibration)
- [ ] Adaptive α via query classification (planned, Exp M)
- [ ] Streamlit UI (planned)

## Tech Stack

- Python 3.12, PyTorch (CUDA)
- LangChain
- OpenAI (gpt-4o-mini, text-embedding-3-small)
- Chroma vector database
- rank-bm25, sentence-transformers (cross-encoder)

## Project Structure

    paper-rag/
    ├── src/                       # Core modules
    │   ├── loader.py              # PDF loading and chunking
    │   ├── retriever.py           # Vector + BM25 + hybrid + rerank + weighted + HyDE
    │   ├── generator.py           # Prompt construction and LLM generation
    │   └── pipeline.py            # End-to-end RAG pipeline
    ├── scripts/
    │   ├── run_demo.py            # Demo script
    │   ├── run_eval.py            # Multi-config evaluation harness
    │   ├── verify_gpu.py          # CUDA availability check
    │   ├── benchmark_gpu.py       # CPU vs GPU cross-encoder benchmark
    │   ├── analyze_results.py     # Automated failure classification
    │   ├── inspect_question.py    # Per-question chunk inspection
    │   └── inspect_chunks.py      # Full chunk index search
    ├── evaluation/
    │   ├── eval_questions.json                                       # 30-question curated QA set (Day 10)
    │   ├── eval_results_{mode}_k{k}_cs{size}_30q.json                # Day 10/11 results
    │   ├── eval_results_hyde_rerank_k5_cs500_30q_run{1,2,3}.json     # Day 11 HyDE 3-run reproducibility
    │   ├── eval_summary_all_day10_30q.json                           # Day 10 7-config summary
    │   ├── eval_summary_all_day11_30q.json                           # Day 11 8-config summary (incl. HyDE)
    │   ├── eval_summary_hyde_reproducibility_day11_30q.json          # Day 11 HyDE × 3 runs
    │   ├── error_analysis_report.md                                  # Chunk-level analysis report
    │   └── experiment_log.md                                         # Experiment notes (Days 1-11)
    ├── requirements.txt
    └── README.md

## Setup

1. Install dependencies:

        pip install -r requirements.txt

2. Set your OpenAI API key as an environment variable:

        Windows: setx OPENAI_API_KEY "sk-your-key-here"
        Mac/Linux: export OPENAI_API_KEY="sk-your-key-here"

3. Place PDF papers in the `papers/` directory.

4. (Optional) For GPU-accelerated reranking, ensure CUDA is available:

        python scripts/verify_gpu.py

## Quick Start

Run the demo:

    python scripts/run_demo.py

Run the multi-config evaluation:

    python scripts/run_eval.py

## Usage

    from src.pipeline import RAGPipeline

    rag = RAGPipeline(papers_dir="papers", k=3)
    result = rag.ask("What is retrieval-augmented generation?")
    rag.pretty_print(result)

## Evaluation Snapshot (Day 11, 30-question set, chunk_size=500, k=5)

**Day 10 baselines + α-sweep** (`eval_summary_all_day10_30q.json`):

| Config                    | Keyword Hit       | Routing Precision |
|---------------------------|-------------------|-------------------|
| hybrid                    | 17/30 (56.7%)     | 88.0%             |
| rerank                    | 18/30 (60.0%)     | 91.3%             |
| rerank_weighted α=0.0     | 17/30 (56.7%)     | 88.0%             |
| rerank_weighted α=0.3     | 16/30 (53.3%)     | 90.0%             |
| rerank_weighted α=0.5     | 17/30 (56.7%)     | 90.0%             |
| rerank_weighted α=0.7     | 18/30 (60.0%)     | 90.7%             |
| rerank_weighted α=1.0     | 18/30 (60.0%)     | 91.3%             |

**Day 11 HyDE** (3-run reproducibility, `eval_summary_hyde_reproducibility_day11_30q.json`):

| Config                    | Keyword Hit       | Routing Precision |
|---------------------------|-------------------|-------------------|
| **hyde_rerank** (run 1)   | **19/30 (63.3%)** | **92.0%**         |
| **hyde_rerank** (run 2)   | **19/30 (63.3%)** | **92.0%**         |
| **hyde_rerank** (run 3)   | **19/30 (63.3%)** | **92.0%**         |
| **mean ± std**            | **19.0 ± 0.0**    | **92.0%**         |

**Best config: hyde_rerank k=5 (63.3% answer accuracy, 92.0% chunk routing precision)** — project-wide high, +1 question over the rerank baseline with 3-run zero variance. First method-level improvement to clearly exceed the Lesson 21 noise floor (±1 question = LLM API non-determinism at T=0). Q08 chunk-level analysis attributes the gain to chunk-selection-level vocabulary alignment: HyDE's hypothetical passage pulls the vector search toward chunks containing the answer term ("PPO") rather than topic-related chunks lacking it.

## Reproducibility Audit (Day 10)

Same evaluation set, same config, 3 independent runs:

| Config                    | 3-run Keyword Hit | std    |
|---------------------------|-------------------|--------|
| rerank_weighted α=0.0     | 16, 17, 16        | ≈0.5   |
| hyde_rerank (Day 11)      | 19, 19, 19        | 0      |

Single-run differences ≤1 question (3.3%) on this 30q set are within LLM API non-determinism at T=0 and require N≥3 averaging to interpret. The +1 question gain from HyDE is the first to be reproduced across all 3 independent runs without variance.

See `evaluation/experiment_log.md` for the full Day 1–11 analysis (28 verified lessons), and `evaluation/error_analysis_report.md` for chunk-level audit.

## Author

Guangwen Xiong