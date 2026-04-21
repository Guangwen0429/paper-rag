# Paper-RAG: Citation-Grounded Academic QA

A RAG (Retrieval-Augmented Generation) system for academic paper question answering, with hybrid retrieval, reranking, and citation-grounded answer generation.

## Current Status

🚧 Work in progress — Day 6 of 21

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
    │   ├── eval_questions.json              # 15-question curated QA set
    │   ├── eval_results_{mode}_k{k}_15q.json # Per-config evaluation results
    │   ├── eval_summary_all.json            # Summary across all configs
    │   ├── error_analysis_report.md         # Chunk-level analysis report
    │   └── experiment_log.md                # Experiment notes (Days 1-6)
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

## Current Evaluation Snapshot (Day 6, 15 questions, 4 configs)

| Config          | Keyword Hit    | Source Hit | Routing Precision |
| --------------- | -------------- | ---------- | ----------------- |
| vector k=3      | 6/15 (40.0%)   | 100%       | 86.7%             |
| vector k=5      | 8/15 (53.3%)   | 100%       | 86.7%             |
| hybrid k=3      | 5/15 (33.3%)   | 100%       | 84.4%             |
| **hybrid k=5**  | **9/15 (60.0%)** | **100%** | 82.7%             |

Best config: **hybrid k=5 (60% answer accuracy)**. All 6 failed questions under this config have been verified at chunk level as retrieval failures (not generator failures), motivating chunking / query-rewriting / reranker improvements as next steps.

See `evaluation/experiment_log.md` for full Day 6 analysis including 9 verified retrieval failure modes, and `evaluation/error_analysis_report.md` for chunk-level audit.

## Author

Guangwen Xiong