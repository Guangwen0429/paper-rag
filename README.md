# Paper-RAG: Citation-Grounded Academic QA

A RAG (Retrieval-Augmented Generation) system for academic paper question answering, with hybrid retrieval, reranking, and citation-grounded answer generation.

## Current Status

🚧 Work in progress — Day 4 of 21

## Features (so far)

- [x] PDF loading and chunking
- [x] OpenAI embedding + Chroma vector store
- [x] Semantic retrieval
- [x] Citation-grounded answer generation
- [x] Multi-paper support (8 NLP research papers)
- [x] Cross-paper routing via semantic similarity
- [x] Modular architecture (loader / retriever / generator / pipeline)
- [x] Evaluation set (6 questions, single-fact category) with automatic metrics
- [x] First controlled experiment (k=3 vs k=5) with chunk-level error analysis
- [ ] Hybrid retrieval (BM25 + dense)
- [ ] Reranking
- [ ] Full evaluation set (target: 15–30 questions)
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
    │   ├── run_demo.py            # Demo script
    │   └── run_eval.py            # Automatic evaluation on eval set
    ├── evaluation/
    │   ├── eval_questions.json    # Manually curated QA evaluation set
    │   ├── eval_results.json      # Latest evaluation run output
    │   └── experiment_log.md      # Experiment notes and error analysis
    ├── papers/                    # Research papers (not tracked in git)
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

## Current Evaluation Snapshot (Day 4, 6 questions, baseline k=3)

| Metric | Value |
|--------|-------|
| Keyword Hit (answer accuracy) | 50% |
| Source Hit (routing accuracy) | 100% |
| Avg chunk routing precision | 72% |

See `evaluation/experiment_log.md` for full error analysis, failure mode categorization, and the k=3 → k=5 controlled experiment.

## Author

Guangwen Xiong