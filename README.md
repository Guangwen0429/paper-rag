# Paper-RAG: Citation-Grounded Academic QA

A RAG (Retrieval-Augmented Generation) system for academic paper question answering, with hybrid retrieval, cross-encoder reranking, weighted fusion, and citation-grounded answer generation. Includes a 30-question curated evaluation set with chunk-level error analysis and a measured LLM-noise floor.

## Current Status

🚧 Work in progress — Day 10 of 21

## Features (so far)

- [x] PDF loading and chunking
- [x] OpenAI embedding + Chroma vector store
- [x] Semantic retrieval
- [x] Citation-grounded answer generation
- [x] Multi-paper support (8 NLP research papers)
- [x] Cross-paper routing via semantic similarity
- [x] Modular architecture (loader / retriever / generator / pipeline)
- [x] Hybrid retrieval (BM25 + dense, RRF fusion)
- [x] 30-question curated evaluation set with single-fact / multi-chunk / cross-paper categories
- [x] Controlled multi-config experiments (vector / hybrid / rerank / weighted-fusion × k=3/5)
- [x] Chunking ablation (chunk_size 250 vs 500) with context-fragmentation analysis
- [x] Cross-encoder reranker with answer-match vs topic-match failure analysis
- [x] Weighted fusion of hybrid + cross-encoder scores (full α-sweep)
- [x] GPU-accelerated cross-encoder rerank (10× single-stage, 2× end-to-end on RTX 4060)
- [x] Reproducibility audit (multi-run identical-config) — measured LLM ±1-question noise floor at T=0
- [ ] Per-query-adaptive α via query classification
- [ ] Streamlit UI

## Tech Stack

- Python 3.12
- LangChain + Chroma vector database
- OpenAI gpt-4o-mini (generator), text-embedding-3-small (retriever)
- sentence-transformers + Hugging Face `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker)
- PyTorch with optional CUDA acceleration

## Project Structure

    paper-rag/
    ├── src/                       # Core modules
    │   ├── loader.py              # PDF loading and chunking
    │   ├── retriever.py           # Vector / BM25 / hybrid / rerank / weighted fusion (auto-GPU)
    │   ├── generator.py           # Prompt construction and LLM generation
    │   └── pipeline.py            # End-to-end RAG pipeline
    ├── scripts/
    │   ├── run_demo.py            # Demo script (4 sample questions with citations)
    │   ├── run_eval.py            # Multi-config evaluation harness (Day 6-10 sweeps)
    │   ├── analyze_results.py     # Automated failure classification
    │   ├── inspect_question.py    # Per-question chunk inspection tool
    │   ├── inspect_chunks.py      # Full chunk index search tool
    │   ├── verify_gpu.py          # GPU health check (Day 10)
    │   └── benchmark_gpu.py       # End-to-end CPU vs GPU benchmark (Day 10)
    ├── evaluation/
    │   ├── eval_questions.json                                       # 30-question curated QA set
    │   ├── eval_results_{mode}_k{k}_15q.json                         # Day 6 results (cs=500)
    │   ├── eval_results_{mode}_k{k}_cs{size}_15q.json                # Day 7/8 results
    │   ├── eval_results_rerank_weighted_k5_cs500_a{alpha}_15q.json   # Day 9 α-sweep (15q)
    │   ├── eval_results_{mode}_k5_cs500_30q.json                     # Day 10 baselines
    │   ├── eval_results_rerank_weighted_k5_cs500_a{alpha}_30q.json   # Day 10 α-sweep (30q)
    │   ├── eval_results_rerank_weighted_k5_cs500_a0.0_30q_run{N}.json  # Day 10 reproducibility audit
    │   ├── eval_summary_all.json                                     # Day 6 summary
    │   ├── eval_summary_all_day7.json                                # Day 7 summary
    │   ├── eval_summary_all_day8.json                                # Day 8 summary
    │   ├── eval_summary_all_day9.json                                # Day 9 summary
    │   ├── eval_summary_all_day10_30q.json                           # Day 10 summary
    │   ├── eval_summary_alpha_sweep_day10_30q.json                   # Day 10 α-sweep summary
    │   ├── eval_summary_reproducibility_run{N}.json                  # Day 10 audit summary
    │   ├── error_analysis_report.md                                  # Chunk-level analysis report
    │   └── experiment_log.md                                         # Experiment notes (Days 1-10)
    ├── papers/                    # 8 NLP research papers (gitignored)
    ├── requirements.txt
    └── README.md

## Setup

1. Install dependencies:

        pip install -r requirements.txt

2. Set your OpenAI API key as an environment variable:

        Windows: setx OPENAI_API_KEY "sk-your-key-here"
        Mac/Linux: export OPENAI_API_KEY="sk-your-key-here"

3. Place PDF papers in the `papers/` directory.

### Optional: GPU acceleration

If you have an NVIDIA GPU, install the CUDA build of PyTorch for ~10× cross-encoder speedup
(~2× end-to-end pipeline speedup, bottlenecked by OpenAI embedding API latency):

    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Verify the install:

    python scripts/verify_gpu.py

The cross-encoder will automatically use GPU when available, with CPU fallback otherwise —
no code changes needed.

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

## Current Evaluation Snapshot (Day 10, 30 questions, with reproducibility audit)

**Eval set expanded from 15 to 30 questions on Day 10**, rebalancing paper coverage (every paper now has 3–7 questions) and adding category-balanced new items. All ground-truth answers verified against original PDFs.

**Day 10 main 7-config sweep on 30q (chunk_size=500, k=5)**:

| Config                        | Keyword Hit       | Routing Precision |
| ----------------------------- | ----------------- | ----------------- |
| hybrid k=5                    | 17/30 (56.7%)     | 88.0%             |
| rerank k=5                    | **18/30 (60.0%)** | **91.3%**         |
| rerank_weighted α=0.0         | 16/30 (53.3%)     | 88.0%             |
| rerank_weighted α=0.3         | 16/30 (53.3%)     | 90.0%             |
| rerank_weighted α=0.5         | 16/30 (53.3%)     | 90.0%             |
| rerank_weighted α=0.7         | 17/30 (56.7%)     | 90.7%             |
| rerank_weighted α=1.0         | **18/30 (60.0%)** | **91.3%**         |

**Reproducibility audit (3 runs of `rerank_weighted α=0.0`, identical config)**:

| Run | Keyword Hit | Routing Precision |
|---|:---:|:---:|
| Run 1 | 16/30 (53.3%) | 88.0% |
| Run 2 | 17/30 (56.7%) | 88.0% |
| Run 3 | 16/30 (53.3%) | 88.0% |

**Routing is fully deterministic; LLM generation contributes ±1-question noise even at temperature=0.** This sets a hard noise floor: single-run differences ≤2 questions on the 30q set are within LLM API non-determinism and should not be interpreted as method-level differences.

**Cross-eval-set stability (Day 9 vs Day 10)**:

| Config | 15q | 30q | Δ |
|---|:---:|:---:|:---:|
| hybrid k=5 | 60.0% | 56.7% | -3.3pp |
| **rerank k=5** | **60.0%** | **60.0%** | **0.0pp ✓** |
| rerank_weighted α=0.7 | **66.7%** | 56.7% | **-10.0pp** |
| **rerank_weighted α=1.0** | **60.0%** | **60.0%** | **0.0pp ✓** |

The Day 9 α=0.7 peak (66.7%) does not transfer to 30q, exactly as Lesson 16 predicted: a single scalar α optimized on a small set overfits to that set's specific failure-mode mixture. The two rerank-equivalent configs (`rerank` and `rerank_weighted α=1.0`) are the only configurations stable across both eval sets — stability across eval sets is a stronger property than peak performance on a small set.

**Performance**: GPU acceleration brings end-to-end per-query latency from 763 ms (CPU) to 388 ms (RTX 4060), a 1.97× speedup. Cross-encoder rerank in isolation is 10.1× faster on GPU; end-to-end gain is bounded by OpenAI embedding API network latency (~400 ms / query) per Amdahl's law.

See `evaluation/experiment_log.md` for the full Day 1–10 analysis (24 verified lessons, including failure-mode taxonomy, chunk-level error mechanisms, and LLM noise quantification), and `evaluation/error_analysis_report.md` for chunk-level audit.

## Author

Guangwen Xiong