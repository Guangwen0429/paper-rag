"""
End-to-end benchmark: CPU vs GPU on the actual paper-rag pipeline.

Runs hybrid_then_rerank_weighted at alpha=0.7, k=5, on all 15 eval questions,
once with cross-encoder on CPU and once on GPU. Reports per-question and
total wall-clock time.

This benchmark only times the rerank stage's wall-clock difference because
indexing (vector + BM25) is a one-time cost reused across both runs.
"""
import json
import sys
import time
from pathlib import Path

# Make `src/` importable when running this script directly:
#   python scripts/benchmark_gpu.py
# Without this, Python's import path doesn't include the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.loader import load_all_pdfs, split_into_chunks
from src.retriever import (
    build_vectorstore,
    build_bm25_index,
    hybrid_then_rerank_weighted,
)
import src.retriever as retriever_module
from sentence_transformers import CrossEncoder


EVAL_PATH = PROJECT_ROOT / "evaluation" / "eval_questions.json"
PAPERS_DIR = PROJECT_ROOT / "papers"

ALPHA = 0.7
K = 5
N_CANDIDATES = 20
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def force_reload_cross_encoder(device: str):
    """
    Forcibly reload the cross-encoder on a specific device.
    Bypasses the module-level singleton cache so we can compare CPU vs GPU.
    """
    retriever_module._cross_encoder_instance = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=device,
    )
    print(f"  [benchmark] cross-encoder reloaded on {device}")


def run_all_questions(vs, bm25, chunks, questions):
    """Run hybrid_then_rerank_weighted on all questions, return per-question times."""
    times = []
    for q in questions:
        t0 = time.perf_counter()
        _ = hybrid_then_rerank_weighted(
            vs, bm25, chunks, q["question"],
            k=K, n_candidates=N_CANDIDATES, alpha=ALPHA,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return times


def main():
    # ---- Load data once (shared between CPU and GPU runs) ----
    print("=" * 60)
    print("Setup: loading PDFs, building indices")
    print("=" * 60)
    docs = load_all_pdfs(str(PAPERS_DIR))
    chunks = split_into_chunks(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vs = build_vectorstore(chunks)
    bm25, _ = build_bm25_index(chunks)

    with open(EVAL_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  loaded {len(questions)} eval questions")
    print()

    # ---- CPU run ----
    print("=" * 60)
    print(f"CPU run: alpha={ALPHA}, k={K}, n_candidates={N_CANDIDATES}")
    print("=" * 60)
    force_reload_cross_encoder("cpu")
    # warmup (first call has overhead from inference graph build)
    _ = hybrid_then_rerank_weighted(
        vs, bm25, chunks, questions[0]["question"],
        k=K, n_candidates=N_CANDIDATES, alpha=ALPHA,
    )
    cpu_times = run_all_questions(vs, bm25, chunks, questions)
    cpu_total = sum(cpu_times)
    print(f"  CPU total: {cpu_total*1000:.0f} ms ({cpu_total/len(questions)*1000:.1f} ms/q)")
    print()

    # ---- GPU run ----
    if not torch.cuda.is_available():
        print("CUDA not available — skipping GPU run.")
        return
    print("=" * 60)
    print(f"GPU run: alpha={ALPHA}, k={K}, n_candidates={N_CANDIDATES}")
    print("=" * 60)
    force_reload_cross_encoder("cuda")
    # warmup (CUDA kernel JIT compile)
    _ = hybrid_then_rerank_weighted(
        vs, bm25, chunks, questions[0]["question"],
        k=K, n_candidates=N_CANDIDATES, alpha=ALPHA,
    )
    torch.cuda.synchronize()
    gpu_times = run_all_questions(vs, bm25, chunks, questions)
    gpu_total = sum(gpu_times)
    print(f"  GPU total: {gpu_total*1000:.0f} ms ({gpu_total/len(questions)*1000:.1f} ms/q)")
    print()

    # ---- Comparison ----
    print("=" * 60)
    print("Result")
    print("=" * 60)
    speedup = cpu_total / gpu_total
    print(f"  speedup (end-to-end pipeline): {speedup:.2f}x")
    print(f"  CPU per-q: {cpu_total/len(questions)*1000:.1f} ms")
    print(f"  GPU per-q: {gpu_total/len(questions)*1000:.1f} ms")
    print(f"  saved per question: {(cpu_total - gpu_total)/len(questions)*1000:.1f} ms")
    print()
    print("  Note: end-to-end speedup is smaller than the cross-encoder-only")
    print("  benchmark because retrieval (vector search + BM25) runs on CPU")
    print("  in both cases. The GPU only accelerates the rerank stage.")


if __name__ == "__main__":
    main()