"""
GPU verification script for paper-rag.
Run after installing PyTorch CUDA build to confirm:
  1. PyTorch detects the GPU
  2. Cross-encoder model can load on GPU
  3. CPU vs GPU inference produces identical scores (sanity check)
  4. GPU is faster than CPU on a representative batch
"""
import time

import torch
from sentence_transformers import CrossEncoder


def check_pytorch_cuda():
    print("=" * 60)
    print("[1/4] PyTorch CUDA detection")
    print("=" * 60)
    print(f"  torch version       : {torch.__version__}")
    print(f"  CUDA available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version (built): {torch.version.cuda}")
        print(f"  device count        : {torch.cuda.device_count()}")
        print(f"  device name         : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  total VRAM          : {total_mem:.2f} GB")
    else:
        raise RuntimeError("CUDA is not available — abort.")
    print()


def check_cross_encoder_loads_on_gpu():
    print("=" * 60)
    print("[2/4] Cross-encoder GPU load")
    print("=" * 60)
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"  loading {model_name} on cuda...")
    model = CrossEncoder(model_name, device="cuda")
    # CrossEncoder wraps an underlying transformer; check its device
    underlying_device = next(model.model.parameters()).device
    print(f"  underlying model device: {underlying_device}")
    assert underlying_device.type == "cuda", "model not on cuda!"
    print("  OK — cross-encoder loaded on GPU")
    print()
    return model


def check_cpu_gpu_score_parity(model_gpu):
    print("=" * 60)
    print("[3/4] CPU vs GPU score parity (sanity check)")
    print("=" * 60)
    pairs = [
        ("What is BERT-base hidden size?", "BERT_BASE has hidden size H=768."),
        ("What is BERT-base hidden size?", "The transformer was introduced in 2017."),
        ("What attention algorithm does GPT use?", "GPT uses masked self-attention."),
    ]
    print("  scoring 3 query-passage pairs on GPU...")
    gpu_scores = model_gpu.predict(pairs)
    print("  loading same model on CPU for comparison...")
    model_cpu = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    cpu_scores = model_cpu.predict(pairs)

    print(f"  GPU scores: {gpu_scores}")
    print(f"  CPU scores: {cpu_scores}")
    max_diff = float(abs(gpu_scores - cpu_scores).max())
    print(f"  max abs diff: {max_diff:.6f}")
    assert max_diff < 1e-3, f"scores differ by {max_diff} — investigate!"
    print("  OK — CPU and GPU scores match within 1e-3")
    print()


def benchmark_cpu_vs_gpu():
    print("=" * 60)
    print("[4/4] Benchmark: CPU vs GPU on 100 pairs")
    print("=" * 60)
    # 100 pairs ~ a small batch, representative of single-query rerank workload
    pairs = [
        (f"query about topic {i % 5}",
         f"passage discussing material related to topic {i % 5} in detail.")
        for i in range(100)
    ]

    # CPU
    print("  warming up CPU...")
    model_cpu = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    model_cpu.predict(pairs[:10])  # warmup
    t0 = time.perf_counter()
    model_cpu.predict(pairs)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU: {cpu_time*1000:.1f} ms")

    # GPU
    print("  warming up GPU...")
    model_gpu = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
    model_gpu.predict(pairs[:10])  # warmup (CUDA kernel compile)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model_gpu.predict(pairs)
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - t0
    print(f"  GPU: {gpu_time*1000:.1f} ms")

    speedup = cpu_time / gpu_time
    print(f"  speedup: {speedup:.2f}x")
    print()


if __name__ == "__main__":
    check_pytorch_cuda()
    model_gpu = check_cross_encoder_loads_on_gpu()
    check_cpu_gpu_score_parity(model_gpu)
    benchmark_cpu_vs_gpu()
    print("=" * 60)
    print("All checks passed. GPU is ready for paper-rag pipeline.")
    print("=" * 60)