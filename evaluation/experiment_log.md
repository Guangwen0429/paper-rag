# Experiment Log

Baseline system: `chunk_size=500`, `chunk_overlap=50`, `k=3`, `embedding=text-embedding-3-small`, `llm=gpt-4o-mini (temperature=0)`.

---

## Exp 1: Increase k from 3 to 5

**Date**: 2026-04-19
**Motivation**: Baseline evaluation (k=3) showed Keyword Hit 50%. Chunk-level inspection of failed questions revealed possible retrieval recall issues; test whether increasing k improves coverage.

### Hypothesis

Increasing k from 3 to 5 will expand retrieval coverage and improve Keyword Hit on failed questions.

### Results

| Metric | k=3 | k=5 | Δ |
|--------|-----|-----|---|
| Keyword Hit | 3/6 (50%) | 4/6 (66.7%) | **+16.7%** |
| Source Hit | 6/6 (100%) | 6/6 (100%) | 0 |

### Per-question change

| Q | k=3 | k=5 | Note |
|---|-----|-----|------|
| Q01 | ✓ | ✓ | Stable |
| Q02 | ✗ | ✗ | Cross-paper contamination; BERT chunks dominate |
| Q03 | ✓ | ✓ | Stable |
| Q04 | ✓ | ✓ | Stable |
| Q05 | ✗ | **✓** | Recovered; authoritative chunk (page 4) retrieved at rank 4 |
| Q06 | ✗ | ✗ | Authoritative paragraph (Section 3 Encoders) not retrieved at rank ≤5 |

### Root cause analysis (validated via full-chunk inspection)

Initial hypothesis for Q05 was "chunking boundary cut". After retrieving all 5 chunks in full, the real picture is:

- **k=3 top chunks** (page 77, 35, 48) do NOT contain the authoritative claim about LLaMA 2's largest size.
- **Source 2 (page 35)** contains "With 70B parameters, Chinchilla (Hoffmann et al., 2022)..." — attributing 70B to Chinchilla, not LLaMA 2. GPT correctly refused to attribute.
- **k=5 additionally retrieves page 4**: *"We are releasing variants of Llama 2 with 7B, 13B, and 70B parameters."* — the direct, authoritative statement.

**Actual failure mode for Q05**: *Retrieval recall miss*, not chunking cut. Same failure category as Q06.

### Why is the authoritative chunk ranked only 4th by vector retrieval?

Not because "the key sentence is short". OpenAI embeddings are normalized unit vectors; cosine similarity is length-independent.

The real cause is **semantic signal dilution**. The page-4 chunk mixes "70B parameters" with other topics (corpus size, context length, GQA). Its overall embedding reflects a mix of subjects, not a pure "model scale" signal.

By contrast, the page-35 chunk is uniformly about "LLM parameter counts at scale" (discussing GPT-3, Gopher, Chinchilla as 100B+ models). Its embedding direction is more topically aligned with the query "largest model size in the LLaMA 2 family" — despite actually referring to *other* models.

This reveals a **vector retrieval blind spot**: the topic-consistency preference of embedding-based similarity can favor chunks that are *about* the query topic over chunks that contain the *literal answer*. BM25 (keyword-level lexical matching) does not share this bias — motivating hybrid retrieval as the next experiment.

### Failure type summary (updated)

| Q | Failure type |
|---|--------------|
| Q02 | Cross-paper contamination (BERT paper chunks dominate when asking about Transformer) |
| Q05 | Retrieval recall miss — authoritative chunk at rank 4 |
| Q06 | Retrieval recall miss — authoritative chunk at rank > 5 |

### Takeaway

- Increasing k helps when the authoritative chunk is ranked around 4–5 (Q05).
- It does NOT help when the chunk is ranked far outside top-k (Q06) or when the issue is cross-paper contamination (Q02).
- These remaining failures motivate hybrid retrieval (BM25 + dense) as Exp D.

### Next experiments

- **Exp B**: chunk_overlap 50 → 200. Hypothesis: limited effect, since no true chunking-cut failures were confirmed in the current 6-question set.
- **Exp C**: chunk_size 500 → 1000. Hypothesis: may help Q06 if larger chunks pack Section 3 Encoders together with other high-signal DPR content, lifting its similarity score.
- **Exp D** (Day 5): Hybrid retrieval with BM25. Hypothesis: lexical matching of "Transformer", "BERT", "encoder" should directly address Q02 and Q06.

---

## Exp D: Hybrid Retrieval (BM25 + Vector, RRF fusion)

**Date**: 2026-04-20
**Motivation**: Exp 1 (k=5) recovered Q05 but left Q02 and Q06 unchanged. Hypothesis: BM25 lexical matching can complement vector retrieval on cross-paper contamination (Q02) and retrieval blind spots (Q06).

### Implementation

- BM25 index built with `rank_bm25` library, using simple regex-based tokenization (lowercase, `\w+` splitting)
- Hybrid retrieval combines top-20 candidates from each retriever via Reciprocal Rank Fusion (RRF) with `k=60`, returns top-3

### Results

| Metric | Baseline k=3 | Exp 1 (k=5) | **Exp D (Hybrid k=3)** |
|--------|-------------|-------------|----------------------|
| Keyword Hit | 3/6 (50%) | 4/6 (66.7%) | **4/6 (66.7%)** |
| Source Hit | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| **Answer error** (answered but wrong) | 0 | 0 | **1** ← new failure mode |

### Per-question change

| Q | Baseline | k=5 | **Hybrid** | Note |
|---|----------|-----|------------|------|
| Q01 | ✓ | ✓ | ✓ | Stable |
| Q02 | ✗ (refuse) | ✗ (refuse) | **✗ (wrong: "12")** | **New failure mode: refusal → wrong answer** |
| Q03 | ✓ | ✓ | ✓ | Stable |
| Q04 | ✓ | ✓ | ✓ | Stable |
| Q05 | ✗ (refuse) | ✓ | **✓** | Recovered by BM25 matching "Llama 2" / "70B" |
| Q06 | ✗ (refuse) | ✗ (refuse) | ✗ (refuse) | Vocabulary mismatch: query has no "BERT", answer needs "BERT" |

### Root cause analysis for Q02 failure (validated via full-chunk inspection)

Hybrid retrieved top-3 chunks for Q02:

- **Rank 1: BERT paper, page 3** — contains "self-attention heads as A... BERT_BASE (L=12, H=768, A=12, ...)"
- **Rank 2: Transformer paper, page 9** — Conclusion section, discusses multi-head self-attention, no numbers
- **Rank 3: Transformer paper, page 5** — Section 3.2.3, describes three uses of multi-head attention, no numbers

BERT paper Section 3.1 introduces Transformer terminology (L, H, A) before defining BERT, so the chunk reads as if it is describing Transformer architecture itself. Vector retrieval matches it semantically; BM25 matches "attention heads" lexically. RRF fusion pushes this chunk to rank 1.

GPT sees 2/3 chunks from Transformer paper (no numbers) and 1/3 from BERT paper (A=12). Because Transformer chunks are the majority, GPT treats Transformer as the main subject and **incorrectly attributes the A=12 figure from the BERT chunk to the base Transformer**. Answer: "12 attention heads" — confidently wrong.

### Key findings

1. **Hybrid's gain is offset by a new failure mode**. Q05 recovered, but Q02 went from "refuse" to "wrong". Net keyword-hit gain is identical to Exp 1 (k=5), but Exp D introduces answer errors, which carry higher user-facing risk than refusals.

2. **Cross-paper reference contamination** is a distinct failure mode that neither vector retrieval nor BM25 alone can solve. When paper A extensively references paper B (BERT references Transformer), chunks from A can read as if describing B's subject, misleading both retrieval and downstream generation.

3. **Keyword Hit metric hides answer-error risk**. Both "refuse" and "wrong" count equally as failures, but they have very different downstream consequences. Future evaluation should separately track: answered correctly / refused / answered wrong.

4. **Q06 remains unsolvable by lexical + semantic retrieval**. The query lacks the answer's key term ("BERT"), so no amount of BM25 weight shifts can surface the answer paragraph. This is a true vocabulary-mismatch failure — only query rewriting (HyDE or similar) can address it.

### Recommendation

On the current 6-question eval set, **Exp 1 (k=5) outperforms Exp D (hybrid)** in practice: same answer-correctness rate with zero answer-error risk. However, 6 questions is too small a sample to draw a conclusion. Next steps:

- Expand eval set to 15–30 questions to test whether hybrid's theoretical advantage materializes at scale
- Add cross-encoder reranker after hybrid retrieval to filter cross-paper contamination
- Introduce a separate "answer error" metric alongside "keyword hit"