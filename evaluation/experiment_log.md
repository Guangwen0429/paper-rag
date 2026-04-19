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