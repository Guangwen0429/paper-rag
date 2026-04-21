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

> **Note (updated Day 6)**: The Q02 "wrong: 12" failure under hybrid k=3
> shown above is **resolved when k is increased to 5** (see Day 6 Exp E).
> This suggests the Q02 failure was a hybrid-at-small-k issue — BM25's
> lexical matches pushed the BERT reference chunk to rank 1 in the k=3
> window, leaving no room for the correct Transformer §3.2.2 context.
> At k=5 the additional slots allow both BM25's lexical picks and the
> semantically-specific Transformer chunk to coexist. The "cross-paper
> reference contamination" mechanism described below remains valid as a
> description of *what went wrong under hybrid k=3*, but is not an
> intrinsic property of hybrid retrieval.

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

---

> **⚠️ Revised in Day 6**: The conclusion above ("Exp 1 k=5 outperforms
> Exp D hybrid") was based on a 6-question eval set and an unfair comparison
> (hybrid k=3 vs vector k=5). Day 6 expanded the eval set to 15 questions
> and added controlled hybrid k=5 comparison, which shows hybrid k=5 (60.0%)
> actually outperforms vector k=5 (53.3%). The "context contamination" failure
> mode described for Q02 is now understood to be caused by insufficient k,
> not by hybrid retrieval itself — Q02 is fixed under hybrid k=5.
> See Day 6 for full analysis.

---

## Exp E: Scaled Evaluation (15 questions × 4 configs) with Chunk-Level Error Analysis

**Date**: 2026-04-21
**Motivation**: Day 5 Exp D concluded hybrid retrieval was marginally worse than vector k=5 on a 6-question eval set. That conclusion had two problems: (a) 6 questions is too small to distinguish mechanism effects from noise; (b) the comparison was **unfair** — hybrid k=3 was compared to vector k=5, conflating the retrieval mechanism with the k parameter. This experiment re-runs all configs on a 15-question set with controlled-variable analysis, plus chunk-level error inspection to verify every failure classification.

### Setup

**Eval set expansion** (`evaluation/eval_questions.json`, 6 → 15 questions):

| Category | Count | IDs |
|---|---|---|
| single_fact | 9 | Q01–Q09 |
| multi_chunk | 3 | Q10, Q11, Q12 |
| cross_paper | 3 | Q13, Q14, Q15 |

Paper coverage extended from 5/8 papers (Day 4/5) to **8/8 papers** including previously untested RAG, InstructGPT, and CoT papers. All 15 ground-truth answers manually verified against original paper PDFs before running experiments.

**Experiment matrix**: 4 configs × 15 questions = 60 runs, executed in a single script pass with shared vectorstore/BM25 indices for reproducibility. Per-chunk contents (source, page, rank, text) saved for every retrieval — enabling chunk-level error analysis that was not possible in Day 4/5.

### Results

| Config | Keyword Hit | Source Hit | Routing Precision |
|---|:---:|:---:|:---:|
| vector k=3 | 6/15 (40.0%) | 15/15 | 86.7% |
| vector k=5 | 8/15 (53.3%) | 15/15 | 86.7% |
| hybrid k=3 | 5/15 (33.3%) | 15/15 | 84.4% |
| **hybrid k=5** | **9/15 (60.0%)** | 15/15 | 82.7% |

### Controlled-variable comparisons

**At k=3 (vector vs hybrid)**: vector 40% > hybrid 33.3%. Hybrid crowds the small retrieval window with BM25's lexical picks, displacing semantically-specific chunks.

**At k=5 (vector vs hybrid)**: vector 53.3% < **hybrid 60.0%**. With a larger window, lexical and semantic signals coexist; BM25's coverage contributions are net positive.

**Fixed mechanism, variable k**: vector gains +13.3% from k=3→k=5; hybrid gains **+26.7%**. Hybrid's marginal return on k is roughly 2× vector's.

**Interpretation**: Hybrid retrieval is not intrinsically better or worse than vector — its usefulness is **k-dependent**. Below a certain context budget, the BM25 signal acts as noise crowding out specific semantic matches. Above it, the two signals complement each other.

This overturns Day 5's conclusion (hybrid marginally worse) while confirming the motivation for hybrid in the first place — just with a narrower operating range than initially assumed.

### Chunk-level error analysis tooling

Three scripts work together to move from summary metrics to chunk-level evidence:

- `scripts/run_eval.py` — runs the 4-config matrix, saves per-chunk content in every result file
- `scripts/inspect_question.py` — prints retrieved chunks for a given QID across all 4 configs, with keyword highlighting
- `scripts/inspect_chunks.py` — dumps or searches the full 2,425-chunk index, used to locate authoritative answer chunks and verify whether they were ever retrieved

This tooling made it possible to verify 6 of the 6 failure cases against original PDF text — see next section.

### Verified failure mode taxonomy (hybrid k=5)

All 6 failed questions under the best config (hybrid k=5) were inspected at chunk level. **Every one is a retrieval failure; none is a generator failure.** This contradicts the automated classification in `error_analysis_report.md` Section 5 (see that file's Addendum for specifics).

| QID | Failure mode | Mechanism (verified) |
|---|---|---|
| Q06 | **Definitional blind spot** | DPR §3's *"use two independent BERT networks"* chunk has its key sentence embedded in a larger discussion of encoder alternatives; embedding direction is dominated by the discussion, not the definition. Never retrieved. |
| Q07 | **BM25 noise intrusion** | Vector k=5 retrieves BART-containing chunks at rank 3 and 4. Hybrid k=5 replaces them with chunks rich in RAG-internal high-frequency terms ("RAG-Token", "RAG-Sequence"), plus one chunk from `01_transformer.pdf` that shares query words ("training", "model"). BM25 systematically penalizes cross-paper references (BART is mentioned only a few times in the RAG paper). |
| Q11 | **Embedding blind spot** | The authoritative chunk (`#1831`, *"releasing variants of Llama 2 with 7B, 13B, and 70B parameters"*) exists but was never retrieved. Its embedding is pulled toward the adjacent sentence about 34B delay, making it semantically ambiguous on the "released" dimension. |
| Q12 | **Scattered decoy trap** | Three expected keywords (`supervised`, `reward`, `PPO`) all appear across retrieved chunks — but each in an **unrelated** context (deployment history, ablation study, abstract wording). The canonical §3.1 three-step chunk was never retrieved. LLM correctly refused to answer. |
| Q14 | **Formula-diluted semantic signal + cross-paper attribution** | Transformer §3.2.2 chunk containing *"we employ h = 8 parallel attention layers"* (chunk #27, verified via `inspect_chunks.py`) was never retrieved in any of the 4 configs. Its embedding is dominated by ~200 characters of LaTeX projection-matrix formulas preceding the key sentence. LLM used a substitute source — BERT paper p7's citation *"Vaswani et al. (2017) is (L=6, H=1024, A=16)"* (Transformer *big*, not *base*) — producing a confident but wrong "16 heads" answer. |
| Q15 | **Compound query smearing** | The query asks for both Transformer's training task AND BERT's pre-training objectives. The resulting embedding is a convex mixture of the two subjects' directions, favoring "bridging" chunks (BERT's introduction, architecture comparisons) over either subject's specific-answer chunks. Q10 (BERT objectives alone) succeeds under all configs; Q15 (Q10 + Transformer task) fails under all configs. |

### Methodological lessons from Day 6

1. **Character-level keyword matching is not a proxy for semantic fact presence.** Numeric keywords in particular produce systematic false positives (Q11: `7` in page number; Q14: `8` in citation `[8]`).

2. **Mechanism explanations require tooling, not intuition.** Pre-inspection hypotheses about Q14 (*"Table 3 row confusion"*) were wrong; the authoritative chunk was never in context at all.

3. **A chunk being correctly split does not imply its embedding represents its facts.** Transformer §3.2.2's key sentence survives chunk boundaries intact (verified via `inspect_chunks.py`) but its embedding is dominated by surrounding LaTeX.

4. **Automated failure classification requires manual verification.** The 2 cases reported by `analyze_results.py` as "Generator failure" (Q11, Q14) were both misclassified; both are Retrieval failures.

5. **An authoritative chunk existing in the index does not imply retrievability.** Q11's chunk `#1831` and Q14's chunk `#27` both exist, both contain exact ground-truth answers, and neither was retrieved by any of {vector k=3, vector k=5, hybrid k=3, hybrid k=5}.

6. **Multiple keyword hits in the same answer can be scattered false positives.** Q12 shows all 3 keywords "present" in chunks, but in 3 different chunks, each in unrelated semantic contexts.

7. **BM25 penalizes cross-paper reference terms.** High-frequency intra-document terms dominate RRF fusion, pushing rare cross-paper references (BART, in the RAG paper) out of the top-k window.

8. **Definitional answers ("X is Y") get semantically crowded by surrounding discussion.** Short, specific answer sentences are weak embedding-match targets when embedded in longer "why/how" discussions.

9. **Compound queries smear both sub-answers.** An A+B query is close to "bridging" chunks (discussing both A and B abstractly) but far from either A-specific or B-specific answer chunks.

Each lesson maps to at least one verified case; see the failure-mode table above for which questions demonstrate which lesson.

### Implications for next experiments

With retrieval identified as the bottleneck for 6/6 failed questions (not generation), optimization effort should concentrate on:

- **Exp F (chunking)**: try `chunk_size=250` with `chunk_overlap=100` to surface short-key-sentence chunks like Transformer §3.2.2 as more distinct semantic units. Expected to directly address lessons 3 and 8.
- **Exp G (query rewriting / HyDE)**: decompose compound queries (Q15) and generate hypothetical answer passages to bridge the vocabulary gap (Q06, where the query does not contain "BERT"). Addresses lessons 8 and 9.
- **Exp H (cross-encoder reranker)**: re-rank top-20 hybrid candidates to filter BM25 noise intrusion (Q07) and surface specific-answer chunks ranked 6–20 by initial retrieval. Addresses lesson 7.
- **Defer generator changes (GPT-4, prompt engineering)**: no verified generator failures in the current eval set mean these changes are unlikely to move the needle on the 6 current failure cases.

### Retrospective on Day 5

The Day 5 Exp D recommendation ("prefer k=5 over hybrid at k=3") was not wrong given the data it had, but it overgeneralized from a specific operating point (small eval set, unfair k comparison) to a mechanism-level conclusion. The lesson is not "hybrid retrieval is unreliable" but "6 questions × unfair control are not enough to characterize a retrieval mechanism." Day 6's 4-config × 15-question matrix produces an interpretable picture: **hybrid's value depends on the context budget (k); at k=5 it beats vector by 1 question (6.7 percentage points) in this eval set**. Larger eval sets are needed to narrow the confidence interval around this effect.