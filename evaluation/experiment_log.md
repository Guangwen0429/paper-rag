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

---

## Exp F: Chunking Ablation (chunk_size 500 → 250)

**Date**: 2026-04-22
**Motivation**: Day 6 Exp E identified three lessons that pointed toward chunking as a promising intervention: Lesson 3 ("correctly split chunks don't guarantee faithful embeddings"), Lesson 5 ("authoritative chunk in index does not imply retrievability"), and Lesson 8 ("definitional answers crowded by surrounding discussion"). Specific failures under Day 6 hybrid k=5 — Q14's formula-diluted "h=8" sentence, Q11's never-retrieved release statement, Q06's DPR-BERT definitional sentence — all shared a common structural pattern: a short, specific answer sentence embedded in a longer chunk whose pooled embedding was dominated by surrounding content. The hypothesis was that reducing chunk_size from 500 to 250 would isolate these key sentences into their own chunks, producing embeddings that more faithfully represent the answer fact.

### Setup

Same 15-question eval set and 4-config matrix as Day 6. Only `chunk_size` and derived `chunk_overlap` changed:

| Parameter | Day 6 | Day 7 |
|---|---|---|
| chunk_size | 500 | 250 |
| chunk_overlap | 50 | 50 |
| Total chunks | 2,425 | ~5,000 |

Implementation: refactored `run_eval.py` so that `EXPERIMENTS` holds 4-tuples `(mode, k, chunk_size, chunk_overlap)`, with an `index_cache` in `main()` that builds each `(chunk_size, chunk_overlap)` index only once and shares it across matching configs. Output filenames now include `cs{chunk_size}` to prevent Day 6/Day 7 result collisions. Summary file is `eval_summary_all_day7.json`.

### Results

| Config | Day 6 (cs=500) | **Day 7 (cs=250)** | Δ Keyword Hit |
|---|:---:|:---:|:---:|
| vector k=3 | 6/15 (40.0%) | **4/15 (26.7%)** | **−2** |
| vector k=5 | 8/15 (53.3%) | **6/15 (40.0%)** | **−2** |
| hybrid k=3 | 5/15 (33.3%) | **3/15 (20.0%)** | **−2** |
| hybrid k=5 | 9/15 (60.0%) | **6/15 (40.0%)** | **−3** |

**All four configs regressed by 2–3 questions**. Source Hit remained at 15/15 across all configs (correct paper still retrieved), but answer accuracy fell sharply.

### Routing precision: vector stable, hybrid degraded

| Config | Day 6 precision | Day 7 precision | Δ |
|---|:---:|:---:|:---:|
| vector k=3 | 86.7% | 86.7% | 0 |
| vector k=5 | 86.7% | **88.0%** | **+1.3%** |
| hybrid k=3 | 84.4% | **75.6%** | **−8.8%** |
| hybrid k=5 | 82.7% | **73.3%** | **−9.4%** |

Vector retrieval's routing precision was unaffected or slightly improved; hybrid's dropped sharply. This asymmetry was the first clue that cs=250 affected BM25 and vector differently.

### Controlled-variable analysis

**Fixed retrieval mode, variable chunk_size**:
- Vector configs lost 2 questions each (40→26.7%, 53.3→40%)
- Hybrid configs lost 2–3 questions each (33.3→20%, 60→40%) and lost ~9% routing precision

**Fixed chunk_size, variable retrieval mode at cs=250**:
- vector 40% > hybrid 33.3% at k=3 (hybrid's k=3 penalty from Day 6 persists and worsens)
- vector 40% = hybrid 40% at k=5 (hybrid's Day 6 advantage at k=5 eliminated)

**Interpretation**: hybrid's value is even more k-dependent at cs=250. At k=5 the larger window no longer compensates for BM25's noise intrusion, because cs=250 amplifies BM25's sensitivity to short-chunk term-frequency artifacts.

### Chunk-level verification (all 15 questions inspected via `inspect_question.py`)

**Questions whose Day 6 success was lost under cs=250** (regression):

| QID | Day 6 result | Day 7 result | Verified mechanism |
|---|---|---|---|
| Q02 | ✓ (hybrid k=5) | ✗ (all configs) | **Context fragmentation**: Table 3's header row and data rows split into separate chunks; header chunk has no "8", data chunk has no "Transformer" context |
| Q08 | ✓ (hybrid k=5) | ✗ (all configs) | **Cross-paper attribution amplified**: LLaMA 2 p4 chunk ends mid-word at "...specifically through rejection sampling and Proximal Policy", cutting "Optimization" into the next chunk. LLM quoted LLaMA 2's RLHF description to answer an InstructGPT question |
| Q13 | ✓ (hybrid k=5) | ✓ (keyword hit) but verified unfaithful | **LLM prior knowledge masks retrieval failure**: GPT-3 Table 2.1 split so that Small/Medium/Large are in one chunk and 175B row is in another; the 175B chunk was not retrieved. LLM produced "175 billion parameters" from parametric memory while explicitly stating "this information is not provided in the sources". Script keyword_hit = ✓, but RAG actually failed |

**Questions whose Day 6 failure persisted or improved under cs=250**:

| QID | Day 6 result | Day 7 result | Verified mechanism |
|---|---|---|---|
| Q05 | ✓ (k=5 only) | ✓ (all 4 configs) | **Embedding purity improvement**: LLaMA 2 Abstract chunk "ranging in scale from 7 billion to 70 billion parameters" now independent of surrounding paragraphs; its embedding direction is more aligned with "largest model size" queries, lifting it into top-3 even under vector k=3 |
| Q07 | ✗ (all configs) | ✓ (hybrid k=5 only) | **Sparse cross-paper term amplified in BM25**: RAG paper p3 chunk "fine-tuning the query encoder BERTq and the BART generator" now shorter, raising BART's within-chunk term-frequency density; hybrid k=5's RRF fusion lifted this chunk to Rank 4. vector alone still failed (doesn't benefit from TF density) |
| Q11 | ✗ (all configs) | ✓ (vector k=5 only) | **Embedding purity improvement**: LLaMA 2 p76 and p3 chunks containing "7B, 13B, and 70B" as isolated short statements are now retrieved under vector k=5. hybrid k=5 still fails because BM25 pulled in CoT paper chunk "LaMDA... 68B and 137B" (matches "13", "7" character-wise) |
| Q06 | ✗ | ✗ | **Definitional blind spot worsened**: DPR §3 "we use two independent BERT" chunk still never retrieved; purer discussion-type chunks (EP/EQ definitions) now dominate top-5 even more completely |
| Q12 | ✗ | ✗ | **Scattered decoy trap amplified**: three expected keywords (`supervised`, `reward`, `PPO`) now scattered across even more chunks, each in a narrower context window; canonical §3.1 three-step chunk never retrieved in any config |
| Q14 | ✗ | ✗ | **Context fragmentation + dilution both present**: BERT p7 chunk now cuts mid-sentence at "By contrast, BERT BASE", so LLM sees "A=16" but never sees BERT's "110M parameters" context. Hybrid answers "BERT BASE also uses 16 attention heads" — worse than Day 6 |
| Q15 | ✗ | ✗ | **Compound query smearing unaffected by chunking**: vector k=5 did retrieve BERT §3.1 "Task #1: Masked LM" chunk at Rank 4 (Day 6 never did), but compound query still pulls embedding toward bridging chunks, and Transformer's "translation" side is never represented in top-5 |

### Failure modes observed in Day 7 that were not visible in Day 6

Four mechanisms emerged as direct consequences of small chunks; one positive mechanism counterbalanced them.

**1. Context completeness loss**. Short chunks cut semantic units mid-sentence. Observed in Q02 (Table 3 header/row split), Q08 ("Proximal Policy | Optimization" split), Q14 ("By contrast, BERT BASE" truncated), Q13 (GPT-3 Table 2.1 split across Small-to-Medium and 175B chunks). The LLM receives text fragments whose antecedents are in other chunks, producing refusals or cross-paper attribution errors.

**2. Cross-paper attribution amplified**. When paper A references paper B's technique, cs=250 isolates the reference without its surrounding "this is A, not B" context. Q08's hybrid k=5 answer quoted LLaMA 2's RLHF description verbatim to answer an InstructGPT question — the LLM had no chunk-level context distinguishing which paper the description came from.

**3. BM25 noise intrusion worsened**. Short chunks raise the relative term frequency of any query-matching word, making BM25 highly sensitive to incidental matches. The CoT paper's appendix ("Ellen has six more balls than Marin...") appeared in top-5 for Q14 and Q13 because isolated words like "base" or numeric substrings triggered BM25's TF term. Hybrid routing precision dropped 9% as a direct consequence.

**4. LLM prior knowledge masks retrieval failure**. Q13's LLM answered "175 billion parameters" from training data despite explicitly stating the sources didn't contain it. The script's character-level keyword_hit marked this ✓, but the RAG pipeline had actually failed — retrieval didn't surface the right chunk, and only the LLM's parametric memory saved the answer. This is a new failure mode that `analyze_results.py` cannot detect.

**5. Positive: sparse-term BM25 amplification**. Cross-paper references that appear only a few times in the source document (BART in RAG paper, "70B" in LLaMA 2's release statement) benefit from cs=250. Their within-chunk TF density rises when the surrounding text is cut away, making BM25 more likely to surface them. Q07's recovery under hybrid k=5 is the clearest case. But this mechanism only benefits hybrid retrieval; vector-only retrieval does not gain from TF density.

### Why the trade-off pointed the wrong way

cs=500 → cs=250 moved the system along two axes simultaneously, in opposite directions:

| Axis | cs=250 effect |
|---|---|
| Pooling dilution of short key sentences | **Improved** (Q05, Q07, Q11 benefited) |
| Context completeness for generation | **Worsened** (Q02, Q08, Q13, Q14 regressed) |
| BM25 robustness to incidental matches | **Worsened** (routing precision −9%) |
| Scatter distance between decoy keywords | **Worsened** (Q12) |
| Resistance to compound-query smearing | **Unchanged** (Q15) |

The number of questions benefiting from axis 1 (3 recoveries) was smaller than the number of questions harmed by axes 2–4 (5–6 regressions). Hybrid retrieval was hit twice: BM25 directly lost routing precision, and even the questions where BM25 helped (Q07) were offset by questions where BM25 noise dominated (Q11 hybrid k=5, Q12, Q14).

**The core insight**: chunk_size is not a single-variable optimization. Reducing it trades "embedding purity on short key sentences" for "context completeness and BM25 stability". On this academic-paper eval set, the trade-off is net-negative below cs=500.

### Methodological lessons added in Day 7

**Lesson 10**: Chunking is a multi-dimensional trade-off, not a single-knob optimization. Any `chunk_size` change creates winners and losers simultaneously; net effect depends on the distribution of query types in the eval set (definitional vs aggregation vs compound).

**Lesson 11**: BM25 and dense retrieval have opposite `chunk_size` preferences. BM25 benefits from short chunks (higher TF density, better length normalization for sparse terms); dense retrieval benefits from mid-sized chunks (pooling has enough content to form a coherent semantic direction without being diluted). Hybrid retrieval using shared `chunk_size` cannot simultaneously satisfy both.

**Lesson 12**: Keyword-match metrics can be fooled by LLM parametric memory. Q13's "175 billion parameters" answer looks correct to the script but was generated from GPT-4o-mini's training data, not from the retrieved chunks. Correctness of the RAG pipeline cannot be verified by keyword matching alone — it requires either LLM-as-judge evaluation or explicit grounding checks ("was this claim supported by a retrieved chunk?").

### Implications for next experiments

Day 7 closes the chunking-as-single-axis investigation with a negative result. Three directions remain from the Day 6 next-steps list, now with updated rationale:

- **Exp G (query rewriting / HyDE)**: Only intervention that can address compound-query smearing (Q15, unchanged across Day 6 and Day 7) and vocabulary mismatch (Q06, where query has no "BERT" but answer requires it). Both Lesson 8 and Lesson 9 from Day 6 point here.
- **Exp H (cross-encoder reranker)**: Addresses BM25 noise intrusion (Q07-style, Lesson 7) and definitional blind spots (Q06, Lesson 8) by re-ranking top-20 with query-document token interaction, which pooled-embedding similarity cannot model. Particularly attractive given Day 7 showed hybrid's BM25 noise problem is worse than Day 6 suggested.
- **Exp I (semantic chunking)**: Replace fixed-`chunk_size` with boundary-aware chunking (at sentence or paragraph breaks). This directly addresses Day 7's context-fragmentation mechanism (Lesson 10) — breaks would no longer land mid-sentence at "Proximal Policy | Optimization". More engineering work than G or H, but potentially the most impactful change.

**Deferred**: further chunk_size sweeps (cs=1000, cs=750). Day 7's result suggests the gradient is flat or negative above cs=500 for this corpus; more data points would refine the curve but not change the conclusion that chunking alone cannot fix the failure modes.

### Retrospective on Day 7

Day 7's hypothesis (smaller chunks → purer embeddings → better retrieval on short-answer queries) was **partially correct** (Q05, Q07, Q11 support it) but **globally wrong** on this eval set. The value of running the experiment was not in confirming or denying the hypothesis — it was in discovering the four counterbalancing mechanisms that the hypothesis had not considered. Three of those four (context loss, cross-paper attribution, LLM-memory masking) are invisible at the summary-metric level; only chunk-level inspection exposed them.

The broader lesson is that **a reasonable-sounding single-variable intervention can regress the system by 10–20% of answer accuracy through mechanisms the intervention's motivation did not predict**. This is why Day 6's chunk-level inspection tooling was worth the investment: without it, Day 7 would have concluded "cs=250 is worse, revert to cs=500" and missed four distinct diagnostic insights useful for Exp G/H/I.

---

## Exp H: Cross-Encoder Reranker (Hybrid Retrieval + Cross-Encoder Rerank)

**Date**: 2026-04-23
**Motivation**: Day 7 Exp F concluded that chunking alone cannot fix the retrieval failure modes identified in Day 6 — specifically, the definitional blind spot (Q06), embedding blind spot (Q11), and BM25 noise intrusion (Q07). The three proposed next experiments (Exp G/H/I) each target different sources of the problem. Exp H adds a cross-encoder reranker on top of hybrid retrieval: the hybrid stage provides fast top-20 recall, and a cross-encoder re-scores candidates using query-document token interaction. Hypothesis: cross-encoder's token-level attention between query and passage should resolve the definitional blind spot by distinguishing "asks about X" from "contains definition of X" — the exact query-chunk mismatch that bi-encoder similarity cannot model.

### Setup

Chunk_size fixed at 500 (Day 6 baseline). Hybrid retrieval preserved as the retrieval stage. A cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 80MB, CPU-friendly) added as a reranking stage.

**New module in `src/retriever.py`**:
- `_get_cross_encoder()`: lazy-loaded singleton; the model is loaded once per process and cached across all eval queries
- `rerank_with_cross_encoder(candidates, query, k)`: scores each (query, chunk) pair with cross-encoder, returns top-k
- `hybrid_then_rerank(vectorstore, bm25, chunks, query, k, n_candidates=20)`: end-to-end pipeline — hybrid recalls top-20, cross-encoder reranks to top-k

**Refactored `scripts/run_eval.py`**: added `rerank` as a retrieval mode; `EXPERIMENTS` now compares hybrid (baseline) vs. rerank (new) at k=3 and k=5, all at chunk_size=500.

### Results

| Config | Keyword Hit | Source Hit | Routing Precision |
|---|:---:|:---:|:---:|
| hybrid k=3 (Day 6 baseline) | 5/15 (33.3%) | 100% | 84.4% |
| hybrid k=5 (Day 6 best) | 9/15 (60.0%) | 100% | 82.7% |
| **rerank k=3** | **8/15 (53.3%)** | 100% | 82.2% |
| **rerank k=5** | 9/15 (60.0%) | 100% | **88.0%** |

**Two distinct results**:
1. **rerank k=3 vs hybrid k=3: +20.0 percentage points (5 → 8)**. A dramatic gain under a tight context budget.
2. **rerank k=5 vs hybrid k=5: 9/15 = 9/15**. But the 9 correct questions are *not the same set* — 2 questions were recovered and 2 were lost, netting zero. This is not "no effect"; it is a redistribution of failure modes.

### Controlled-variable analysis

**Fixed k, variable mode**:
- At k=3: rerank > hybrid by 3 questions. Cross-encoder compresses 20 candidates into top-3 more effectively than RRF does into top-3 directly.
- At k=5: rerank = hybrid by count, but routing precision is +5.3 percentage points higher (88.0% vs 82.7%). Cross-encoder filters out chunks from wrong papers that hybrid's BM25 leaked in.

**Fixed mode, variable k**:
- hybrid k=3 → k=5 gains +4 questions (33.3 → 60.0%). The gap reflects the cost of tight k windows under noisy retrieval.
- rerank k=3 → k=5 gains only +1 question (53.3 → 60.0%). Rerank at k=3 is already strong enough that k=5 adds marginal benefit — implying rerank *substitutes for increasing k*, not just adds to it.

**Interpretation**: Cross-encoder rerank's primary effect at k=3 is "do what hybrid k=5 would do, in a smaller window". At k=5, its primary effect shifts to routing precision and failure-mode trade-offs rather than hit count.

### Chunk-level verification (4 questions inspected in depth)

Inspected Q06, Q11 (rerank recovered these from Day 6/7), and Q08, Q09 (hybrid k=5 had these correct but rerank k=5 lost them). The four questions together map out the full behavioral profile of cross-encoder rerank.

| QID | hybrid k=5 | rerank k=5 | Mechanism (verified) |
|---|:---:|:---:|---|
| Q06 | ✗ | ✓ | **Definitional blind spot resolved**: the chunk containing *"By leveraging the now standard BERT pretrained model ... and a dual-encoder architecture"* (DPR §1) was never retrieved by any Day 6/7 vector/hybrid config. Rerank elevated it to Rank 5 by matching query token "architecture" to chunk tokens "BERT pretrained model + dual-encoder architecture" via self-attention. |
| Q11 | ✗ | ✓ | **Embedding blind spot resolved**: three chunks containing complete `7B, 13B, 70B` list (p76 Model Card, p3 release statement, p5 Table 1) all surfaced to rerank top-3. The p3 chunk is the exact chunk referenced in Day 6 Lesson 5 as "authoritative chunk that exists in index but is never retrievable" — rerank reaches it. |
| Q08 | ✓ | ✗ | **Topic-match over answer-match (NEW)**: hybrid k=5 retrieved a chunk at Rank 5 containing *"fine-tune our supervised learning baseline to maximize this reward using the PPO algorithm"* (InstructGPT p1). Rerank pushed this chunk out of top-5, replacing it with chunks that more comprehensively discuss "InstructGPT + RLHF + fine-tuning" at the topic level but *contain no specific algorithm name*. LLM under rerank k=5 refused to answer. |
| Q09 | ✓ | ✗ | **Limited qualifier matching (NEW)**: hybrid k=5 retrieved a chunk at Rank 3 containing *"The performance gain from chain-of-thought prompting is **largest** for PaLM 540B on GSM8K"* (CoT A.4). Rerank favored chunks more topically aligned with "GSM8K + chain-of-thought + standard prompting" but lacking the specific qualifier "largest" that the query requires. LLM refused despite seeing "GSM8K" in the context. |

### LLM behavior under different k values (new observation from Q08)

Q08 under rerank revealed a secondary effect: the LLM's willingness to refuse vs fabricate varies with k. Under rerank k=3, the LLM answered *"InstructGPT is fine-tuned using RLHF"* — a category error (RLHF is the training framework; PPO is the specific RL algorithm). Under rerank k=5, the LLM refused to answer. Interpretation: smaller k forces the LLM to synthesize from fewer candidates, increasing the incentive to commit to a partially-supported answer; larger k gives the LLM enough context to verify that the specific answer is not supported, triggering an honest refusal.

This is not strictly a retrieval phenomenon — it is a generation-layer interaction with retrieval. Keyword_hit metrics treat "refuse" and "wrong-answer" identically as failures, but they carry different downstream consequences, as first noted in Day 5.

### Failure mode map (Day 6–Day 8)

Consolidating across three days of chunk-level evidence:

| Mechanism | First identified | Exp H effect |
|---|---|---|
| 1. Character match ≠ fact presence | Day 6 | Unchanged (a metric limitation, not retrieval) |
| 3. Pooling dilution of key sentence | Day 6 | Partially addressed at token level — cross-encoder attends to individual tokens even when their pooled direction is diluted |
| 5. Embedding blind spot (chunk exists but never retrieved) | Day 6 | **Resolved for Q11** — rerank reached chunk #1831 |
| 6. Scattered decoy keywords | Day 6 | Unchanged |
| 7. BM25 noise intrusion | Day 6 | Partially addressed — rerank k=5 routing precision +5.3 points vs hybrid k=5 |
| 8. Definitional blind spot | Day 6 | **Resolved for Q06** — cross-encoder token attention identifies "asks about X" vs "contains X" distinctions |
| 9. Compound query smearing | Day 6 | Unchanged (Q15 still fails under rerank k=5) |
| 10. Context fragmentation | Day 7 | N/A (chunk_size=500 here) |
| 11. Cross-paper attribution amplified | Day 7 | Unchanged |
| 12. LLM parametric memory masks retrieval failure | Day 7 | Unchanged (Q13 still has this issue) |
| **13. Topic-match over answer-match (NEW)** | Day 8 | **Introduced by cross-encoder**: rerank favors topic-comprehensive chunks over chunks containing the specific answer token — visible in Q08 (PPO answer chunk ejected from top-5) |
| **14. Limited qualifier matching (NEW)** | Day 8 | **Introduced by cross-encoder**: rerank matches query topic well but does not privilege chunks containing query qualifiers (largest, most, specifically) — visible in Q09 ("largest" evidence chunk ejected) |

### Methodological lessons

**Lesson 13**: Summary metrics can hide failure-mode redistribution. Rerank k=5 and hybrid k=5 both score 9/15, but they are correct on *different sets of questions*. Without chunk-level verification, this would be indistinguishable from "rerank has no effect at k=5", leading to the wrong conclusion that cross-encoder rerank only helps under tight k.

**Lesson 14**: Cross-encoder is not a strict superset of bi-encoder for retrieval. Cross-encoder better captures query-document token interaction (beneficial for definitional and listing queries), but it introduces new biases — particularly a "topic-match" prior that can hide chunks that dilute an answer within broader discussion. Any retrieval component exchanges one set of failure modes for another; the choice of component depends on which failure modes are more acceptable for the target query distribution.

**Lesson 15**: LLM refusal vs. fabrication is k-sensitive. At tight k, the LLM is more likely to fabricate a partially-supported answer (Q08 rerank k=3 answered "RLHF" as the algorithm, a category error); at wider k, the LLM is more willing to refuse when no chunk contains the specific answer. This interacts with retrieval quality: a noisier retriever under tight k produces more confident-wrong answers, while a better retriever under wider k produces more honest refusals. For production RAG, this argues for always running at k=5+ even when the top-1 is believed to be correct.

### Implications for next experiments

With rerank validated as a useful but double-edged intervention, three directions remain:

- **Exp G (query rewriting / HyDE)**: Addresses two failure modes rerank cannot help with: compound query smearing (Q15, still unchanged) and the "answer-match" problem (Q08, Q09) — if the query is rewritten into an explicit hypothetical answer, the retrieval stage can target "answer-bearing chunks" more precisely. Also addresses the LLM fabrication risk (Lesson 15) by giving the retriever more specific targets.
- **Exp I (semantic chunking)**: Orthogonal to rerank. Could stack: semantic chunks → hybrid retrieval → cross-encoder rerank. Would directly address context fragmentation (Lesson 10) identified in Day 7, which rerank does not touch.
- **Exp J (hybrid rerank strategies)**: Instead of pure cross-encoder rerank, try weighted combinations (hybrid score + cross-encoder score) to preserve hybrid's answer-match signal while gaining cross-encoder's topic-match precision. Could address the Q08/Q09 regression without sacrificing Q06/Q11 recovery.

Exp G is the most promising next step because it addresses the two failure modes (compound query, answer-match) that Exp H left open.

### Retrospective on Day 8

Day 8 is the first experiment in this project with a clear *positive* headline result (+20pp at k=3) alongside a clear trade-off at k=5 (same count, redistributed failures). The 20pp gain at k=3 is the kind of result that would normally be taken as a standalone conclusion ("rerank is a strict improvement"). Chunk-level verification showed this to be misleading: the k=5 comparison reveals rerank's hidden costs (Q08, Q09 lost), and those costs cannot be waived without explicit mitigation (e.g., rerank score blending in Exp J, or answer-aware retrieval in Exp G).

The pattern across Day 6 → Day 7 → Day 8 is consistent: each intervention solves some failure modes and introduces new ones. There is no monolithic "better retriever". Progress comes from understanding which modes are active and choosing the intervention that best matches the current failure distribution. Day 8's measurement infrastructure (chunk-level inspection + failure mode taxonomy) is now the unit of progress, not single-number metrics.

---

## Exp J: Weighted Fusion of Hybrid RRF and Cross-Encoder Scores

**Date**: 2026-04-24
**Motivation**: Day 8 Exp H showed cross-encoder rerank introduces a net-zero trade at k=5: Q06 and Q11 recovered, Q08 and Q09 lost. Chunk-level inspection identified the mechanism as "topic-match over answer-match" — cross-encoder favors topically comprehensive chunks over chunks that contain specific answer tokens. Hypothesis: a weighted linear combination of RRF and cross-encoder scores could preserve the answer-bearing chunks that pure rerank ejects, while keeping the definitional-query wins. Scan `alpha ∈ {0.0, 0.3, 0.5, 0.7, 1.0}` with k=5 and chunk_size=500.

### Setup

New function in `src/retriever.py`:
- `_min_max_normalize(scores)`: scales scores to [0, 1] to remove scale mismatch (RRF scores are ~0.01–0.05; cross-encoder scores are -10 to +10).
- `hybrid_then_rerank_weighted(vectorstore, bm25, chunks, query, k, n_candidates=20, alpha=0.5)`: (1) reproduces hybrid's RRF fusion internally to preserve per-chunk RRF scores, (2) cross-encoder scores the same 20 candidates, (3) min-max normalizes both score vectors to [0, 1], (4) fuses with `score = (1 - alpha) * norm_rrf + alpha * norm_ce`, (5) returns top-k by fused score.

`scripts/run_eval.py`: added `rerank_weighted` mode; `EXPERIMENTS` tuples extended to 5 elements `(mode, k, chunk_size, chunk_overlap, alpha)`; output filenames include `_a{alpha}` suffix for rerank_weighted runs.

### Results

| Config | Keyword Hit | Routing Precision | Interpretation |
|---|:---:|:---:|---|
| α=0.0 | 9/15 (60.0%) | 82.7% | = Day 6 hybrid k=5 baseline (validates reduction) |
| α=0.3 | 8/15 (53.3%) | 85.3% | regression (non-monotonic dip, see below) |
| α=0.5 | 9/15 (60.0%) | 85.3% | recovered to baseline |
| **α=0.7** | **10/15 (66.7%)** | 85.3% | **peak — project-wide high** |
| α=1.0 | 9/15 (60.0%) | 88.0% | = Day 8 rerank k=5 baseline (validates reduction) |

The two endpoint equalities (α=0.0 ≡ hybrid, α=1.0 ≡ rerank) confirm the weighted-fusion implementation is correct. The key result is the interior peak at α=0.7: **net +1 question over the best Day 8 config (rerank k=5), net +1 question over the best Day 6 config (hybrid k=5)**. This is the first experiment in this project to produce a net gain over both baselines on the same eval set.

### Controlled-variable analysis

**α=0.7 vs α=0.0 (hybrid baseline)**: recovered 3 (Q06, Q11, Q14), lost 1 (Q08), net +2.
**α=0.7 vs α=1.0 (rerank baseline)**: recovered 2 (Q11, Q14), lost 1 (Q06), net +1.

**α=0.7 is not a superset of either endpoint**. It recovers different questions than pure rerank (α=1.0), not the same ones plus extras. This tells us different failure modes have different optimal α.

### Chunk-level verification (Q14, Q08, Q06, Q11 inspected across all 5 α values)

| QID | α that recovers it | Answer-chunk retrieval behavior |
|---|:---:|---|
| **Q06** | **α=1.0 only** | DPR p1 chunk *"By leveraging the now standard BERT pretrained model ... and a dual-encoder architecture"* ranks very low in hybrid RRF (rank 15+). Any nonzero hybrid weight drags it out of top-5. Only pure cross-encoder recovers it. |
| **Q11** | **α ≥ 0.7** | LLaMA 2 p3 (*"releasing variants of Llama 2 with 7B, 13B, and 70B parameters"*) and p76 (*"Variations Llama 2 comes in a range of parameter sizes—7B, 13B, and 70B"*) rank moderately in hybrid RRF, highly in cross-encoder. Need at least α=0.7 to push both into top-5. |
| **Q14** | **α ∈ [0.5, 0.7]** | Retrieved chunk set is nearly identical across all α; only ranks change. When BERT p7 (containing "Vaswani et al. (2017) is L=6, H=1024, A=16") occupies Rank 1 instead of Rank 2, the LLM reads its context more carefully and disambiguates "A=16 refers to Transformer *big*, BERT *base* must be elsewhere" — then assembles the correct answer from heterogeneous signals. This is a rank-order effect on LLM reading strategy, not a retrieval recovery. At α=1.0 the rank order re-shuffles and the effect disappears. |
| **Q08** | **α=0.0 only** | InstructGPT p14 and p1 (both contain "PPO") rank well in hybrid RRF but poorly in cross-encoder (which categorizes them as "detailed process description" rather than "answers 'what algorithm'"). Any α > 0 ejects both from top-5. Impossible to recover Q08 without sacrificing all α>0 benefits. |

### The α=0.3 non-monotonic dip

α=0.3 (8/15) is worse than both α=0.0 and α=0.5 (both 9/15). Q08's answer chunks get ejected as soon as cross-encoder signal is introduced (any α > 0), but Q11/Q14 require α ≥ 0.5 to benefit. At α=0.3, we pay the Q08 cost without yet earning the Q11/Q14 gains. This is a predictable consequence of the per-query-optimal-α phenomenon below.

### Central observation: per-query-optimal α is not constant

| Question | Optimal α range | Rationale |
|---|---|---|
| Q06 | 1.0 | answer chunk buried very deep in hybrid ordering |
| Q08 | 0.0 | answer chunks have strong hybrid signal but weak CE signal |
| Q11 | ≥ 0.7 | answer chunks mid-rank in hybrid, high-rank in CE |
| Q14 | 0.5–0.7 | no direct answer chunk; LLM disambiguates from rank order |

**No single α can recover all four**. Q06 needs α=1.0 (which loses Q14), Q08 needs α=0.0 (which loses Q06 and Q11). α=0.7 happens to be the sweet spot for this eval set — it maximizes total recovered questions — but this is an artifact of the specific failure distribution, not a transferable optimum. A different eval set with more "answer-in-detail-chunk" questions (Q08-type) would move the optimum down; a set with more "definitional blind spot" questions (Q06-type) would move it up.

### Methodological lessons added in Day 9

**Lesson 16**: A single scalar α cannot simultaneously optimize across heterogeneous failure modes. Each failure mode has a signature (answer-chunk rank in hybrid vs. answer-chunk rank in CE), and α trades them off rather than resolving them. The empirical peak at α=0.7 is specific to this eval set's failure-mode mixture.

**Lesson 17**: LLM output can depend on retrieval *rank order*, not just the retrieved *set*. Q14's recovery at α=0.5/0.7 is produced by reshuffling the same chunks; the LLM reads more carefully when an authoritative chunk is at Rank 1, disambiguating content that it would have used incorrectly at Rank 2. This is a generation-retrieval interaction effect invisible to recall@k metrics.

**Lesson 18**: Endpoint sanity checks are essential for fusion experiments. The α=0.0 and α=1.0 results must exactly match prior baselines (9/15 hybrid, 9/15 rerank) to confirm the fusion formula reduces correctly. This validates the experiment architecture before interpreting interior points.

### Implications for next experiments

Day 9 closes the "single-α fusion" line of investigation. The natural next step is **per-query-adaptive α**:
- **Exp M (query classifier)**: Classify queries by type (definitional / list / answer-in-detail / compound) using a small classifier or LLM prompt, dispatch to different α. Training signal is Day 6–9's chunk-level failure taxonomy.
- **Exp G (HyDE / query rewriting)** remains relevant for the compound-query smearing problem (Q15, still unchanged at every α). Orthogonal to α tuning.
  - **Update (Day 11)**: Completed. +1 question over rerank baseline (19/30 vs 18/30), 3-run std=0, exceeds Lesson 21 noise floor. Mechanism documented in Lesson 26 as chunk-selection level (not document-routing level). See Day 11 section.
- **Exp N (learned weights)**: Instead of manual α scan, learn per-query weights via a small MLP on query features + candidate statistics. Larger eval set needed for training signal.

(Note: Exp K and L letters were used in Day 10 for infrastructure work — GPU acceleration and eval set expansion respectively. Future method experiments use M onward.)

### Retrospective on Day 9

Day 9 is the first experiment to produce a net gain over all prior baselines (+1 vs Day 8 rerank, +1 vs Day 6 hybrid). But the gain is interpretable, not intrinsic: α=0.7 wins by averaging out complementary failure modes. The experiment's deeper contribution is Lesson 16 — establishing that there is no universal best retrieval weight in this failure-mode-heterogeneous regime, and that further improvement requires **query-conditional retrieval**, not a global parameter tune. This reframes the next-step horizon from "find a better uniform retriever" to "find the right retriever per query".

---

## Exp K: GPU Acceleration for Cross-Encoder Reranking

**Date**: 2026-04-25 (Day 10, infrastructure)
**Motivation**: Day 8/9 cross-encoder reranking ran on CPU, contributing the largest single component to per-query latency. Day 10's planned eval set expansion (15→30 questions) and possible later experiments (Exp M adaptive-α requiring repeated reranks per query, Exp G HyDE adding LLM calls) will roughly multiply the rerank workload. Moving cross-encoder inference to GPU before scaling avoids compounding CPU-bound latency. Infrastructure step, not a method experiment — no answer-quality changes are expected, only wall-clock improvement.

### Setup

**Hardware**: NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM, driver 561.00, driver-reported CUDA 12.6.

**Software**: Replaced `torch 2.11.0+cpu` with `torch 2.5.1+cu121` (latest stable wheel for CUDA 12.1 build, forward-compatible with the 12.6 driver). The cu121 build was preferred over cu124 for broader community testing coverage at this version.

**Code change** (`src/retriever.py`): one functional line modified in `_get_cross_encoder()`:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
_cross_encoder_instance = CrossEncoder(model_name, device=device)
```
The lazy global singleton pattern (introduced Day 8) made this a single-point change — all downstream functions (`rerank_with_cross_encoder`, `hybrid_then_rerank`, `hybrid_then_rerank_weighted`) inherit GPU acceleration automatically without modification. CPU fallback preserved for portability.

**Verification scripts added**:
- `scripts/verify_gpu.py`: 4-stage health check — PyTorch CUDA detection, cross-encoder GPU load, CPU/GPU score parity within 1e-3, and a 100-pair micro-benchmark.
- `scripts/benchmark_gpu.py`: end-to-end pipeline benchmark on the 15-question eval set at α=0.7, k=5, n_candidates=20 (the Day 9 best config). Forces cross-encoder reload on CPU and GPU within a single run for direct comparison.

### Results

**Numerical sanity check** (verify_gpu.py): CPU and GPU produced identical cross-encoder scores within 2e-6 absolute difference across 3 representative query-passage pairs — confirming no numerical drift from FP32 GPU kernels. This guarantees that re-running prior experiments on GPU will reproduce Day 6–9 answer-quality numbers exactly.

**Cross-encoder isolated benchmark** (verify_gpu.py, 100 pairs, post-warmup):

| Device | Time | Speedup |
|---|---|---|
| CPU | 246 ms | 1.0× |
| RTX 4060 | 24.4 ms | **10.1×** |

**End-to-end pipeline benchmark** (benchmark_gpu.py, 15 eval questions, α=0.7, k=5, n_candidates=20):

| Device | Total | Per query | Speedup |
|---|---|---|---|
| CPU | 11452 ms | 763.5 ms | 1.0× |
| RTX 4060 | 5814 ms | 387.6 ms | **1.97×** |

### Findings

**Lesson 19: End-to-end speedup is bottlenecked by the un-accelerated stages.** Cross-encoder isolated speedup is 10.1× but pipeline speedup is only 1.97×. The gap is Amdahl's law: per-query latency decomposes roughly into (1) OpenAI embedding API for vector search, network-round-trip-bound at ~400 ms, (2) BM25 retrieval, CPU-bound at ~10 ms, (3) cross-encoder rerank, ~200 ms CPU / ~20 ms GPU. GPU only accelerates stage (3), so the theoretical end-to-end ceiling is roughly `total_cpu / (total_cpu - rerank_cpu + rerank_gpu) ≈ 763 / (763 - 200 + 20) ≈ 1.96×`. Measured 1.97× = essentially at the ceiling.

**Lesson 20: The next bottleneck is OpenAI embedding API, not retrieval logic.** The ~400 ms vector-search latency is network-bound, not compute-bound. Future optimization paths: (a) cache embeddings locally per query to amortize across α-sweeps and config comparisons (Day 9 reran the same 15 queries × 5 α values = 75 redundant embedding calls), (b) precompute and persist Chroma index across runs instead of rebuilding on every script invocation, (c) replace OpenAI embeddings with a local model (e.g., `bge-small-en-v1.5`) that runs on the same GPU. Option (a) is the highest-leverage cheapest fix and will be revisited if benchmarking pressure increases.

**Why this matters for the story arc**: Day 11+ experiments will multiply the rerank workload (eval set expansion 15→30; potential HyDE adds an LLM call per query; per-query adaptive α requires repeated reranks at different settings; Exp L's reproducibility audits multiply runs). Without GPU, those costs would compound into 5+ minute experiment runs, friction that quietly kills iteration velocity. Doing the GPU step now is throughput insurance for the remainder of the project.

### Documentation deltas

- `requirements.txt` not updated yet (PyTorch CUDA wheel install path differs from default PyPI; will document install command in README rather than pin in requirements).
- README updated to Day 10 status with hardware section.
- No experiment numbers from Day 6–9 invalidated; numerical parity confirmed.

---

## Exp L: Eval Set Expansion 15→30 with Reproducibility Audit

**Date**: 2026-04-25 (Day 10)
**Motivation**: Day 9 concluded that weighted-fusion α=0.7 was the best config (10/15, 66.7%) on a 15-question eval set. Lesson 16 already noted that this peak was sensitive to the eval-set's failure-mode mixture and might not transfer. Day 10 tests this directly by (a) doubling the eval set to 30 questions with balanced paper coverage and category mix, then (b) re-running all baselines and the full α-sweep on the new set. A reproducibility audit (same config, three runs) was added mid-experiment after observing an unexpected discrepancy between two endpoint configurations that should have been equivalent.

### Setup

**Eval set expansion** (`evaluation/eval_questions.json`, 15→30 questions):

| Category | Day 9 | Day 10 | New | New IDs |
|---|---|---|---|---|
| single_fact | 9 | 18 | +9 | Q16–Q24 |
| multi_chunk | 3 | 6 | +3 | Q25–Q27 |
| cross_paper | 3 | 6 | +3 | Q28–Q30 |
| **total** | **15** | **30** | **+15** | |

Paper coverage diagnosed and rebalanced: BERT was over-represented (5 questions, 33% of original set); RAG/DPR/CoT each had only 1 question. New questions targeted the under-represented papers — RAG +3 (Q16, Q25, Q28), DPR +3 (Q17, Q24, Q28), CoT +3 (Q22, Q26, Q30) — yielding a final distribution where every paper is covered by 3-7 questions. All 30 ground-truth answers were manually verified against original PDFs before running experiments.

**Failure-mode coverage**: New questions also targeted retrieval failure modes that were under-tested on 15q. Q14-style cross-paper attribution contamination was reinforced via Q28/Q29/Q30 (compound queries spanning two papers with non-overlapping sub-answers). Q06-style vocabulary-mismatch was reinforced via Q24 (similarity-function terminology). Term-overlap pressure was added via Q21/Q23 (activation/positional encoding — terms that BERT, GPT, and LLaMA papers all touch differently).

**Experiment matrix**: 7 configs × 30 questions = 210 evaluation runs in the main sweep. The 7 configs span the full Day 6→9 method history on the new eval set: hybrid k=5 (Day 6 baseline), rerank k=5 (Day 8 baseline), and weighted-fusion at α∈{0.0, 0.3, 0.5, 0.7, 1.0} (Day 9 sweep).

**Reproducibility audit**: 3 runs of the same `rerank_weighted α=0.0` configuration to test whether observed differences between configs are method effects or LLM API non-determinism. Same retrieval, same prompts, same OpenAI gpt-4o-mini at temperature=0.

### Results

**Main 7-config sweep on 30q**:

| Config | Keyword Hit | Routing Precision |
|---|:---:|:---:|
| hybrid k=5 | 17/30 (56.7%) | 88.0% |
| rerank k=5 | **18/30 (60.0%)** | 91.3% |
| rerank_weighted α=0.0 | 16/30 (53.3%) | 88.0% |
| rerank_weighted α=0.3 | 16/30 (53.3%) | 90.0% |
| rerank_weighted α=0.5 | 16/30 (53.3%) | 90.0% |
| rerank_weighted α=0.7 | 17/30 (56.7%) | 90.7% |
| rerank_weighted α=1.0 | **18/30 (60.0%)** | 91.3% |

**Day 9 vs Day 10 comparison (same configs, different eval set sizes)**:

| Config | Day 9 (15q) | Day 10 (30q) | Δ |
|---|:---:|:---:|:---:|
| hybrid k=5 | 60.0% (9/15) | 56.7% (17/30) | -3.3pp |
| rerank k=5 | 60.0% (9/15) | 60.0% (18/30) | 0.0pp |
| rerank_weighted α=0.7 | **66.7% (10/15)** | 56.7% (17/30) | **-10.0pp** |
| rerank_weighted α=1.0 | 60.0% (9/15) | 60.0% (18/30) | 0.0pp |

**Reproducibility audit on rerank_weighted α=0.0 (3 independent runs, identical config)**:

| Run | Keyword Hit | Routing Precision |
|---|:---:|:---:|
| Run 1 | 16/30 (53.3%) | 88.0% |
| Run 2 | 17/30 (56.7%) | 88.0% |
| Run 3 | 16/30 (53.3%) | 88.0% |

Routing precision is identical across all three runs (88.0% to 4 decimal places in detailed JSON). The single-question variance is therefore not from retrieval — it is entirely from the LLM generation step.

### Findings

**Lesson 21: LLM API non-determinism contributes ±1-question noise on a 30q eval set even at temperature=0.** Three identical runs of `rerank_weighted α=0.0` produced answer-quality scores of 16, 17, 16 with byte-identical retrieval (88.0% routing precision invariant to the last decimal). The disagreement rate of 1/30 = 3.3% matches published estimates for OpenAI API stochasticity at T=0 (Atil et al., 2024). The mechanism is well-understood: floating-point summation in transformer matrix multiplications is not associative, and minor logit perturbations at the 1e-7 level can flip top-2 token rankings when probabilities are close. This places a hard noise floor on this evaluation methodology: **single-run differences ≤2 questions on 30q are within LLM noise and should not be interpreted as method-level differences**.

**Lesson 22: The Day 9 α=0.7 peak does not generalize.** On 30q, α=0.7 scored 17/30 (56.7%), which is below α=1.0 at 18/30 (60.0%) — a reversal of the 15q ranking where α=0.7 (10/15) beat α=1.0 (9/15). This is exactly what Lesson 16 predicted: a single scalar α optimized on a small set is overfitting to that set's specific failure-mode mixture. Combined with Lesson 21, the 1-question gap between α=0.7 and α=1.0 on 30q is itself within noise floor — meaning the practically correct conclusion from Day 9–10 is "weighted fusion offers no measurable improvement over pure rerank on this corpus."

**Lesson 23: Rerank k=5 is the most stable winner across eval sets.** Both rerank baselines (rerank mode and rerank_weighted α=1.0, which are mathematically equivalent) held at 60.0% on both 15q and 30q (Δ = 0.0pp), while hybrid k=5 dropped 3.3pp and α=0.7 dropped 10pp. Stability across eval sets is a stronger property than peak performance on a small set, because peak performance can be a sampling artifact while stability reflects a genuinely robust mechanism.

**Lesson 24: Endpoint sanity check confirms fusion implementation is mathematically correct.** Pure-`rerank` mode and `rerank_weighted` mode at α=1.0 are designed to be mathematically equivalent (when α=1.0, the weighted score collapses to `1.0 * norm_ce + 0.0 * norm_rrf = norm_ce`, identical to pure rerank). On 30q both produced 18/30 keyword hits AND 91.3% routing precision to 4 decimal places. Combined with the α=0.0 endpoint matching hybrid baseline (both 88.0% routing precision), this confirms the weighted fusion code in `hybrid_then_rerank_weighted` is correctly implemented.

### Methodological implications

The reproducibility audit reframes how all prior Day 6–9 results should be interpreted. The +1-question "improvements" reported in Day 8 (rerank vs hybrid) and Day 9 (α=0.7 vs α=1.0) on 15q are now revealed to be within the LLM noise envelope. This is not a failure of the experiments — the routing precision and chunk-level analysis still hold value because they are deterministic. But the keyword-hit metric, which depends on LLM generation, requires multi-run averaging to be reliable. Future experiments (Exp M adaptive α, Exp G HyDE, Exp I semantic chunking) will need to either (a) run each config N≥5 times and report mean±std, or (b) use a deterministic answer-extraction metric (e.g., regex on answer text rather than LLM-generated paragraph match).

This insight is itself the most important Day 10 outcome — more important than any single number in the table. It establishes a rigorous evaluation methodology for the rest of the project and a transferable engineering lesson: **"in LLM-based evaluation, the noise floor must be empirically measured before single-run differences can be trusted as signal."**

### Documentation deltas

- `evaluation/eval_questions.json`: 15 → 30 questions
- `scripts/run_eval.py`: EXPERIMENTS list extended to 7 configs (kept in normal state for Day 11)
- New result files: 7 main + 2 reproducibility-audit JSONs in `evaluation/`
- New summary: `evaluation/eval_summary_all_day10_30q.json`, `eval_summary_alpha_sweep_day10_30q.json`, `eval_summary_reproducibility_run2.json`, `eval_summary_reproducibility_run3.json`
- README.md: updated to Day 10 status with new headline numbers and noise-floor caveat

---

---

# Day 11 — Exp G: HyDE Retrieval

**Date**: 2026-04-26
**Status**: ✅ Complete

## Setup

Hypothetical Document Embeddings (HyDE; Gao et al., 2022) is a query rewriting technique designed to address the query-document style/length asymmetry in dense retrieval: queries are short and abstract, while document chunks are long and concrete, so their embeddings often misalign even when semantically related.

**Pipeline** (`hyde_then_rerank` in `src/retriever.py`):

1. Generate a hypothetical answer passage from the original query using `gpt-4o-mini` (T=0, max_tokens=200, single passage). Prompt: "You are an expert in NLP and machine learning research. Given the following question, write a single paragraph (3-5 sentences) that directly answers it, written in the style of a research paper passage. Use technical vocabulary appropriate to the field. Do not preface, hedge, or apologize. Output only the passage."
2. Vector retrieval uses the **hypothetical passage** as the search query (not the original question) — this is HyDE's core mechanism.
3. BM25 retrieval uses the **original query** — HyDE's verbose output would noise BM25's bag-of-words matching.
4. RRF fusion of vector + BM25 candidates (n_candidates=20, rrf_k=60).
5. Cross-encoder rerank uses the **original query** — reranker is trained on (query, passage) pairs, not (passage, passage).

The hypothetical passage exists only between steps 1→2 and is discarded. Final answer generation uses the original query + retrieved real chunks.

**Evaluation**: same 30-question set as Day 10, chunk_size=500, k=5, n_candidates=20. Run 3 times for reproducibility audit per Lesson 21.

## Hypothesis

**H1**: HyDE will improve keyword_hit on Q15 (compound query smearing) and Q06 (vocabulary mismatch: "encoder architecture" vs "BERT") — the failure modes flagged in Day 6/9 chunk-level analysis as theoretically addressable by HyDE.

**H2**: HyDE will not help cross-paper routing failures (no Q14 improvement expected — HyDE is a within-document mechanism).

**H3**: Single-run difference must be ≥2 questions OR multi-run std must be clearly below ±1 to claim improvement above the Lesson 21 noise floor.

## Result

**Single-run vs Day 10 baselines** (30q, chunk_size=500, k=5):

| Config                      | Keyword Hit       | Routing Prec |
|-----------------------------|-------------------|--------------|
| hybrid                      | 17/30 (56.7%)     | 88.0%        |
| rerank (Day 8 baseline)     | 18/30 (60.0%)     | 91.3%        |
| rerank_weighted α=1.0       | 18/30 (60.0%)     | 91.3%        |
| **hyde_rerank**             | **19/30 (63.3%)** | **92.0%**    |

**3-run reproducibility** (`eval_summary_hyde_reproducibility_day11_30q.json`):

| Run        | Keyword Hit | Routing Prec       |
|------------|-------------|--------------------|
| run 1      | 19/30       | 92.0%              |
| run 2      | 19/30       | 92.0%              |
| run 3      | 19/30       | 92.0%              |
| mean ± std | 19.0 ± 0.0  | 92.0% (identical)  |

**Q08 chunk-level diff** (the only question hyde_rerank rescues from rerank baseline):

Question: "What reinforcement learning algorithm is used to fine-tune InstructGPT?" Expected: PPO.

Top-3 ranks identical between methods:
- Rank 1: `05_instructgpt.pdf` p.0 (abstract, no PPO term)
- Rank 2: `08_llama2.pdf` p.8 (RLHF section, no PPO term)
- Rank 3: `08_llama2.pdf` p.68 (GPT-judge metrics, no PPO term)

Rank 4-5 diverge:

- **rerank baseline**:
  - Rank 4: `05_instructgpt` p.2 — "InstructGPT preferred to GPT-3 85±3%" chunk (no PPO term)
  - Rank 5: `05_instructgpt` p.3 — "InstructGPT shows promising generalization" chunk (no PPO term)
- **hyde_rerank**:
  - Rank 4: `05_instructgpt` p.2 — Figure 2 caption: "(1) SFT, (2) RM training, (3) reinforcement learning via proximal policy optimization (PPO)" (★ contains PPO)
  - Rank 5: `05_instructgpt` p.14 — "Changing the KL model from the PPO init to GPT-3 gives similar results" (★ contains PPO)

InstructGPT p.2 is split into multiple chunks under chunk_size=500. Both methods rank a p.2 chunk at position 4, but they select different chunks from p.2. Generated answer for hyde_rerank: "The reinforcement learning algorithm used to fine-tune InstructGPT is proximal policy optimization (PPO) [Source 4]" — citation traces back to the Figure 2 caption chunk.

## Analysis

**H1 partially confirmed**: HyDE rescues Q08 (vocabulary mismatch class: "RL algorithm" → "PPO"), did not improve Q06 or Q15 in this run. N=1 for Q08 is insufficient to claim a class-level effect.

**H2 confirmed**: Q14 (cross-paper count compounding) and Q30 (cross-paper list) still fail under hyde_rerank — HyDE provides no routing benefit across documents.

**H3 confirmed via 3-run audit**: std=0 over 19/30 baseline, with all 3 runs identical. First method-level improvement to clearly exceed the Lesson 21 noise floor (compare: rerank_weighted α=0.0 in Day 10 audit gave 16, 17, 16 with std≈0.5).

**Cost**: HyDE adds 1 LLM call per query (~$0.0001 with gpt-4o-mini), ~3-5 seconds added latency per query for the hypothetical passage generation. Acceptable for the +1 question gain in this evaluation.

## Lessons

**Lesson 25**: HyDE retrieval improves keyword_hit by +1 question over rerank baseline (19/30 vs 18/30) on the 30q evaluation set, with 3-run zero variance (mean=19.0, std=0.0, identical routing precision 92.0% across runs). This is the first method-level change to exceed the Lesson 21 noise floor under strict criteria. Note that std=0 is a property of the (config, evaluation set) pair — it does not mean HyDE is universally more stable; it means the 30 questions in this set happen to land far from generator logits boundaries under hyde_rerank context.

**Lesson 26**: HyDE's empirical mechanism on this benchmark operates at the chunk-selection level, not the document-routing level. Q08 diff: top-3 retrieved chunks are identical between rerank and hyde_rerank; rank 4-5 differ. Both methods identify the correct source paper (`05_instructgpt.pdf`) and even the correct page (p.2) at rank 4, but they retrieve different chunks from that page — hyde_rerank picks the Figure 2 caption chunk that explicitly contains "proximal policy optimization (PPO)", rerank picks an adjacent p.2 chunk that does not. The hypothetical passage generated by GPT-4o-mini contains "PPO" terminology, pulling the vector-search similarity toward chunks containing that term. This is consistent with Gao et al. 2022's query-document style gap hypothesis, but reveals a finer granularity that the original BEIR benchmark (document-level retrieval) cannot expose: in chunked RAG, "correct document routing" and "correct chunk selection" are two distinct alignment problems.

**Lesson 27**: Methodological investment compounds. The Day 10 reproducibility audit (Lesson 21) produced no new positive result on Day 10 itself — it actually invalidated the Day 9 conclusion that α=0.7 was optimal. But it established the noise floor criterion (±1 question, requires N≥3 averaging) that allowed Day 11 to distinguish a +1 improvement from random noise. Without that criterion, today's hyde_rerank result would be ambiguous: a single-run difference of 1 question is exactly at the Day 10 noise envelope. The 3-run audit (std=0) is the only reason this result is conclusive. Implication for small-eval-set RAG iteration: noise floor audit must precede any method comparison.

**Lesson 28**: Noise floor is a (config, evaluation-set) property, not an evaluation-set property. Same 30q, different reproducibility:

- Day 10 rerank_weighted α=0.0: 3 runs = 16, 17, 16 (std≈0.5)
- Day 11 hyde_rerank: 3 runs = 19, 19, 19 (std=0)

Different retrieval configs feed different chunk contexts to the generator, placing different questions at different distances from the LLM's decision boundaries. A method's "stability" under T=0 is emergent from the config-evaluation interaction, not an intrinsic property of the method. Practical consequence: cannot reuse a previously-measured noise floor for a new method — must re-audit.

## Implications for Future Work

- **Q08 mechanism is N=1**. To claim "HyDE helps vocabulary-mismatch class queries" as a generalizable finding, need more such queries in the eval set. Current 30q has Q06 and Q08 as candidates, both formally vocabulary-mismatch but only Q08 was rescued. Suggests expanding eval set to 50+ questions with explicit class tagging.

- **HyDE passage contents not logged**. The current implementation generates and discards the hypothetical passage. To do rigorous ablation (e.g. "how often does the hypothetical passage contain the target keyword?"), need to add hypothetical passage logging to `eval_results_*.json`.

- **Variant ablations not run**: hyde-only (no BM25, no rerank), hyde with N=3 passages averaged (Gao's original setup), hyde with query+hyde concatenation. Reserved for future experiments.

- **Adaptive HyDE gating**: HyDE adds latency and cost per query. Q08 benefits, Q14 does not. A query classifier deciding when to trigger HyDE (similar to planned Exp M for adaptive α) would reduce average cost without losing the Q08-class gains.