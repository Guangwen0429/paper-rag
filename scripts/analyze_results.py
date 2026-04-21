"""
analyze_results.py

Chunk-level error analysis for Paper-RAG evaluation results.
Loads 4 experiment result files, produces terminal output + markdown report.

Run: python scripts/analyze_results.py

⚠️  KNOWN LIMITATION — READ BEFORE USING OUTPUT:

This script uses **character-level keyword matching** (`kw.lower() in
content.lower()`) as a proxy for "is the expected answer present in the
retrieved chunks."

This proxy is systematically unreliable for:
  - Numeric keywords ("8" matches "2018", "768", "[8]" citation, etc.)
  - Short words ("7" matches page numbers and benchmark scores)
  - Keywords that appear in unrelated semantic contexts ("reward" in
    "reward signal" vs "reward model training step")

As a result, Section 4 (Keyword Presence Audit) and Section 5 (Failure
Classification) produce **false positives** that systematically misclassify
Retrieval failures as Generator failures. See `evaluation/error_analysis_report.md`
Addendum for 3 verified cases (Q11, Q12, Q14) where this occurred.

Recommended use: Use this script's output as a first-pass hypothesis generator.
Always verify failure classifications via `scripts/inspect_question.py` before
drawing conclusions. A future version should replace character matching with
LLM-as-judge evaluation of whether the answer is semantically supported by
the retrieved context.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

EVAL_DIR = project_root / "evaluation"
CONFIGS = [
    ("vector", 3),
    ("vector", 5),
    ("hybrid", 3),
    ("hybrid", 5),
]


def load_results(mode: str, k: int) -> List[Dict[str, Any]]:
    path = EVAL_DIR / f"eval_results_{mode}_k{k}_15q.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_id(chunk: Dict[str, Any]) -> str:
    """Unique chunk identity = source + page + first 50 chars."""
    content_prefix = chunk["content"][:50].replace("\n", " ")
    return f"{chunk['source']}:p{chunk['page']}:{content_prefix}"


def keywords_in_chunks(chunks: List[Dict[str, Any]], keywords: List[str]) -> Dict[str, bool]:
    combined = " ".join(c["content"].lower() for c in chunks)
    return {kw: kw.lower() in combined for kw in keywords}


def main():
    all_data = {}
    for mode, k in CONFIGS:
        all_data[(mode, k)] = load_results(mode, k)

    by_qid = {}
    for (mode, k), results in all_data.items():
        for r in results:
            by_qid.setdefault(r["id"], {})[(mode, k)] = r

    md_lines = []
    def emit(line=""):
        print(line)
        md_lines.append(line)

    # Section 1: Summary matrix
    emit("# Paper-RAG Chunk-Level Error Analysis")
    emit("")
    emit("## 1. Keyword Hit Summary (4 configs × 15 questions)")
    emit("")
    emit("| QID | vector k=3 | vector k=5 | hybrid k=3 | hybrid k=5 |")
    emit("|-----|:----------:|:----------:|:----------:|:----------:|")

    qids = sorted(by_qid.keys())
    for qid in qids:
        row = f"| {qid} |"
        for mode, k in CONFIGS:
            r = by_qid[qid][(mode, k)]
            row += f" {'✓' if r['keyword_hit'] else '✗'} |"
        emit(row)

    total_row = "| **Total** |"
    for mode, k in CONFIGS:
        cnt = sum(1 for qid in qids if by_qid[qid][(mode, k)]["keyword_hit"])
        total_row += f" **{cnt}/15** |"
    emit(total_row)
    emit("")

    # Section 2: Fair k=5 comparison
    emit("## 2. Fair Comparison at k=5: What hybrid fixes / breaks")
    emit("")
    fixed, broken, both_ok, both_fail = [], [], [], []
    for qid in qids:
        v5 = by_qid[qid][("vector", 5)]["keyword_hit"]
        h5 = by_qid[qid][("hybrid", 5)]["keyword_hit"]
        if not v5 and h5: fixed.append(qid)
        elif v5 and not h5: broken.append(qid)
        elif v5 and h5: both_ok.append(qid)
        else: both_fail.append(qid)

    emit(f"- **Hybrid fixed** (vector ✗ → hybrid ✓): `{fixed}`")
    emit(f"- **Hybrid broke** (vector ✓ → hybrid ✗): `{broken}`")
    emit(f"- **Both correct**: `{both_ok}`")
    emit(f"- **Both failed**: `{both_fail}`")
    emit("")

    # Section 3: Chunk overlap (testing user's hypothesis)
    emit("## 3. Chunk Overlap: vector_k5 ∩ hybrid_k5")
    emit("")
    emit("Quantifies BM25's actual contribution — chunks unique to hybrid are BM25's additions.")
    emit("")
    emit("| QID | Overlap | Unique to vector | Unique to hybrid | Result (hybrid_k5) |")
    emit("|-----|:-------:|:----------------:|:----------------:|:------------------:|")

    for qid in qids:
        v5_chunks = by_qid[qid][("vector", 5)]["retrieved_chunks"]
        h5_chunks = by_qid[qid][("hybrid", 5)]["retrieved_chunks"]
        v5_ids = {chunk_id(c) for c in v5_chunks}
        h5_ids = {chunk_id(c) for c in h5_chunks}
        overlap = len(v5_ids & h5_ids)
        only_v = len(v5_ids - h5_ids)
        only_h = len(h5_ids - v5_ids)
        result = "✓" if by_qid[qid][("hybrid", 5)]["keyword_hit"] else "✗"
        emit(f"| {qid} | {overlap} | {only_v} | {only_h} | {result} |")
    emit("")

    # Section 4: Keyword presence audit
    emit("## 4. Keyword Presence Audit")
    emit("")
    emit("Did the expected answer keywords appear in ANY retrieved chunk?")
    emit("If all keywords present but keyword_hit=✗ → **generator failure** (not retrieval).")
    emit("")
    emit("| QID | Keywords | vector k=3 | vector k=5 | hybrid k=3 | hybrid k=5 |")
    emit("|-----|----------|:----------:|:----------:|:----------:|:----------:|")

    for qid in qids:
        expected = by_qid[qid][("vector", 3)]["expected_keywords"]
        row = f"| {qid} | {expected} |"
        for mode, k in CONFIGS:
            chunks = by_qid[qid][(mode, k)]["retrieved_chunks"]
            presence = keywords_in_chunks(chunks, expected)
            n_present = sum(presence.values())
            if n_present == len(expected):
                mark = "**all**"
            elif n_present == 0:
                mark = "none"
            else:
                mark = f"{n_present}/{len(expected)}"
            row += f" {mark} |"
        emit(row)
    emit("")

    # Section 5: Failure classification at hybrid_k5
    emit("## 5. Failure Classification (hybrid k=5)")
    emit("")
    emit("- **Retrieval failure**: expected keywords missing from all retrieved chunks")
    emit("- **Generator failure**: keywords present but LLM refused or answered wrong")
    emit("")
    emit("| QID | KW in chunks | Answer (first 80 chars) | Classification |")
    emit("|-----|:-----------:|-------------------------|----------------|")

    for qid in qids:
        r = by_qid[qid][("hybrid", 5)]
        if r["keyword_hit"]: continue
        expected = r["expected_keywords"]
        chunks = r["retrieved_chunks"]
        presence = keywords_in_chunks(chunks, expected)
        all_present = all(presence.values())
        none_present = not any(presence.values())
        n_present = sum(presence.values())

        answer = r["answer"][:80].replace("\n", " ").replace("|", "/")
        if none_present:
            mark = "none"; classification = "**Retrieval failure**"
        elif all_present:
            mark = "all"; classification = "**Generator failure**"
        else:
            mark = f"{n_present}/{len(expected)}"
            missing = [kw for kw, v in presence.items() if not v]
            classification = f"Partial retrieval (missing: {missing})"
        emit(f"| {qid} | {mark} | {answer}... | {classification} |")
    emit("")

    # Save report
    output_file = EVAL_DIR / "error_analysis_report.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print()
    print("=" * 70)
    print(f"Report saved to: {output_file.relative_to(project_root)}")
    print("=" * 70)


if __name__ == "__main__":
    main()