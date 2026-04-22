"""
inspect_question.py

Print full chunk contents for a specific question across all 4 experiment configs.
Useful for verifying hypotheses about why a question failed/succeeded.

Run: python scripts/inspect_question.py Q14
     python scripts/inspect_question.py Q14 Q07 Q11    # multiple at once
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

EVAL_DIR = project_root / "evaluation"
# 每个 config: (mode, k, chunk_size)
# Day 6 用 cs=500, Day 7 用 cs=250
# 默认对比 Day 7 的 4 组 (cs=250)
CONFIGS = [
    ("vector", 3, 250),
    ("vector", 5, 250),
    ("hybrid", 3, 250),
    ("hybrid", 5, 250),
]


def load_results(mode: str, k: int, chunk_size: int = 250) -> List[Dict[str, Any]]:
    # Day 6 文件名（无 cs）作为兼容回退，Day 7 用带 cs 的新文件名
    if chunk_size == 500:
        # Day 6 文件名
        path = EVAL_DIR / f"eval_results_{mode}_k{k}_15q.json"
    else:
        # Day 7+ 文件名
        path = EVAL_DIR / f"eval_results_{mode}_k{k}_cs{chunk_size}_15q.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Wrap each keyword in the text with >>KW<< markers for visual scanning."""
    out = text
    for kw in keywords:
        # case-insensitive replace, preserving original case
        idx = 0
        result = ""
        low = out.lower()
        kw_low = kw.lower()
        while idx < len(out):
            pos = low.find(kw_low, idx)
            if pos == -1:
                result += out[idx:]
                break
            result += out[idx:pos]
            result += f">>>{out[pos:pos+len(kw)]}<<<"
            idx = pos + len(kw)
            low = out.lower()
        out = result
    return out


def inspect_question(qid: str):
    print("\n" + "█" * 75)
    print(f"█  QUESTION: {qid}")
    print("█" * 75)

    # Load all 4 configs
    all_configs = {}
    question_text = None
    expected_keywords = None
    expected_files = None
    for mode, k, chunk_size in CONFIGS:
        results = load_results(mode, k, chunk_size)
        q = next((r for r in results if r["id"] == qid), None)
        if q is None:
            print(f"[!] {qid} not found in {mode} k={k}")
            return
        all_configs[(mode, k)] = q
        if question_text is None:
            question_text = q["question"]
            expected_keywords = q["expected_keywords"]
            expected_files = q["expected_files"]

    # Question metadata
    print(f"\nQ: {question_text}")
    print(f"Expected keywords: {expected_keywords}")
    print(f"Expected source files: {expected_files}")

    # Per-config breakdown
    for (mode, k), r in all_configs.items():
        print("\n" + "─" * 75)
        print(f"  CONFIG: {mode}  k={k}  cs={chunk_size}")
        print("─" * 75)
        kw_mark = "✓" if r["keyword_hit"] else "✗"
        print(f"  Keyword Hit: [{kw_mark}]    Source Hit: [{'✓' if r['source_hit'] else '✗'}]    "
              f"Correct chunks: {r['correct_chunks']}/{r['total_chunks']}")
        print(f"  Answer: {r['answer']}")
        print()

        # Each chunk
        for chunk in r["retrieved_chunks"]:
            src_mark = "✓" if chunk["is_correct_source"] else "✗"
            print(f"  ┌─ [Rank {chunk['rank']}] {chunk['source']} p{chunk['page']}  "
                  f"source={src_mark}")

            # Per-keyword presence in THIS chunk
            content_low = chunk["content"].lower()
            kw_presence = [(kw, kw.lower() in content_low) for kw in expected_keywords]
            presence_str = "  ".join(
                f"{kw}={'✓' if present else '✗'}" for kw, present in kw_presence
            )
            print(f"  │  Keywords in this chunk: {presence_str}")

            # Content with keywords highlighted
            highlighted = highlight_keywords(chunk["content"], expected_keywords)
            # Indent each line for readability
            for line in highlighted.split("\n"):
                print(f"  │  {line}")
            print(f"  └─")
            print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_question.py Q14 [Q07 Q11 ...]")
        sys.exit(1)

    qids = sys.argv[1:]
    for qid in qids:
        inspect_question(qid)


if __name__ == "__main__":
    main()