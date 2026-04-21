"""
run_eval.py

对评估集 eval_questions.json 里的每道题跑 RAG 系统，
支持三种检索模式：vector / bm25 / hybrid。

运行：python scripts/run_eval.py
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# 把项目根目录加入 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.loader import load_all_pdfs, split_into_chunks
from src.retriever import (
    build_vectorstore,
    build_bm25_index,
    retrieve_chunks,
    bm25_retrieve,
    hybrid_retrieve,
)
from src.generator import generate_answer


# ============================================================
# 配置：切换检索策略
# ============================================================
RETRIEVAL_MODE = "hybrid"   # 可选 "vector" / "bm25" / "hybrid"
K = 3                       # 最终返回几个 chunks
PAPERS_DIR = "papers"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ============================================================
# 评估指标
# ============================================================

def check_keyword_hit(answer: str, expected_keywords: List[str]) -> bool:
    answer_lower = answer.lower()
    for kw in expected_keywords:
        if kw.lower() not in answer_lower:
            return False
    return True


def check_source_hit(retrieved_docs: List, expected_files: List[str]) -> bool:
    retrieved_sources = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "")
        source_filename = Path(source).name
        retrieved_sources.add(source_filename)

    for expected in expected_files:
        if expected in retrieved_sources:
            return True
    return False


def count_correct_source_chunks(retrieved_docs: List, expected_files: List[str]) -> int:
    count = 0
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "")
        source_filename = Path(source).name
        if source_filename in expected_files:
            count += 1
    return count


# ============================================================
# 检索切换
# ============================================================

def retrieve(mode: str, vectorstore, bm25, chunks, question: str, k: int):
    """根据模式调用对应的检索方法"""
    if mode == "vector":
        return retrieve_chunks(vectorstore, question, k=k)
    elif mode == "bm25":
        return bm25_retrieve(bm25, chunks, question, k=k)
    elif mode == "hybrid":
        return hybrid_retrieve(vectorstore, bm25, chunks, question, k=k)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")


# ============================================================
# 单题评估
# ============================================================

def evaluate_single(
        vectorstore, bm25, chunks,
        question_data: Dict[str, Any],
        mode: str,
        k: int,
) -> Dict[str, Any]:
    qid = question_data["id"]
    question = question_data["question"]
    expected_keywords = question_data["expected_answer_keywords"]
    expected_files = question_data["expected_source_files"]

    # 检索
    retrieved = retrieve(mode, vectorstore, bm25, chunks, question, k)

    # 生成答案
    answer = generate_answer(question, retrieved)

    # 计算指标
    keyword_hit = check_keyword_hit(answer, expected_keywords)
    source_hit = check_source_hit(retrieved, expected_files)
    correct_chunks = count_correct_source_chunks(retrieved, expected_files)
    total_chunks = len(retrieved)

    return {
        "id": qid,
        "question": question,
        "expected_keywords": expected_keywords,
        "expected_files": expected_files,
        "answer": answer,
        "keyword_hit": keyword_hit,
        "source_hit": source_hit,
        "correct_chunks": correct_chunks,
        "total_chunks": total_chunks,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    # 1. 读取评估集
    eval_file = project_root / "evaluation" / "eval_questions.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("=" * 70)
    print(f"Evaluation Set: {len(questions)} questions")
    print(f"Retrieval mode: {RETRIEVAL_MODE}, k={K}")
    print("=" * 70)

    # 2. 加载 + 切分 + 建两种索引
    docs = load_all_pdfs(PAPERS_DIR)
    chunks = split_into_chunks(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vectorstore = build_vectorstore(chunks)
    bm25, _ = build_bm25_index(chunks)

    # 3. 依次评估每道题
    results = []
    for q in questions:
        print(f"\n{'=' * 70}")
        print(f"[{q['id']}] {q['question']}")
        print(f"{'=' * 70}")

        result = evaluate_single(vectorstore, bm25, chunks, q, RETRIEVAL_MODE, K)
        results.append(result)

        kw_mark = "✓" if result["keyword_hit"] else "✗"
        src_mark = "✓" if result["source_hit"] else "✗"
        print(f"Keyword Hit:  [{kw_mark}]  expected: {result['expected_keywords']}")
        print(f"Source Hit:   [{src_mark}]  {result['correct_chunks']}/{result['total_chunks']} chunks from {result['expected_files']}")
        print(f"Answer: {result['answer'][:200]}...")

    # 4. 汇总
    total = len(results)
    keyword_correct = sum(1 for r in results if r["keyword_hit"])
    source_correct = sum(1 for r in results if r["source_hit"])
    both_correct = sum(1 for r in results if r["keyword_hit"] and r["source_hit"])
    avg_routing_precision = sum(
        r["correct_chunks"] / r["total_chunks"] if r["total_chunks"] > 0 else 0
        for r in results
    ) / total

    print("\n" + "=" * 70)
    print(f"OVERALL METRICS (mode={RETRIEVAL_MODE}, k={K})")
    print("=" * 70)
    print(f"Total questions:              {total}")
    print(f"Keyword Hit (answer correct): {keyword_correct}/{total}  ({keyword_correct/total*100:.1f}%)")
    print(f"Source Hit (routing correct): {source_correct}/{total}  ({source_correct/total*100:.1f}%)")
    print(f"Both correct:                 {both_correct}/{total}  ({both_correct/total*100:.1f}%)")
    print(f"Avg chunk routing precision:  {avg_routing_precision*100:.1f}%")
    print("=" * 70)

    # 5. 保存结果
    output_file = project_root / "evaluation" / "eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_file.relative_to(project_root)}")


if __name__ == "__main__":
    main()