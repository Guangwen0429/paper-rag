"""
run_eval.py

对评估集 eval_questions.json 里的每道题跑 RAG 系统。
支持一次跑多组 (retrieval_mode, k) 配置，每组独立保存结果文件。

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
# 配置：要跑的实验组合
# ============================================================
EXPERIMENTS = [
    ("vector", 3),
    ("vector", 5),
    ("hybrid", 3),
    ("hybrid", 5),
]
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

    # 把每个检索到的 chunk 整理成可存储的 dict
    retrieved_chunks_info = []
    for i, doc in enumerate(retrieved):
        source = doc.metadata.get("source", "")
        source_filename = Path(source).name
        page = doc.metadata.get("page", -1)
        is_correct_source = source_filename in expected_files

        retrieved_chunks_info.append({
            "rank": i + 1,
            "source": source_filename,
            "page": page,
            "is_correct_source": is_correct_source,
            "content": doc.page_content,
        })

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
        "retrieved_chunks": retrieved_chunks_info,
    }


# ============================================================
# 跑一组实验
# ============================================================

def run_one_experiment(vectorstore, bm25, chunks, questions, mode, k):
    print("\n" + "#" * 70)
    print(f"# EXPERIMENT: mode={mode}, k={k}")
    print("#" * 70)

    results = []
    for q in questions:
        print(f"\n[{q['id']}] {q['question']}")
        result = evaluate_single(vectorstore, bm25, chunks, q, mode, k)
        results.append(result)

        kw_mark = "✓" if result["keyword_hit"] else "✗"
        src_mark = "✓" if result["source_hit"] else "✗"
        print(f"  Keyword Hit:  [{kw_mark}]  expected: {result['expected_keywords']}")
        print(f"  Source Hit:   [{src_mark}]  {result['correct_chunks']}/{result['total_chunks']} chunks from {result['expected_files']}")

    # 汇总
    total = len(results)
    keyword_correct = sum(1 for r in results if r["keyword_hit"])
    source_correct = sum(1 for r in results if r["source_hit"])
    both_correct = sum(1 for r in results if r["keyword_hit"] and r["source_hit"])
    avg_routing_precision = sum(
        r["correct_chunks"] / r["total_chunks"] if r["total_chunks"] > 0 else 0
        for r in results
    ) / total

    summary = {
        "mode": mode,
        "k": k,
        "total": total,
        "keyword_hit": keyword_correct,
        "source_hit": source_correct,
        "both_correct": both_correct,
        "avg_routing_precision": avg_routing_precision,
    }

    print("\n" + "=" * 70)
    print(f"OVERALL METRICS (mode={mode}, k={k})")
    print("=" * 70)
    print(f"Total questions:              {total}")
    print(f"Keyword Hit (answer correct): {keyword_correct}/{total}  ({keyword_correct/total*100:.1f}%)")
    print(f"Source Hit (routing correct): {source_correct}/{total}  ({source_correct/total*100:.1f}%)")
    print(f"Both correct:                 {both_correct}/{total}  ({both_correct/total*100:.1f}%)")
    print(f"Avg chunk routing precision:  {avg_routing_precision*100:.1f}%")
    print("=" * 70)

    # 保存详细结果
    n_questions = len(results)
    output_filename = f"eval_results_{mode}_k{k}_{n_questions}q.json"
    output_file = project_root / "evaluation" / output_filename
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {output_file.relative_to(project_root)}")

    return summary


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
    print(f"Experiments to run: {EXPERIMENTS}")
    print("=" * 70)

    # 2. 加载 + 切分 + 建两种索引（只做一次，所有实验共用）
    docs = load_all_pdfs(PAPERS_DIR)
    chunks = split_into_chunks(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vectorstore = build_vectorstore(chunks)
    bm25, _ = build_bm25_index(chunks)

    # 3. 依次跑每组实验
    all_summaries = []
    for mode, k in EXPERIMENTS:
        summary = run_one_experiment(vectorstore, bm25, chunks, questions, mode, k)
        all_summaries.append(summary)

    # 4. 最后打印对比表
    print("\n" + "#" * 70)
    print("# FINAL COMPARISON ACROSS ALL EXPERIMENTS")
    print("#" * 70)
    print(f"\n{'Config':<20} {'Keyword Hit':<20} {'Source Hit':<20} {'Routing Prec':<15}")
    print("-" * 75)
    for s in all_summaries:
        config = f"{s['mode']} k={s['k']}"
        kw = f"{s['keyword_hit']}/{s['total']} ({s['keyword_hit']/s['total']*100:.1f}%)"
        src = f"{s['source_hit']}/{s['total']} ({s['source_hit']/s['total']*100:.1f}%)"
        prec = f"{s['avg_routing_precision']*100:.1f}%"
        print(f"{config:<20} {kw:<20} {src:<20} {prec:<15}")
    print("=" * 75)

    # 保存汇总表
    summary_file = project_root / "evaluation" / "eval_summary_all.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_file.relative_to(project_root)}")


if __name__ == "__main__":
    main()