"""
run_eval.py
对评估集 eval_questions.json 里的每道题跑 RAG 系统，
自动判断 keyword hit 和 source hit，输出整体指标。

运行：python scripts/run_eval.py
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# 把项目根目录加入 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import RAGPipeline


# ============================================================
# 评估指标
# ============================================================

def check_keyword_hit(answer: str, expected_keywords: List[str]) -> bool:
    """
    检查 GPT 答案里是否包含所有预期 keywords。
    大小写不敏感。
    """
    answer_lower = answer.lower()
    for kw in expected_keywords:
        if kw.lower() not in answer_lower:
            return False
    return True


def check_source_hit(retrieved_docs: List, expected_files: List[str]) -> bool:
    """
    检查检索到的 chunks 中，是否至少有一个来自预期的 source files。
    """
    retrieved_sources = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "")
        # source 可能是 "papers\\01_transformer.pdf" 或 "papers/01_transformer.pdf"
        # 只取文件名
        source_filename = Path(source).name
        retrieved_sources.add(source_filename)

    for expected in expected_files:
        if expected in retrieved_sources:
            return True
    return False


def count_correct_source_chunks(retrieved_docs: List, expected_files: List[str]) -> int:
    """
    统计检索到的 chunks 中，有几个来自预期的 source files。
    """
    count = 0
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "")
        source_filename = Path(source).name
        if source_filename in expected_files:
            count += 1
    return count


# ============================================================
# 单题评估
# ============================================================

def evaluate_single(rag: RAGPipeline, question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    对单道题跑 RAG 并计算指标。
    """
    qid = question_data["id"]
    question = question_data["question"]
    expected_keywords = question_data["expected_answer_keywords"]
    expected_files = question_data["expected_source_files"]

    # 跑 RAG
    result = rag.ask(question)
    answer = result["answer"]
    retrieved = result["sources"]

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
# 主函数：跑完整评估
# ============================================================

def main():
    # 1. 读取评估集
    eval_file = project_root / "evaluation" / "eval_questions.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("=" * 70)
    print(f"Evaluation Set: {len(questions)} questions")
    print("=" * 70)

    # 2. 初始化 RAGPipeline（建库一次）
    rag = RAGPipeline(papers_dir="papers", k=3)

    # 3. 依次评估每道题
    results = []
    for q in questions:
        print(f"\n{'=' * 70}")
        print(f"[{q['id']}] {q['question']}")
        print(f"{'=' * 70}")

        result = evaluate_single(rag, q)
        results.append(result)

        # 打印单题结果
        kw_mark = "✓" if result["keyword_hit"] else "✗"
        src_mark = "✓" if result["source_hit"] else "✗"
        print(f"Keyword Hit:  [{kw_mark}]  expected: {result['expected_keywords']}")
        print(
            f"Source Hit:   [{src_mark}]  {result['correct_chunks']}/{result['total_chunks']} chunks from {result['expected_files']}")
        print(f"Answer: {result['answer'][:200]}...")

    # 4. 汇总统计
    total = len(results)
    keyword_correct = sum(1 for r in results if r["keyword_hit"])
    source_correct = sum(1 for r in results if r["source_hit"])
    both_correct = sum(1 for r in results if r["keyword_hit"] and r["source_hit"])

    # 平均 chunk 路由精度
    avg_routing_precision = sum(
        r["correct_chunks"] / r["total_chunks"] if r["total_chunks"] > 0 else 0
        for r in results
    ) / total

    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    print(f"Total questions:              {total}")
    print(f"Keyword Hit (answer correct): {keyword_correct}/{total}  ({keyword_correct / total * 100:.1f}%)")
    print(f"Source Hit (routing correct): {source_correct}/{total}  ({source_correct / total * 100:.1f}%)")
    print(f"Both correct:                 {both_correct}/{total}  ({both_correct / total * 100:.1f}%)")
    print(f"Avg chunk routing precision:  {avg_routing_precision * 100:.1f}%")
    print("=" * 70)

    # 5. 保存结果到 JSON（方便后续分析）
    output_file = project_root / "evaluation" / "eval_results.json"
    # 转 result 里的 sources 不存（太大），只存其他字段
    results_to_save = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "sources"}
        results_to_save.append(r_copy)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_file.relative_to(project_root)}")


if __name__ == "__main__":
    main()