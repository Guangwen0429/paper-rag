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
    hybrid_then_rerank,
    hybrid_then_rerank_weighted,
    hyde_then_rerank,
    hyde_ensemble_then_rerank,
)
from src.generator import generate_answer


# ============================================================
# 配置：要跑的实验组合
# ============================================================
EXPERIMENTS = [
    # Day 12 (post-bug-fix re-audit): re-run rerank baseline and HyDE single
    # 3 times each, with corrected check_keyword_hit (OR support for Q23/Q24).
    # Goal: verify Lesson 25 (HyDE +1 over rerank exceeds noise floor) still
    # holds after the OR/AND fix.
    ("rerank", 5, 500, 50, 0.0),       # run 1
    ("rerank", 5, 500, 50, 0.0),       # run 2
    ("rerank", 5, 500, 50, 0.0),       # run 3
    ("hyde_rerank", 5, 500, 50, 0.0),  # run 1
    ("hyde_rerank", 5, 500, 50, 0.0),  # run 2
    ("hyde_rerank", 5, 500, 50, 0.0),  # run 3
]
PAPERS_DIR = "papers"


# ============================================================
# 评估指标
# ============================================================

def check_keyword_hit(answer: str, expected_keywords: List[str], match_mode: str = "all") -> bool:
    """
    检查答案是否命中期望关键词。

    match_mode:
        "all": 所有关键词都必须出现（默认，适合多事实题如 Q11/Q13）
        "any": 任一关键词出现即可（适合同义词题如 Q23/Q24，
               expected_keywords 是同义词列表如 ['sinusoidal','sine','cosine']）

    Day 12 引入 match_mode 参数，修复之前所有题目都按 AND 判定导致
    Q23/Q24 这类同义词题永远被错判 ✗ 的 bug。详见 Lesson 31。
    """
    answer_lower = answer.lower()
    if match_mode == "any":
        for kw in expected_keywords:
            if kw.lower() in answer_lower:
                return True
        return False
    else:  # default "all"
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

def retrieve(mode: str, vectorstore, bm25, chunks, question: str, k: int, alpha: float = 0.5):
    if mode == "vector":
        return retrieve_chunks(vectorstore, question, k=k)
    elif mode == "bm25":
        return bm25_retrieve(bm25, chunks, question, k=k)
    elif mode == "hybrid":
        return hybrid_retrieve(vectorstore, bm25, chunks, question, k=k)
    elif mode == "rerank":
        return hybrid_then_rerank(vectorstore, bm25, chunks, question, k=k, n_candidates=20)
    elif mode == "rerank_weighted":
        return hybrid_then_rerank_weighted(
            vectorstore, bm25, chunks, question,
            k=k, n_candidates=20, alpha=alpha,
        )
    elif mode == "hyde_rerank":
        return hyde_then_rerank(
            vectorstore, bm25, chunks, question,
            k=k, n_candidates=20,
        )
    elif mode == "hyde_ensemble_rerank":
        return hyde_ensemble_then_rerank(
            vectorstore, bm25, chunks, question,
            k=k, n_candidates=20, n_passages=5,
        )
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
        alpha: float = 0.5,
) -> Dict[str, Any]:
    qid = question_data["id"]
    question = question_data["question"]
    expected_keywords = question_data["expected_answer_keywords"]
    expected_files = question_data["expected_source_files"]
    # Day 12 新增：从题目读取 keyword_match_mode（默认 "all"）
    match_mode = question_data.get("keyword_match_mode", "all")

    # 检索
    retrieved = retrieve(mode, vectorstore, bm25, chunks, question, k, alpha)

    # 生成答案
    answer = generate_answer(question, retrieved)

    # 计算指标
    keyword_hit = check_keyword_hit(answer, expected_keywords, match_mode=match_mode)
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
        "keyword_match_mode": match_mode,
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

def run_one_experiment(vectorstore, bm25, chunks, questions, mode, k, chunk_size, chunk_overlap, alpha=0.5, run_id=None):
    print("\n" + "#" * 70)
    print(f"# EXPERIMENT: mode={mode}, k={k}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print("#" * 70)

    results = []
    for q in questions:
        print(f"\n[{q['id']}] {q['question']}")
        result = evaluate_single(vectorstore, bm25, chunks, q, mode, k, alpha)
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
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "alpha": alpha,
        "total": total,
        "keyword_hit": keyword_correct,
        "source_hit": source_correct,
        "both_correct": both_correct,
        "avg_routing_precision": avg_routing_precision,
    }

    print("\n" + "=" * 70)
    print(f"OVERALL METRICS (mode={mode}, k={k}, chunk_size={chunk_size})")
    print("=" * 70)
    print(f"Total questions:              {total}")
    print(f"Keyword Hit (answer correct): {keyword_correct}/{total}  ({keyword_correct/total*100:.1f}%)")
    print(f"Source Hit (routing correct): {source_correct}/{total}  ({source_correct/total*100:.1f}%)")
    print(f"Both correct:                 {both_correct}/{total}  ({both_correct/total*100:.1f}%)")
    print(f"Avg chunk routing precision:  {avg_routing_precision*100:.1f}%")
    print("=" * 70)

    # 保存详细结果（文件名包含 chunk_size，run_id 区分多次重跑——Day 11 引入）
    n_questions = len(results)
    run_suffix = f"_run{run_id}" if run_id is not None else ""
    if mode == "rerank_weighted":
        output_filename = f"eval_results_{mode}_k{k}_cs{chunk_size}_a{alpha}_{n_questions}q{run_suffix}.json"
    else:
        output_filename = f"eval_results_{mode}_k{k}_cs{chunk_size}_{n_questions}q{run_suffix}.json"
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

    # 2. 按 (chunk_size, chunk_overlap) 分组建索引（相同参数只建一次，共用）
    docs = load_all_pdfs(PAPERS_DIR)

    # 用 dict 缓存已构建的索引，key = (chunk_size, chunk_overlap)
    index_cache = {}

    def get_or_build_index(chunk_size, chunk_overlap):
        key = (chunk_size, chunk_overlap)
        if key in index_cache:
            return index_cache[key]

        print(f"\n{'*' * 70}")
        print(f"* Building index for chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        print(f"{'*' * 70}")
        chunks = split_into_chunks(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vectorstore = build_vectorstore(chunks)
        bm25, _ = build_bm25_index(chunks)
        index_cache[key] = (vectorstore, bm25, chunks)
        return vectorstore, bm25, chunks

    # 3. 依次跑每组实验
    # 检测同一个配置是否出现多次：如果是，给每次跑加 run_id 区分输出文件
    from collections import Counter
    config_counts = Counter(EXPERIMENTS)
    config_seen = Counter()

    all_summaries = []
    for mode, k, chunk_size, chunk_overlap, alpha in EXPERIMENTS:
        config = (mode, k, chunk_size, chunk_overlap, alpha)
        # 如果某个配置出现多次（多 run 重跑），给每次加编号；只出现一次的配置 run_id=None
        if config_counts[config] > 1:
            config_seen[config] += 1
            run_id = config_seen[config]
        else:
            run_id = None

        vectorstore, bm25, chunks = get_or_build_index(chunk_size, chunk_overlap)
        summary = run_one_experiment(
            vectorstore, bm25, chunks, questions,
            mode, k, chunk_size, chunk_overlap, alpha, run_id=run_id,
        )
        # 把 run_id 也存进 summary，方便后续分析
        summary["run_id"] = run_id
        all_summaries.append(summary)

    # 4. 最后打印对比表
    print("\n" + "#" * 70)
    print("# FINAL COMPARISON ACROSS ALL EXPERIMENTS")
    print("#" * 70)
    print(f"\n{'Config':<28} {'Keyword Hit':<20} {'Source Hit':<20} {'Routing Prec':<15}")
    print("-" * 83)
    for s in all_summaries:
        alpha_str = f" a={s.get('alpha', '-')}" if s['mode'] == 'rerank_weighted' else ""
        run_str = f" run={s.get('run_id')}" if s.get('run_id') is not None else ""
        config = f"{s['mode']} k={s['k']} cs={s['chunk_size']}{alpha_str}{run_str}"
        kw = f"{s['keyword_hit']}/{s['total']} ({s['keyword_hit'] / s['total'] * 100:.1f}%)"
        src = f"{s['source_hit']}/{s['total']} ({s['source_hit'] / s['total'] * 100:.1f}%)"
        prec = f"{s['avg_routing_precision'] * 100:.1f}%"
        print(f"{config:<40} {kw:<20} {src:<20} {prec:<15}")
    print("=" * 83)

    # Day 12 post-bugfix 专用 summary 文件名
    summary_file = project_root / "evaluation" / "eval_summary_post_bugfix_day12_30q.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_file.relative_to(project_root)}")


if __name__ == "__main__":
    main()