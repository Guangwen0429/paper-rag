"""
run_demo.py
演示脚本：展示 RAG 系统的典型用法。
运行：python scripts/run_demo.py
"""

import sys
from pathlib import Path

# 把项目根目录加到 Python path
# 这样 scripts/ 里的文件也能 import src/ 里的模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import RAGPipeline


def main():
    # 初始化流水线（建库一次）
    rag = RAGPipeline(
        papers_dir="papers",
        chunk_size=500,
        chunk_overlap=50,
        k=3,
    )

    # 预设一组演示问题
    demo_questions = [
        "What is retrieval-augmented generation?",
        "What is the main contribution of BERT?",
        "How does Chain-of-Thought prompting work?",
        "What are the key findings in the GPT-3 paper?",
    ]

    # 依次提问
    for question in demo_questions:
        result = rag.ask(question)
        rag.pretty_print(result)
        print()


if __name__ == "__main__":
    main()