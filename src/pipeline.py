"""
pipeline.py
负责：把加载、切分、向量化、检索、生成串成一条流水线。
对外暴露简单接口：初始化一次，反复问问题。
"""

from typing import List, Dict, Any

from langchain_core.documents import Document

from src.loader import load_all_pdfs, split_into_chunks
from src.retriever import build_vectorstore, retrieve_chunks
from src.generator import generate_answer


class RAGPipeline:
    """
    端到端 RAG 流水线。

    典型用法：
        rag = RAGPipeline(papers_dir="papers")
        answer, sources = rag.ask("What is RAG?")
        print(answer)
    """

    def __init__(
            self,
            papers_dir: str = "papers",
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            embedding_model: str = "text-embedding-3-small",
            llm_model: str = "gpt-4o-mini",
            k: int = 3,
    ):
        """
        初始化流水线：加载 PDFs → 切分 → 建向量库。
        这个过程会调用 OpenAI embedding API，花几秒到几分钟。
        """
        # 保存配置（以后问问题时要用）
        self.papers_dir = papers_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.k = k

        # 执行建库流程
        print("=" * 60)
        print("[pipeline] 初始化 RAG 流水线...")
        print("=" * 60)

        # 1. 加载 PDFs
        documents = load_all_pdfs(papers_dir)

        # 2. 切分 chunks
        self.chunks = split_into_chunks(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 3. 建向量库
        self.vectorstore = build_vectorstore(
            self.chunks,
            embedding_model=embedding_model,
        )

        print("[pipeline] 初始化完成，可以开始提问\n")

    def ask(self, question: str) -> Dict[str, Any]:
        """
        问一个问题，返回答案和来源。

        返回：
            一个字典 {
                "question": 原问题,
                "answer": GPT 生成的答案,
                "sources": 检索到的 chunks 列表,
            }
        """
        # 1. 检索相关 chunks
        retrieved = retrieve_chunks(self.vectorstore, question, k=self.k)

        # 2. 生成答案
        answer = generate_answer(
            question,
            retrieved,
            llm_model=self.llm_model,
        )

        # 3. 打包返回
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved,
        }

    def pretty_print(self, result: Dict[str, Any]) -> None:
        """
        漂亮地打印一次问答结果。
        """
        print("\n" + "=" * 60)
        print("问题:", result["question"])
        print("=" * 60)
        print("答案:")
        print(result["answer"])
        print("=" * 60)
        print("\n引用来源:")
        for i, doc in enumerate(result["sources"], 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page_label", "?")
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"  [Source {i}] {source}, 第 {page} 页")
            print(f"      预览: {preview}...")


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 创建一个 pipeline（建库一次）
    rag = RAGPipeline(papers_dir="papers", k=3)

    # 问 3 个问题（共用同一个库，不用重新建）
    questions = [
        "What is retrieval-augmented generation?",
        "What is the main contribution of BERT?",
        "How does Chain-of-Thought prompting work?",
    ]

    for q in questions:
        result = rag.ask(q)
        rag.pretty_print(result)