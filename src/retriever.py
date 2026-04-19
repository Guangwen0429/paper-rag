"""
retriever.py
负责：把 chunks 向量化、存入向量数据库、根据问题检索相关 chunks。
"""

from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def build_vectorstore(
        chunks: List[Document],
        embedding_model: str = "text-embedding-3-small",
) -> Chroma:
    """
    把 chunks 向量化，存入 Chroma 向量数据库。

    参数：
        chunks: 切分好的文档 chunks
        embedding_model: OpenAI 的 embedding 模型名（默认 text-embedding-3-small）

    返回：
        一个 Chroma 向量数据库对象
    """
    print(f"[retriever] 使用 embedding 模型: {embedding_model}")
    print(f"[retriever] 正在向量化 {len(chunks)} 个 chunks...")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma.from_documents(chunks, embeddings)

    print(f"[retriever] 向量数据库构建完成")
    return vectorstore


def retrieve_chunks(
        vectorstore: Chroma,
        question: str,
        k: int = 3,
) -> List[Document]:
    """
    根据问题检索最相关的 k 个 chunks。

    参数：
        vectorstore: 已经构建好的 Chroma 数据库
        question: 用户的问题
        k: 返回几个最相关的 chunks（默认 3）

    返回：
        最相关的 chunks 列表
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)
    return retrieved_docs


# ============================================================
# 测试代码：单独运行 `python -m src.retriever` 时执行
# ============================================================
if __name__ == "__main__":
    from src.loader import load_all_pdfs, split_into_chunks

    # 1. 加载 + 切分
    docs = load_all_pdfs("papers")
    chunks = split_into_chunks(docs, chunk_size=500, chunk_overlap=50)

    # 2. 建库
    vs = build_vectorstore(chunks)

    # 3. 测试检索
    question = "What is retrieval-augmented generation?"
    results = retrieve_chunks(vs, question, k=3)

    print(f"\n=== 检索结果 ===")
    print(f"问题: {question}")
    print(f"返回 {len(results)} 个 chunks\n")

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page_label", "?")
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"[{i}] {source}, 第 {page} 页")
        print(f"    {preview}...\n")