"""
retriever.py
负责：把 chunks 向量化、存入向量数据库、根据问题检索相关 chunks。

支持两种检索策略：
- 向量检索（语义相似）：build_vectorstore + retrieve_chunks
- BM25 检索（字面匹配）：build_bm25_index + bm25_retrieve
- Hybrid 检索（RRF 融合两种方法）：hybrid_retrieve
"""

import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi


# ============================================================
# 向量检索（原有，保持不变）
# ============================================================

def build_vectorstore(
        chunks: List[Document],
        embedding_model: str = "text-embedding-3-small",
) -> Chroma:
    """
    把 chunks 向量化，存入 Chroma 向量数据库。
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
    根据问题做向量检索，返回 top-k 最相关的 chunks。
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)
    return retrieved_docs


# ============================================================
# BM25 检索（新增）
# ============================================================

def _simple_tokenize(text: str) -> List[str]:
    """
    简单英文分词：小写化、按非字母数字字符切分。
    BM25 需要把文本切成 token list。
    """
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks: List[Document]) -> Tuple[BM25Okapi, List[Document]]:
    """
    把 chunks 建成 BM25 索引。

    返回：
        bm25: BM25Okapi 索引对象
        chunks: 原始 chunks 列表（BM25 返回 index，需要靠 chunks 取回文档）
    """
    print(f"[retriever] 正在构建 BM25 索引（{len(chunks)} 个 chunks）...")

    tokenized_corpus = [_simple_tokenize(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"[retriever] BM25 索引构建完成")
    return bm25, chunks


def bm25_retrieve(
        bm25: BM25Okapi,
        chunks: List[Document],
        question: str,
        k: int = 3,
) -> List[Document]:
    """
    用 BM25 检索 top-k 最相关的 chunks。
    """
    tokenized_query = _simple_tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    # 按分数降序，取 top-k 的 index
    top_k_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    return [chunks[i] for i in top_k_indices]


# ============================================================
# Hybrid 检索（新增）：RRF 融合向量检索和 BM25
# ============================================================

def hybrid_retrieve(
        vectorstore: Chroma,
        bm25: BM25Okapi,
        chunks: List[Document],
        question: str,
        k: int = 3,
        rrf_k: int = 60,
        n_candidates: int = 20,
) -> List[Document]:
    """
    Hybrid 检索：同时用向量和 BM25 检索，用 RRF 融合排名，返回 top-k。

    参数：
        vectorstore: Chroma 向量数据库
        bm25: BM25 索引
        chunks: 原始 chunks 列表（用来在两种检索之间对齐文档）
        question: 用户问题
        k: 最终返回的 chunk 数（默认 3）
        rrf_k: RRF 公式里的常数（默认 60，工业界标准）
        n_candidates: 每种检索方法各自取多少个候选（默认 20，保证融合前有足够候选）

    RRF 公式：
        score(c) = sum over retrievers of  1 / (rrf_k + rank(c))

    返回：
        top-k 融合后的 chunks 列表
    """
    # 1. 向量检索：取 top-n_candidates
    vector_docs = retrieve_chunks(vectorstore, question, k=n_candidates)

    # 2. BM25 检索：取 top-n_candidates
    bm25_docs = bm25_retrieve(bm25, chunks, question, k=n_candidates)

    # 3. 计算 RRF 分数
    # 用 chunk 的 (source, page, 前100字符) 作为"身份" —— 不能直接用对象比较
    def doc_id(doc: Document) -> str:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        preview = doc.page_content[:100]
        return f"{source}|{page}|{preview}"

    rrf_scores = {}  # doc_id -> RRF 分数
    doc_lookup = {}  # doc_id -> Document 对象

    # 向量检索的贡献
    for rank, doc in enumerate(vector_docs, start=1):
        did = doc_id(doc)
        rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (rrf_k + rank)
        doc_lookup[did] = doc

    # BM25 检索的贡献
    for rank, doc in enumerate(bm25_docs, start=1):
        did = doc_id(doc)
        rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (rrf_k + rank)
        doc_lookup[did] = doc

    # 4. 按 RRF 分数降序，取 top-k
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda did: rrf_scores[did],
        reverse=True,
    )[:k]

    return [doc_lookup[did] for did in sorted_ids]


# ============================================================
# 测试代码：单独运行 `python -m src.retriever` 时执行
# ============================================================
if __name__ == "__main__":
    from src.loader import load_all_pdfs, split_into_chunks

    # 1. 加载 + 切分
    docs = load_all_pdfs("papers")
    chunks = split_into_chunks(docs, chunk_size=500, chunk_overlap=50)

    # 2. 建两种索引
    vs = build_vectorstore(chunks)
    bm25, _ = build_bm25_index(chunks)

    # 3. 测试三种检索策略
    question = "What architecture does DPR use as its question and passage encoder?"

    print(f"\n{'=' * 60}")
    print(f"问题: {question}")
    print(f"{'=' * 60}")

    print("\n--- 向量检索 (k=3) ---")
    for i, doc in enumerate(retrieve_chunks(vs, question, k=3), 1):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page_label", "?")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"[{i}] {source}, 第 {page} 页: {preview}...")

    print("\n--- BM25 检索 (k=3) ---")
    for i, doc in enumerate(bm25_retrieve(bm25, chunks, question, k=3), 1):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page_label", "?")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"[{i}] {source}, 第 {page} 页: {preview}...")

    print("\n--- Hybrid 检索 (k=3, RRF) ---")
    for i, doc in enumerate(hybrid_retrieve(vs, bm25, chunks, question, k=3), 1):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page_label", "?")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"[{i}] {source}, 第 {page} 页: {preview}...")