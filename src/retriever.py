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

import torch

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


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
# Cross-Encoder Reranker（Exp H，新增）
# ============================================================

# 全局单例：cross-encoder 模型加载慢（~2 秒），用缓存避免重复加载
_cross_encoder_instance = None


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """
    懒加载 cross-encoder 模型，缓存到全局变量。第一次调用会下载模型（~80MB），后续从本地缓存读取。

    Day 10 起：自动检测 CUDA，可用则在 GPU 上加载。在 RTX 4060 Laptop GPU
    上，100-pair 推理从 ~250ms 降到 ~25ms（10× 加速），这是 cross-encoder
    类模型在消费级 GPU 上的典型加速比。
    """
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[retriever] 正在加载 cross-encoder 模型: {model_name} (device={device})")
        _cross_encoder_instance = CrossEncoder(model_name, device=device)
        print(f"[retriever] cross-encoder 模型加载完成")
    return _cross_encoder_instance


def rerank_with_cross_encoder(
        candidates: List[Document],
        question: str,
        k: int = 3,
) -> List[Document]:
    """
    用 cross-encoder 对候选 chunks 精排，返回 top-k。

    Cross-encoder 和 bi-encoder (OpenAI embedding) 的区别：
    - Bi-encoder: 分别编码 query 和 doc 成两个向量，算余弦相似度。快，但缺少 query-doc token 级交互。
    - Cross-encoder: 把 query 和 doc 拼接一起输入 transformer，输出一个相关度分数。慢 10-100 倍，
      但能捕捉 bi-encoder 看不到的细微语义关系（比如"X 是什么" vs "X 怎么用"的区别）。

    在 RAG 里典型用法：先用快速检索（vector / BM25 / hybrid）召回 top-20 候选，再用 cross-encoder
    对这 20 个精排取 top-3 或 top-5。这样成本可控，质量比单用快速检索高。

    参数：
        candidates: 候选 chunks 列表（通常来自 hybrid/vector/BM25 检索）
        question: 用户问题
        k: 返回 top-k

    返回：
        按 cross-encoder 分数降序排列的 top-k chunks
    """
    if len(candidates) == 0:
        return []

    encoder = _get_cross_encoder()

    # 构造 (query, doc) pairs 批量打分
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = encoder.predict(pairs)

    # 按分数降序取 top-k
    scored_docs = list(zip(candidates, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:k]]


def hybrid_then_rerank(
        vectorstore: Chroma,
        bm25: BM25Okapi,
        chunks: List[Document],
        question: str,
        k: int = 3,
        n_candidates: int = 20,
) -> List[Document]:
    """
    End-to-end reranker 管线：hybrid 召回 top-n_candidates → cross-encoder 精排 → 返回 top-k。

    参数：
        vectorstore, bm25, chunks: 两种索引（共用 hybrid_retrieve 的架构）
        question: 用户问题
        k: 最终返回的 chunk 数
        n_candidates: hybrid 阶段召回多少个候选（默认 20）

    返回：
        top-k chunks（按 cross-encoder 分数排序）
    """
    # 1. Hybrid 召回 top-n_candidates 作为候选
    candidates = hybrid_retrieve(
        vectorstore, bm25, chunks, question,
        k=n_candidates,  # 这里 k 是"召回多少个"，不是"最终返回多少"
        n_candidates=n_candidates,
    )

    # 2. Cross-encoder 精排，返回 top-k
    return rerank_with_cross_encoder(candidates, question, k=k)


# ============================================================
# Weighted Rerank (Exp J)：混合 hybrid RRF 分数和 cross-encoder 分数
# ============================================================

def _min_max_normalize(scores: List[float]) -> List[float]:
    """
    Min-max normalization: 把一组分数压到 [0, 1] 区间。

    为什么需要：hybrid 的 RRF 分数和 cross-encoder 的分数量纲不同
    （RRF 通常在 0.01-0.05 范围，cross-encoder 通常在 -10 到 +10 范围）。
    如果不 normalize，直接加权融合会让大量纲的分数完全主导结果——
    相当于"没有加权"。

    边界情况：如果所有分数相等（max == min），返回全 0.5（中性值）。
    """
    if len(scores) == 0:
        return []
    max_s = max(scores)
    min_s = min(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_then_rerank_weighted(
        vectorstore: Chroma,
        bm25: BM25Okapi,
        chunks: List[Document],
        question: str,
        k: int = 3,
        n_candidates: int = 20,
        alpha: float = 0.5,
) -> List[Document]:
    """
    加权融合 hybrid RRF 分数和 cross-encoder 分数。

    流程：
      1. Hybrid 召回 top-n_candidates，同时保留 RRF 分数
      2. Cross-encoder 给每个候选打分
      3. 两组分数各自 min-max normalize 到 [0, 1]
      4. 融合分数 = (1 - alpha) * norm_rrf + alpha * norm_ce
      5. 按融合分数降序取 top-k

    alpha 的含义：
      - alpha=0.0: 完全用 RRF 排序 (退化成 Day 6 hybrid)
      - alpha=0.5: 各半加权
      - alpha=1.0: 完全用 cross-encoder 排序 (退化成 Day 8 rerank)

    返回：
        按融合分数排序的 top-k chunks
    """
    # ========== 1. Hybrid 召回（保留 RRF 分数）==========
    # 我们需要 RRF 分数，所以不能直接调 hybrid_retrieve（它只返回 chunks）。
    # 重新实现一遍带分数版本：

    # 1a. 各取 n_candidates 的 vector 和 BM25 召回
    vector_docs = retrieve_chunks(vectorstore, question, k=n_candidates)
    bm25_docs = bm25_retrieve(bm25, chunks, question, k=n_candidates)

    # 1b. RRF 融合
    def doc_id(doc: Document) -> str:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        preview = doc.page_content[:100]
        return f"{source}|{page}|{preview}"

    rrf_k_const = 60  # RRF 公式里的标准常数
    rrf_scores = {}
    doc_lookup = {}

    for rank, doc in enumerate(vector_docs, start=1):
        did = doc_id(doc)
        rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (rrf_k_const + rank)
        doc_lookup[did] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        did = doc_id(doc)
        rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (rrf_k_const + rank)
        doc_lookup[did] = doc

    # 1c. 取 RRF 分数 top-n_candidates 作为候选（可能少于 n_candidates，
    # 因为 vector 和 BM25 有重叠）
    sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:n_candidates]
    candidate_docs = [doc_lookup[did] for did in sorted_ids]
    candidate_rrf = [rrf_scores[did] for did in sorted_ids]

    if len(candidate_docs) == 0:
        return []

    # ========== 2. Cross-encoder 打分 ==========
    encoder = _get_cross_encoder()
    pairs = [(question, doc.page_content) for doc in candidate_docs]
    ce_scores = encoder.predict(pairs).tolist()  # numpy array → list

    # ========== 3. 各自 min-max normalize ==========
    norm_rrf = _min_max_normalize(candidate_rrf)
    norm_ce = _min_max_normalize(ce_scores)

    # ========== 4. 加权融合 ==========
    fused_scores = [
        (1 - alpha) * r + alpha * c
        for r, c in zip(norm_rrf, norm_ce)
    ]

    # ========== 5. 取 top-k ==========
    scored_docs = list(zip(candidate_docs, fused_scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:k]]


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