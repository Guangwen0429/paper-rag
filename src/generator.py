"""
generator.py
负责：构造 prompt、调用 LLM、生成带引用的答案。
"""

from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# ============================================================
# Prompt 模板 —— 项目的核心"灵魂"
# ============================================================
PROMPT_TEMPLATE = """You are a helpful research assistant. Answer the user's question based ONLY on the provided sources below.

IMPORTANT RULES:
1. Every factual claim in your answer must be supported by the sources.
2. Cite sources using the format [Source N] immediately after each claim.
3. If the sources don't contain enough information to answer, say "I don't have enough information to answer this question."
4. Do NOT use knowledge outside the provided sources.

===== SOURCES =====
{context}
===== END OF SOURCES =====

Question: {question}

Answer:"""


def format_context(retrieved_docs: List[Document]) -> str:
    """
    把检索到的 chunks 格式化成带编号的 context 字符串。
    """
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_label", "?")
        content = doc.page_content
        block = f"[Source {i}] (file: {source}, page: {page})\n{content}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def generate_answer(
        question: str,
        retrieved_docs: List[Document],
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0,
) -> str:
    """
    根据问题和检索到的 chunks，生成带引用的答案。

    参数：
        question: 用户问题
        retrieved_docs: 检索到的 chunks
        llm_model: 使用的 LLM 模型（默认 gpt-4o-mini）
        temperature: 生成的随机性（0=最确定，1=最有创意）

    返回：
        GPT 生成的答案文本
    """
    # 1. 格式化 context
    context = format_context(retrieved_docs)

    # 2. 填充 prompt 模板
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # 3. 调用 LLM
    print(f"[generator] 使用模型: {llm_model}, temperature={temperature}")
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    from src.loader import load_all_pdfs, split_into_chunks
    from src.retriever import build_vectorstore, retrieve_chunks

    # 1. 完整流程
    docs = load_all_pdfs("papers")
    chunks = split_into_chunks(docs)
    vs = build_vectorstore(chunks)

    # 2. 提问
    question = "What is retrieval-augmented generation and why is it useful?"
    retrieved = retrieve_chunks(vs, question, k=3)

    # 3. 生成答案
    answer = generate_answer(question, retrieved)

    # 4. 显示
    print("\n" + "=" * 60)
    print("问题:", question)
    print("=" * 60)
    print("答案:")
    print(answer)
    print("=" * 60)

    print("\n引用来源:")
    for i, doc in enumerate(retrieved, 1):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page_label", "?")
        print(f"  [Source {i}] {source}, 第 {page} 页")