"""
Day 2: 带引用的 RAG 系统
核心改进：让 GPT 答案里标注每句话的来源（论文名 + 页码）
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# ============================================================
# Step 1-3: 跟 Day 1 一样（加载、切分、向量化）
# ============================================================
print("[1/4] 加载 PDF、切分、向量化...")
loader = PyPDFLoader("rag_paper.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)
print(f"      完成，共 {len(chunks)} 个 chunks")

# ============================================================
# Step 4: 检索（不再用 RetrievalQA，自己调用）
# ============================================================
print("\n[2/4] 检索相关 chunks...")

question = "What is retrieval-augmented generation and why is it useful?"

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retrieved_docs = retriever.invoke(question)

print(f"      检索到 {len(retrieved_docs)} 个 chunks")
for i, doc in enumerate(retrieved_docs, 1):
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page_label", "?")
    print(f"      [{i}] {source}, 第 {page} 页")

# ============================================================
# Step 5: 构造带引用要求的 prompt
# ============================================================
print("\n[3/4] 构造 prompt...")

# 把检索到的 chunks 格式化成带编号的文档
context_blocks = []
for i, doc in enumerate(retrieved_docs, 1):
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page_label", "?")
    content = doc.page_content
    block = f"[Source {i}] (file: {source}, page: {page})\n{content}"
    context_blocks.append(block)

context = "\n\n".join(context_blocks)

# 这是最关键的 prompt —— 指令 GPT 必须引用
prompt = f"""You are a helpful research assistant. Answer the user's question based ONLY on the provided sources below.

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

print("      prompt 构造完成")

# ============================================================
# Step 6: 调用 GPT 生成答案
# ============================================================
print("\n[4/4] 调用 GPT 生成答案...\n")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke(prompt)
answer = response.content

# ============================================================
# 输出结果
# ============================================================
print("=" * 60)
print("问题：", question)
print("=" * 60)
print("答案：")
print(answer)
print("=" * 60)
print("\n引用来源：")
for i, doc in enumerate(retrieved_docs, 1):
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page_label", "?")
    preview = doc.page_content[:150].replace("\n", " ")
    print(f"\n[Source {i}] {source}, 第 {page} 页")
    print(f"  原文预览: {preview}...")