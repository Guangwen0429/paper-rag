"""
Day 1: 最小可运行的 RAG 系统
目标：上传一篇 PDF，问一个问题，得到基于 PDF 内容的答案
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# ============================================================
# Step 1: 加载 PDF
# ============================================================
print("[1/5] 正在加载 PDF...")
loader = PyPDFLoader("rag_paper.pdf")
documents = loader.load()
print(f"      加载完成，共 {len(documents)} 页")

# ============================================================
# Step 2: 切分成 chunks
# ============================================================
print("[2/5] 正在切分文档...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每个 chunk 大约 500 个字符
    chunk_overlap=50     # 相邻 chunk 之间重叠 50 字符，避免切断关键信息
)
chunks = splitter.split_documents(documents)
print(f"      切分完成，共 {len(chunks)} 个 chunks")

# ============================================================
# Step 3: 向量化并存入向量数据库
# ============================================================
print("[3/5] 正在生成向量并存入数据库...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)
print(f"      数据库构建完成")

# ============================================================
# Step 4: 构建 RAG 问答链
# ============================================================
print("[4/5] 正在构建问答链...")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # 每次检索 3 个最相关的 chunk
    return_source_documents=True  # 同时返回检索到的原始 chunks
)
print(f"      问答链就绪")

# ============================================================
# Step 5: 提问
# ============================================================
print("[5/5] 开始提问...\n")
question = "What is retrieval-augmented generation and why is it useful?"

result = qa_chain.invoke({"query": question})

print("=" * 60)
print("问题：", question)
print("=" * 60)
print("答案：", result["result"])
print("=" * 60)
print("\n检索到的原始 chunks（前 3 个）：")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"\n--- Chunk {i} ---")
    print(doc.page_content[:200] + "...")  # 只显示前 200 字符