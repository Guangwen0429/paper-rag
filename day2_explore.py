"""
Day 2 探索：看看 chunk 里除了文字还藏着什么
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 PDF
loader = PyPDFLoader("rag_paper.pdf")
documents = loader.load()

# 切 chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 看看第一个 chunk 长什么样
print("========== Chunk 0 的完整结构 ==========")
print("类型：", type(chunks[0]))
print("\n正文内容（page_content）：")
print(chunks[0].page_content)
print("\n元数据（metadata）：")
print(chunks[0].metadata)

# 看看第 50 个 chunk（靠近中间）
print("\n\n========== Chunk 50 的元数据 ==========")
print(chunks[50].metadata)

# 看看最后一个 chunk
print("\n\n========== 最后一个 chunk 的元数据 ==========")
print(chunks[-1].metadata)