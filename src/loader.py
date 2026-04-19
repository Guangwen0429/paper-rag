"""
loader.py
负责：加载 PDF 文件，切分成 chunks。
输入：PDF 文件路径（或 papers 文件夹路径）
输出：chunks 列表
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_single_pdf(pdf_path: str) -> List[Document]:
    """
    加载单个 PDF 文件，返回原始 pages（每页一个 Document）。
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def load_all_pdfs(papers_dir: str) -> List[Document]:
    """
    加载一个文件夹下的所有 PDF 文件。
    """
    pdf_folder = Path(papers_dir)
    all_documents = []

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    print(f"[loader] 发现 {len(pdf_files)} 个 PDF 文件")

    for pdf_file in pdf_files:
        print(f"[loader]   加载 {pdf_file.name}...")
        docs = load_single_pdf(str(pdf_file))
        all_documents.extend(docs)

    print(f"[loader] 共加载 {len(all_documents)} 页")
    return all_documents


def split_into_chunks(
        documents: List[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
) -> List[Document]:
    """
    把文档切分成更小的 chunks。

    参数：
        chunk_size: 每个 chunk 的最大字符数（默认 500）
        chunk_overlap: 相邻 chunk 的重叠字符数（默认 50）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"[loader] 切分完成，共 {len(chunks)} 个 chunks")
    return chunks


# ============================================================
# 下面这段只在直接运行 `python loader.py` 时才会执行
# 目的：让你可以单独测试这个模块是否工作正常
# ============================================================
if __name__ == "__main__":
    # 测试：加载 papers/ 文件夹下所有 PDF
    docs = load_all_pdfs("papers")
    chunks = split_into_chunks(docs, chunk_size=500, chunk_overlap=50)

    # 随便看一个 chunk 的样子
    print("\n=== 样例 Chunk ===")
    print(f"来源：{chunks[0].metadata.get('source')}")
    print(f"页码：{chunks[0].metadata.get('page_label')}")
    print(f"内容预览：{chunks[0].page_content[:200]}...")