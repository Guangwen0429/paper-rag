"""
inspect_chunks.py

Dump all chunks to a file, or search for specific content patterns.
Useful for understanding how chunking split specific parts of papers.

Usage:
    python scripts/inspect_chunks.py --dump-all         # 所有 chunks 导出到文件
    python scripts/inspect_chunks.py --search "h = 8"    # 搜含有该文本的 chunks
    python scripts/inspect_chunks.py --search "h = 8" --source 01_transformer.pdf
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.loader import load_all_pdfs, split_into_chunks

PAPERS_DIR = "papers"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-all", action="store_true", help="Dump all chunks to all_chunks.txt")
    parser.add_argument("--search", type=str, help="Search for chunks containing this text")
    parser.add_argument("--source", type=str, help="Filter by source filename (e.g. 01_transformer.pdf)")
    parser.add_argument("--context", type=int, default=1, help="Show N neighboring chunks before/after")
    args = parser.parse_args()

    print("Loading and chunking papers...")
    docs = load_all_pdfs(PAPERS_DIR)
    chunks = split_into_chunks(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Total chunks: {len(chunks)}")

    if args.dump_all:
        output = project_root / "evaluation" / "all_chunks.txt"
        with open(output, "w", encoding="utf-8") as f:
            for i, c in enumerate(chunks):
                source = Path(c.metadata.get("source", "")).name
                page = c.metadata.get("page", -1)
                f.write(f"\n{'=' * 75}\n")
                f.write(f"[CHUNK #{i}]  source={source}  page={page}  length={len(c.page_content)}\n")
                f.write(f"{'=' * 75}\n")
                f.write(c.page_content)
                f.write("\n")
        print(f"All {len(chunks)} chunks dumped to: {output.relative_to(project_root)}")
        return

    if args.search:
        needle = args.search.lower()
        matches = []
        for i, c in enumerate(chunks):
            source = Path(c.metadata.get("source", "")).name
            if args.source and source != args.source:
                continue
            if needle in c.page_content.lower():
                matches.append(i)

        if not matches:
            print(f"No chunks contain: {args.search!r}")
            if args.source:
                print(f"(restricted to source={args.source})")
            return

        print(f"\nFound {len(matches)} chunk(s) containing {args.search!r}:")
        if args.source:
            print(f"(restricted to source={args.source})")

        for idx in matches:
            # Print with context
            start = max(0, idx - args.context)
            end = min(len(chunks), idx + args.context + 1)
            print(f"\n{'█' * 75}")
            print(f"█  MATCH at chunk #{idx}")
            print(f"█  Showing chunks #{start} to #{end - 1}")
            print(f"{'█' * 75}")
            for j in range(start, end):
                c = chunks[j]
                source = Path(c.metadata.get("source", "")).name
                page = c.metadata.get("page", -1)
                tag = "◀── MATCH" if j == idx else ""
                print(f"\n┌─ [#{j}] {source} p{page} (length={len(c.page_content)}) {tag}")
                for line in c.page_content.split("\n"):
                    print(f"│  {line}")
                print(f"└─")


if __name__ == "__main__":
    main()