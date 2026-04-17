# Paper-RAG: Citation-Grounded Academic QA

A RAG (Retrieval-Augmented Generation) system for academic paper question answering, with hybrid retrieval, reranking, and citation-grounded answer generation.

## Current Status

🚧 Work in progress — Day 2 of 21

## Features (so far)

- [x] PDF loading and chunking
- [x] OpenAI embedding + Chroma vector store
- [x] Semantic retrieval
- [x] Citation-grounded answer generation
- [ ] Multi-paper support
- [ ] Hybrid retrieval (BM25 + dense)
- [ ] Reranking
- [ ] Evaluation set and error analysis
- [ ] Streamlit UI

## Tech Stack

- Python 3.12
- LangChain
- OpenAI (gpt-4o-mini, text-embedding-3-small)
- Chroma vector database

## Setup

1. Install dependencies:

    pip install langchain langchain-openai langchain-community chromadb pypdf

2. Set your OpenAI API key as an environment variable:

    Windows: setx OPENAI_API_KEY "sk-your-key-here"
    Mac/Linux: export OPENAI_API_KEY="sk-your-key-here"

3. Run:

    python day2_rag.py

## Author

Guangwen Xiong