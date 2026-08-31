# Transformer Paper RAG

A simple RAG application for asking questions about the *Attention Is All You Need* paper.

Built to explore **chunking, embeddings, vector search, retrieval, and LLM generation** using LangChain, ChromaDB, OpenAI, and Streamlit.

## Run

```bash
uv sync
uv run python -m indexing.indexing
uv run streamlit run app.py
```