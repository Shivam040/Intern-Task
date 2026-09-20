# Agentic AI RAG Chatbot

A simple RAG chatbot built for the Appening Infotech AI Engineering Intern assignment.

The chatbot answers questions using only the provided Agentic AI ebook and returns:

- Answer
- Retrieved context chunks
- Retrieval score

Knowledge base: https://konverge.ai/pdf/Ebook-Agentic-AI.pdf

## Tech Stack

- Python
- LangGraph
- Pinecone
- FastAPI
- Ollama
- `nomic-embed-text` for embeddings
- `llama3.2:3b` for answer generation
- PyPDF

## Architecture

```text
PDF
 ↓
Extract text
 ↓
Chunk text
 ↓
nomic-embed-text
 ↓
Pinecone


User Question
 ↓
FastAPI
 ↓
LangGraph
 ↓
retrieve_context
 ↓
check_relevance
 ↙            ↘
relevant     irrelevant
 ↓               ↓
generate        fallback
 ↓               ↓
       response