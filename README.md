# Agentic AI RAG Chatbot

A simple RAG chatbot built for the Appening Infotech AI Engineering Intern assignment.

The chatbot answers questions using only the provided Agentic AI ebook and returns:

- Final answer
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
OFFLINE INGESTION

PDF
 ↓
Extract text
 ↓
Chunk text
 ↓
nomic-embed-text
 ↓
Pinecone


RUNTIME RAG

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
```

The PDF ingestion is kept separate from LangGraph because it is a one-time preprocessing step.

LangGraph handles the runtime RAG workflow and conditional routing.

## Project Structure

```text
agentic-ai-rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py
│   ├── ingest.py
│   ├── main.py
│   ├── rag.py
│   └── schemas.py
├── data/
├── scripts/
│   └── ingest_pdf.py
├── tests/
│   └── test_graph_logic.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shivam040/Intern-Task.git
cd Intern-Task
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install Ollama models

Make sure Ollama is installed and running.

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Verify:

```bash
ollama list
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-ebook-768
PINECONE_NAMESPACE=ebook-v1
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
LLM_MODEL=llama3.2:3b

TOP_K=4
RELEVANCE_THRESHOLD=0.60

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

Do not commit your `.env` file or API keys.

## Ingest the Ebook

Run:

```bash
python -m scripts.ingest_pdf
```

The script extracts the PDF text, creates chunks, generates local embeddings using Ollama, and stores the vectors with metadata in Pinecone.

## Run the API

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## API

### POST `/chat`

Request:

```json
{
  "question": "What is Agentic AI?"
}
```

Example response:

```json
{
  "answer": "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
  "context": [
    {
      "text": "A Journey into the Heart of Autonomous Intelligence...",
      "page": 18,
      "score": 0.8299
    }
  ],
  "score": 0.8615
}
```

If the retrieved context is not relevant enough, LangGraph routes to a fallback response:

```text
I could not find sufficient information in the provided knowledge base.
```

## Sample Queries

- What is Agentic AI?
- How is Agentic AI different from traditional AI?
- What are the key components of an Agentic AI system?
- How do autonomous agents make decisions?
- What business applications of Agentic AI are discussed in the ebook?
- What are the benefits of Agentic AI for enterprises?

Out-of-domain test:

```text
Who is the Prime Minister of India?
```

Expected response:

```text
I could not find sufficient information in the provided knowledge base.
```

## Tests

Run:

```bash
python -m pytest
```

Current result:

```text
3 passed
```
