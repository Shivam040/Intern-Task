from functools import lru_cache
import requests
from pinecone import Pinecone
from app.config import settings


def embed_text(text: str) -> list[float]:
    response = requests.post(
        f"{settings.ollama_base_url}/api/embeddings",
        json={ "model": settings.embedding_model, "prompt": text,},
        timeout=60,
    )

    response.raise_for_status()
    return response.json()["embedding"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embed_text(text) for text in texts]


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not configured.")

    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache(maxsize=1)
def get_index():
    return get_pinecone_client().Index(settings.pinecone_index_name)


def retrieve(question: str) -> list[dict]:
    query_vector = embed_text(question)

    result = get_index().query(
        namespace=settings.pinecone_namespace,
        vector=query_vector,
        top_k=settings.top_k,
        include_metadata=True,
        include_values=False,
    )

    chunks = []

    for match in result.matches:
        metadata = match.metadata or {}

        chunks.append(
            {
                "text": str(metadata.get("text", "")),
                "page": int(metadata.get("page", 0)),
                "score": float(match.score or 0.0),
            }
        )

    return chunks


def generate_grounded_answer(
    question: str,
    context: list[dict],
) -> str:

    context_text = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in context
    )

    system_prompt = """ You are a question-answering assistant for the Agentic AI ebook. Answer ONLY using the provided context.
        Rules:
        - Do not use outside knowledge.
        - Do not invent information.
        - Base every claim on the supplied context.
        - Keep the answer concise and factual.
        - If the context is insufficient, respond exactly:

        I could not find sufficient information in the provided knowledge base.
        """.strip()

    user_prompt = f""" Context: {context_text} Question: {question} """.strip()

    response = requests.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt,},
                { "role": "user", "content": user_prompt,},
            ],
            "stream": False,
            "options": { "temperature": 0.1,},
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["message"]["content"].strip()