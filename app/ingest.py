import hashlib
from pathlib import Path
from time import sleep

import requests
from pinecone import ServerlessSpec
from pypdf import PdfReader

from app.config import settings
from app.rag import embed_texts, get_pinecone_client


def download_pdf(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def extract_chunks(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    chunks: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if not text:
            continue

        step = max(1, settings.chunk_size - settings.chunk_overlap)
        page_chunks = [
            text[start : start + settings.chunk_size]
            for start in range(0, len(text), step)
            if text[start : start + settings.chunk_size].strip()
        ]

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            digest = hashlib.sha1(
                f"{page_number}:{chunk_number}:{chunk_text}".encode("utf-8")
            ).hexdigest()[:12]
            chunks.append(
                {
                    "id": f"p{page_number}-c{chunk_number}-{digest}",
                    "text": chunk_text,
                    "page": page_number,
                    "chunk": chunk_number,
                }
            )
    return chunks


def ensure_index() -> None:
    pc = get_pinecone_client()
    if pc.has_index(settings.pinecone_index_name):
        return

    pc.create_index(
        name=settings.pinecone_index_name,
        vector_type="dense",
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )

    # Newly-created serverless indexes can take a moment to become available.
    for _ in range(30):
        description = pc.describe_index(settings.pinecone_index_name)
        status = getattr(description, "status", None)
        ready = getattr(status, "ready", None) if status is not None else None
        if ready is True or (isinstance(status, dict) and status.get("ready")):
            return
        sleep(1)


def ingest_pdf(pdf_path: Path) -> int:
    ensure_index()
    chunks = extract_chunks(pdf_path)
    if not chunks:
        raise ValueError("No extractable text was found in the PDF.")

    index = get_pinecone_client().Index(settings.pinecone_index_name)
    batch_size = 64

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_texts([item["text"] for item in batch])

        records = [
            {
                "id": item["id"],
                "values": vector,
                "metadata": {
                    "text": item["text"],
                    "page": item["page"],
                    "chunk": item["chunk"],
                    "source": pdf_path.name,
                },
            }
            for item, vector in zip(batch, vectors, strict=True)
        ]

        index.upsert(vectors=records, namespace=settings.pinecone_namespace)

    return len(chunks)
