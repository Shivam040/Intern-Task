from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):

    # PDF
    pdf_url: str = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    pdf_path: Path = BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"
    
    # Pinecone
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "agentic-ai-ebook"
    pinecone_namespace: str = "ebook-v1"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Local embeddings
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    ollama_base_url: str = "http://localhost:11434"

    # Retrieval
    top_k: int = 4
    relevance_threshold: float = 0.35

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Local LLM
    llm_model: str = "llama3.2:3b"

    # Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()