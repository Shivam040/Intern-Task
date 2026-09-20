from argparse import ArgumentParser
from pathlib import Path

from app.config import settings
from app.ingest import download_pdf, ingest_pdf


def main() -> None:
    parser = ArgumentParser(description="Ingest the Agentic AI ebook into Pinecone.")
    parser.add_argument("--pdf", type=Path, default=settings.pdf_path)
    parser.add_argument("--url", default=settings.pdf_url)
    args = parser.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"PDF not found at {pdf_path}. Downloading it...")
        download_pdf(args.url, pdf_path)

    print(f"Ingesting: {pdf_path}")
    count = ingest_pdf(pdf_path)
    print(f"Done. Upserted {count} chunks into Pinecone index '{settings.pinecone_index_name}'.")


if __name__ == "__main__":
    main()
