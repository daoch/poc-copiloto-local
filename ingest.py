import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


DOCS_PATH = Path("documentos")
COLLECTION_NAME = "documentos_poc"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
)

collection = client.get_or_create_collection(name=COLLECTION_NAME)


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def reset_collection():
    existing_collections = [item.name for item in client.list_collections()]

    if COLLECTION_NAME in existing_collections:
        client.delete_collection(name=COLLECTION_NAME)

    return client.get_or_create_collection(name=COLLECTION_NAME)


def main():
    collection = reset_collection()

    files = list(DOCS_PATH.rglob("*.txt"))

    if not files:
        print("No se encontraron documentos .txt en la carpeta documentos/")
        return

    ids = []
    documents = []
    metadatas = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            chunk_id = f"{file_path.parent.name}_{file_path.stem}_{index}"

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "file_name": file_path.name,
                    "path": str(file_path),
                    "area": file_path.parent.name,
                }
            )

    embeddings = embedding_model.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Documentos procesados: {len(files)}")
    print(f"Chunks guardados: {len(ids)}")
    print("Ingesta completada en ChromaDB.")


if __name__ == "__main__":
    main()
