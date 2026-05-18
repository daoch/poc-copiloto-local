import os
import re
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


def normalize_text(text: str):
    return text.replace("\r\n", "\n").strip()


def split_header_and_body(text: str):
    parts = re.split(r"\n\s*\n", normalize_text(text), maxsplit=1)

    if len(parts) == 1:
        return "", parts[0]

    header, body = parts
    return header.strip(), body.strip()


def split_body_blocks(body: str):
    raw_blocks = re.split(r"\n\s*\n", body)
    blocks = []
    current_list = []

    for raw_block in raw_blocks:
        block = raw_block.strip()
        if not block:
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        is_list_block = all(re.match(r"^(\d+[\.\)]|[-*])\s+", line) for line in lines)

        if is_list_block:
            current_list.extend(lines)
            continue

        if current_list:
            blocks.append("\n".join(current_list))
            current_list = []

        blocks.append("\n".join(lines))

    if current_list:
        blocks.append("\n".join(current_list))

    return blocks


def split_large_block(prefix: str, block: str, chunk_size: int):
    available = max(chunk_size - len(prefix), 300)
    sentences = re.split(r"(?<=[\.\:\;\?\!])\s+", block)
    chunks = []
    current = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate_length = current_length + len(sentence) + (1 if current else 0)

        if current and candidate_length > available:
            chunks.append(prefix + " ".join(current))
            current = [sentence]
            current_length = len(sentence)
            continue

        if len(sentence) > available:
            if current:
                chunks.append(prefix + " ".join(current))
                current = []
                current_length = 0

            for start in range(0, len(sentence), available):
                piece = sentence[start : start + available].strip()
                if piece:
                    chunks.append(prefix + piece)
            continue

        current.append(sentence)
        current_length = candidate_length

    if current:
        chunks.append(prefix + " ".join(current))

    return chunks


def chunk_blocks(header: str, blocks, chunk_size: int = 900):
    prefix = f"{header}\n\n" if header else ""
    chunks = []
    current_parts = []
    current_length = len(prefix)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        candidate_length = current_length + len(block) + (2 if current_parts else 0)

        if current_parts and candidate_length > chunk_size:
            chunks.append(prefix + "\n\n".join(current_parts))
            current_parts = [block]
            current_length = len(prefix) + len(block)
            continue

        if len(prefix) + len(block) > chunk_size and not current_parts:
            chunks.extend(split_large_block(prefix, block, chunk_size))
            current_length = len(prefix)
            continue

        current_parts.append(block)
        current_length = candidate_length

    if current_parts:
        chunks.append(prefix + "\n\n".join(current_parts))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def chunk_text(text: str, chunk_size: int = 900):
    header, body = split_header_and_body(text)

    if not body:
        return [header] if header else []

    blocks = split_body_blocks(body)

    if not blocks:
        return [normalize_text(text)]

    return chunk_blocks(header, blocks, chunk_size=chunk_size)


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
