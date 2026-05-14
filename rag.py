import os

import chromadb
import requests
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "documentos_poc"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
)

collection = client.get_or_create_collection(name=COLLECTION_NAME)

SYSTEM_PROMPT = """
Eres un copiloto empresarial interno para un POC local.

Reglas:
1. Responde unicamente usando el contexto entregado.
2. No inventes informacion.
3. Si el contexto no contiene la respuesta, responde:
   "No encontre evidencia suficiente en los documentos disponibles."
4. Siempre incluye las fuentes utilizadas.
5. Responde en espanol claro y profesional.
6. Si la respuesta no esta explicitamente en el contexto, no la infieras como hecho.

Formato obligatorio:

Respuesta:
[...]

Fuentes:
- [...]

Confianza:
Alta / Media / Baja

Notas:
[...]
"""


def search_documents(question: str, n_results: int = 4):
    question_embedding = embedding_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
    )

    contexts = []

    if not results.get("documents") or not results["documents"][0]:
        return contexts

    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        contexts.append(
            {
                "text": document,
                "file_name": metadata.get("file_name", "desconocido"),
                "area": metadata.get("area", "desconocida"),
            }
        )

    return contexts


def ask_ollama(prompt: str):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()["response"]


def ask_copilot(question: str, role: str):
    contexts = search_documents(question)

    if not contexts:
        return {
            "answer": """
Respuesta:
No encontré evidencia suficiente en los documentos disponibles.

Fuentes:
- No aplica

Confianza:
Baja

Notas:
No se recuperaron documentos relevantes.
""".strip(),
            "sources": [],
        }

    context_text = "\n\n".join(
        [
            f"Fuente: {item['file_name']}\nÁrea: {item['area']}\nContenido:\n{item['text']}"
            for item in contexts
        ]
    )

    prompt = f"""
{SYSTEM_PROMPT}

Rol del usuario:
{role}

Contexto disponible:
{context_text}

Pregunta del usuario:
{question}

Responde usando solo el contexto disponible.
"""

    answer = ask_ollama(prompt)

    return {
        "answer": answer,
        "sources": contexts,
    }
