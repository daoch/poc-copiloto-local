import os

import chromadb
import requests
from chromadb.errors import NotFoundError
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

SYSTEM_PROMPT = """
Eres un copiloto empresarial interno para un POC local.

Reglas:
1. Responde unicamente usando el contexto entregado.
2. No inventes informacion.
3. Si el contexto no contiene la respuesta, responde exactamente: No encontre evidencia suficiente en los documentos disponibles.
4. Siempre incluye las fuentes utilizadas.
5. Responde en espanol claro y profesional.
6. Si la respuesta no esta explicitamente en el contexto, no la infieras como hecho.
7. No escribas corchetes de ejemplo como [...] ni texto de plantilla.

Devuelve siempre este formato final, reemplazando cada seccion con contenido real:
Respuesta:
<respuesta final>

Fuentes:
- <fuente 1>

Confianza:
<Alta, Media o Baja>

Notas:
<observaciones breves>
"""

EMPTY_RESPONSE = """Respuesta:
No encontre evidencia suficiente en los documentos disponibles.

Fuentes:
- No aplica

Confianza:
Baja

Notas:
No se recuperaron documentos relevantes o la coleccion todavia no fue indexada.
""".strip()


def get_collection():
    return client.get_or_create_collection(name=COLLECTION_NAME)


def search_documents(question: str, n_results: int = 4):
    question_embedding = embedding_model.encode([question]).tolist()[0]

    try:
        collection = get_collection()
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results,
        )
    except NotFoundError:
        return []

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
    return response.json()["response"].strip()


def normalize_answer(answer: str, contexts):
    cleaned = answer.strip()

    if "[...]" in cleaned:
        cleaned = cleaned.replace("[...]", "").strip()

    if not cleaned:
        return EMPTY_RESPONSE

    if "Respuesta:" not in cleaned:
        sources = "\n".join(
            [f"- {item['file_name']} ({item['area']})" for item in contexts]
        )
        return f"""Respuesta:
{cleaned}

Fuentes:
{sources}

Confianza:
Media

Notas:
Respuesta normalizada por la aplicacion para mostrar el contenido sin plantilla.
""".strip()

    return cleaned


def ask_copilot(question: str, role: str):
    contexts = search_documents(question)

    if not contexts:
        return {"answer": EMPTY_RESPONSE, "sources": []}

    context_text = "\n\n".join(
        [
            f"Fuente: {item['file_name']}\nArea: {item['area']}\nContenido:\n{item['text']}"
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
    return {"answer": normalize_answer(answer, contexts), "sources": contexts}
