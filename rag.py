import os
import re

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
1. Responde solamente usando el contexto disponible.
2. No inventes informacion ni hagas inferencias no explicitadas.
3. Si el contexto no contiene la respuesta exacta, responde unicamente:
No encontre evidencia suficiente en los documentos disponibles.
4. No uses plantillas, corchetes, XML, markdown estructurado ni etiquetas.
5. Devuelve solo el cuerpo de la respuesta final, en espanol claro y breve.
"""

EMPTY_ANSWER = "No encontre evidencia suficiente en los documentos disponibles."


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

    distances = results.get("distances") or [[]]

    for index, (document, metadata) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        distance = None
        if distances and distances[0] and index < len(distances[0]):
            distance = distances[0][index]

        contexts.append(
            {
                "text": document,
                "file_name": metadata.get("file_name", "desconocido"),
                "area": metadata.get("area", "desconocida"),
                "distance": distance,
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


def build_context_text(contexts):
    return "\n\n".join(
        [
            f"Fuente: {item['file_name']}\nArea: {item['area']}\nContenido:\n{item['text']}"
            for item in contexts
        ]
    )


def clean_answer(answer: str):
    cleaned = answer.strip()
    markers = [
        "[...]",
        "<respuesta final>",
        "</respuesta final>",
        "<respuesta_final>",
        "</respuesta_final>",
        "Respuesta:",
        "Fuentes:",
        "Confianza:",
        "Notas:",
    ]

    for marker in markers:
        cleaned = cleaned.replace(marker, "")

    cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
    return cleaned.strip()


def tokenize(text: str):
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def select_cited_sources(contexts, normalized_answer: str):
    answer_tokens = tokenize(normalized_answer)
    cited_sources = []

    for item in contexts:
        area_token = item["area"].lower()
        file_tokens = tokenize(item["file_name"].replace(".txt", "").replace("-", "_"))
        area_hit = area_token in answer_tokens
        file_hit = bool(file_tokens.intersection(answer_tokens))

        if area_hit or file_hit:
            cited_sources.append(item)

    if cited_sources:
        return cited_sources

    return contexts[:1]


def estimate_confidence(normalized_answer: str, cited_sources):
    if normalized_answer == EMPTY_ANSWER:
        return "Baja"

    if len(cited_sources) == 1:
        return "Alta"

    return "Media"


def build_notes(normalized_answer: str, cited_sources):
    if normalized_answer == EMPTY_ANSWER:
        return "No se encontro evidencia suficiente en los documentos disponibles."

    if len(cited_sources) == 1:
        return "Respuesta basada en una fuente recuperada del corpus local."

    return "Respuesta basada en multiples fragmentos recuperados del corpus local."


def format_answer(normalized_answer: str, cited_sources):
    sources_text = "\n".join(
        [f"- {item['file_name']} ({item['area']})" for item in cited_sources]
    )
    confidence = estimate_confidence(normalized_answer, cited_sources)
    notes = build_notes(normalized_answer, cited_sources)

    return f"""Respuesta:
{normalized_answer}

Fuentes:
{sources_text}

Confianza:
{confidence}

Notas:
{notes}"""


def ask_copilot(question: str, role: str):
    contexts = search_documents(question)

    if not contexts:
        return {
            "answer": format_answer(EMPTY_ANSWER, [{"file_name": "No aplica", "area": "-"}]),
            "sources": [],
        }

    context_text = build_context_text(contexts)

    prompt = f"""
{SYSTEM_PROMPT}

Rol del usuario:
{role}

Contexto disponible:
{context_text}

Pregunta del usuario:
{question}
"""

    raw_answer = ask_ollama(prompt)
    normalized_answer = clean_answer(raw_answer) or EMPTY_ANSWER

    if normalized_answer != EMPTY_ANSWER and normalized_answer.endswith("."):
        normalized_answer = normalized_answer.strip()

    cited_sources = select_cited_sources(contexts, normalized_answer)

    return {
        "answer": format_answer(normalized_answer, cited_sources),
        "sources": contexts,
    }
