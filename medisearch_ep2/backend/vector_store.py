# ─────────────────────────────────────────────────────────────
# vector_store.py
# Gestión de embeddings e indexación con ChromaDB + BioBERT
# ─────────────────────────────────────────────────────────────

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH      = os.getenv("CHROMA_DB_PATH", "./medisearch_db")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "dmis-lab/biobert-base-cased-v1.2")


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Inicializa el modelo de embeddings BioBERT.
    Pre-entrenado en PubMed y PMC → mejor similitud semántica en dominio médico.
    normalize_embeddings=True para que cosine similarity funcione correctamente.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def crear_vector_store(fragmentos: list) -> Chroma:
    """
    Crea la base vectorial desde una lista de fragmentos (Documents).
    Guarda en disco para no reprocesar en cada arranque.
    """
    embeddings = _get_embeddings()
    vector_store = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    vector_store.persist()
    total = vector_store._collection.count()
    print(f"[VECTOR STORE] Base creada con {total} fragmentos en '{CHROMA_PATH}'")
    return vector_store


def cargar_vector_store() -> Chroma:
    """
    Carga la base vectorial existente desde disco.
    Usar en arranques posteriores para evitar reprocesar PDFs.
    """
    embeddings = _get_embeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    total = vector_store._collection.count()
    print(f"[VECTOR STORE] Base cargada con {total} fragmentos desde '{CHROMA_PATH}'")
    return vector_store


def agregar_textos(textos: list[str], metadatas: list[dict], vector_store: Chroma) -> None:
    """
    Agrega nuevos textos (ej. abstracts de PubMed) a una base ya existente.

    Ejemplo:
        agregar_textos(
            textos=["Abstract del artículo..."],
            metadatas=[{"source": "PubMed:12345", "fecha": "2024"}],
            vector_store=vs
        )
    """
    vector_store.add_texts(texts=textos, metadatas=metadatas)
    vector_store.persist()
    print(f"[VECTOR STORE] Agregados {len(textos)} nuevos textos.")


def estado_vector_store(vector_store: Chroma) -> dict:
    """Retorna estadísticas de la base vectorial."""
    return {
        "total_fragmentos": vector_store._collection.count(),
        "modelo_embeddings": EMBEDDINGS_MODEL,
        "ruta": CHROMA_PATH
    }
