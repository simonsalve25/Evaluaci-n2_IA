# ─────────────────────────────────────────────────────────────
# ingesta.py
# Carga documentos PDF y los divide en fragmentos para ChromaDB
# ─────────────────────────────────────────────────────────────

import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def cargar_pdfs(directorio: str = "./articulos") -> list:
    """
    Carga todos los archivos PDF de un directorio.
    Retorna lista de Document (LangChain).
    """
    documentos = []

    if not os.path.exists(directorio):
        print(f"[INGESTA] Directorio '{directorio}' no encontrado.")
        return documentos

    archivos = [f for f in os.listdir(directorio) if f.endswith(".pdf")]

    if not archivos:
        print(f"[INGESTA] No se encontraron PDFs en '{directorio}'.")
        return documentos

    for archivo in archivos:
        ruta = os.path.join(directorio, archivo)
        try:
            loader = PyPDFLoader(ruta)
            docs = loader.load()
            documentos.extend(docs)
            print(f"[INGESTA] Cargado: {archivo} ({len(docs)} páginas)")
        except Exception as e:
            print(f"[INGESTA] Error al cargar {archivo}: {e}")

    return documentos


def dividir_documentos(documentos: list) -> list:
    """
    Divide documentos en fragmentos manejables para el vector store.
    chunk_size=500, chunk_overlap=50 para no perder contexto entre fragmentos.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    fragmentos = splitter.split_documents(documentos)
    print(f"[INGESTA] Total de fragmentos generados: {len(fragmentos)}")
    return fragmentos


def buscar_pubmed(termino: str, max_resultados: int = 10) -> str:
    """
    Busca artículos en PubMed vía API NCBI E-utilities.
    Retorna los abstracts como texto plano.

    Ejemplo:
        texto = buscar_pubmed("diabetes tipo 2 insulin resistance", max_resultados=5)
    """
    # Paso 1: obtener IDs de artículos
    url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params_search = {
        "db": "pubmed",
        "term": termino,
        "retmax": max_resultados,
        "retmode": "json"
    }

    try:
        r = requests.get(url_search, params=params_search, timeout=10)
        r.raise_for_status()
        ids = r.json()["esearchresult"]["idlist"]
    except Exception as e:
        print(f"[PUBMED] Error en búsqueda: {e}")
        return ""

    if not ids:
        print(f"[PUBMED] Sin resultados para: {termino}")
        return ""

    # Paso 2: obtener abstracts
    url_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params_fetch = {
        "db": "pubmed",
        "id": ",".join(ids),
        "rettype": "abstract",
        "retmode": "text"
    }

    try:
        r2 = requests.get(url_fetch, params=params_fetch, timeout=15)
        r2.raise_for_status()
        print(f"[PUBMED] Recuperados {len(ids)} artículos para: {termino}")
        return r2.text
    except Exception as e:
        print(f"[PUBMED] Error al obtener abstracts: {e}")
        return ""
