# ─────────────────────────────────────────────────────────────
# validacion.py
# Agente verificador: filtra fragmentos por score de similitud
# y construye el contexto formateado para el LLM
# ─────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

load_dotenv()

UMBRAL_SCORE = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))


def filtrar_por_score(
    resultados: list[tuple],
    umbral: float = UMBRAL_SCORE
) -> list[tuple]:
    """
    Filtra fragmentos recuperados por ChromaDB según score de similitud.

    Args:
        resultados: Lista de (Document, score) retornada por similarity_search_with_score.
        umbral:     Score mínimo aceptable (0.0 a 1.0). Default: 0.75.

    Returns:
        Lista filtrada con solo los fragmentos que superan el umbral.
    """
    validos = [(doc, score) for doc, score in resultados if score >= umbral]

    print(
        f"[VALIDACION] {len(validos)}/{len(resultados)} fragmentos superan "
        f"el umbral de score {umbral}"
    )
    return validos


def tiene_evidencia(resultados_validos: list) -> bool:
    """
    Determina si hay suficiente evidencia para generar una respuesta.
    Si retorna False → el agente activa el fallback.
    """
    return len(resultados_validos) > 0


def construir_contexto(resultados_validos: list[tuple]) -> tuple[str, list[str]]:
    """
    Construye el texto de contexto que se insertará en el prompt de usuario,
    y extrae la lista de fuentes para incluir en la respuesta final.

    Args:
        resultados_validos: Lista de (Document, score) ya filtrados.

    Returns:
        contexto (str):  Texto con fragmentos y etiquetas de fuente.
        fuentes (list):  Lista de nombres de fuentes únicas.
    """
    fragmentos_texto = []
    fuentes = []

    for doc, score in resultados_validos:
        fuente = doc.metadata.get("source", "Fuente desconocida")
        pagina = doc.metadata.get("page", "?")
        fuentes.append(fuente)
        fragmentos_texto.append(
            f"[Fuente: {fuente} | Página: {pagina} | Score: {score:.2f}]\n"
            f"{doc.page_content}"
        )

    contexto = "\n\n---\n\n".join(fragmentos_texto)
    fuentes_unicas = list(dict.fromkeys(fuentes))  # elimina duplicados manteniendo orden

    return contexto, fuentes_unicas
