# ─────────────────────────────────────────────────────────────
# prompts.py  —  EP2: Prompts con soporte de historial de sesión
#
# Cambios respecto a EP1:
#   + construir_prompt_usuario acepta parámetro historial
#     para incluir el contexto de la sesión activa en el prompt.
# ─────────────────────────────────────────────────────────────


# ── PROMPT DE SISTEMA ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres MediSearch, un asistente médico especializado que solo responde \
usando evidencia científica verificada. Sigue estas reglas sin excepción:

1. Basa TODAS tus respuestas únicamente en los documentos científicos recuperados del \
sistema RAG que se incluyen en el contexto.
2. Cita SIEMPRE el artículo o estudio en formato (Autor, Año, Revista) al final de cada \
afirmación que hagas.
3. Usa lenguaje claro y accesible para usuarios sin formación médica (nivel lego). \
Si el usuario es profesional, usa terminología técnica precisa.
4. Si no existe evidencia suficiente en las fuentes disponibles, responde exactamente: \
"No encontré información científica verificada para esta consulta. \
Consulta a un profesional de la salud."
5. NUNCA inventes información, datos estadísticos, ni uses fuentes no verificadas.
6. Si la pregunta es ambigua o incompleta, solicita clarificación antes de responder.
7. No emitas diagnósticos médicos ni recomendaciones de tratamiento específicas. \
Informa, no prescribes.
8. Si el historial de conversación incluye preguntas anteriores relacionadas, \
mantén coherencia con tus respuestas previas."""


# ── MENSAJE DE FALLBACK ────────────────────────────────────────────────────

FALLBACK_MSG = (
    "No encontré información científica verificada para esta consulta. "
    "Consulta a un profesional de la salud."
)


# ── CONSTRUCTOR DE PROMPT DE USUARIO ──────────────────────────────────────

def construir_prompt_usuario(
    consulta: str,
    contexto_rag: str,
    nivel: str = "lego",
    historial: str = ""           # ← NUEVO EP2
) -> str:
    """
    Construye el prompt de usuario dinámicamente.

    EP2: incorpora el historial de la sesión para que el LLM
    mantenga coherencia en conversaciones con múltiples turnos.

    Args:
        consulta:     Pregunta original del usuario.
        contexto_rag: Fragmentos científicos recuperados (+ contexto largo plazo).
        nivel:        "lego" o "profesional".
        historial:    Historial de la sesión actual (MemoriaCortoplazo). Vacío = primer turno.

    Returns:
        str: Prompt completo listo para enviar al LLM como HumanMessage.
    """
    estilo = (
        "simple y accesible para una persona sin conocimientos médicos"
        if nivel == "lego"
        else "técnico y preciso, apropiado para un profesional de la salud"
    )

    # Sección de historial (solo se incluye si hay turnos previos)
    seccion_historial = ""
    if historial and historial != "Sin historial previo en esta sesión.":
        seccion_historial = f"""
Historial de la conversación actual (para mantener coherencia):
─────────────────────────────────────────────────────
{historial}
─────────────────────────────────────────────────────
"""

    return f"""Perfil del usuario: {nivel.upper()}
{seccion_historial}
Documentos científicos disponibles (fuentes verificadas del sistema RAG):
─────────────────────────────────────────────────────
{contexto_rag}
─────────────────────────────────────────────────────

Pregunta: {consulta}

Instrucción: Responde usando EXCLUSIVAMENTE los documentos anteriores. \
Usa un lenguaje {estilo}. \
Incluye las citas bibliográficas al final de cada afirmación en formato (Autor, Año, Revista). \
Si hay historial de conversación, mantén coherencia con respuestas anteriores."""
