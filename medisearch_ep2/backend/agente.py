# ─────────────────────────────────────────────────────────────
# agente.py  —  EP2: Pipeline RAG con planificación ReAct
#               y memoria de corto/largo plazo
#
# Cambios respecto a la EP1:
#   + Integra Planificador (ReAct): cada paso del pipeline
#     está gobernado por decisiones explícitas de planificacion.py.
#   + Integra MemoriaCortoplazo: historial de sesión incluido
#     en el prompt para mantener coherencia entre turnos.
#   + Integra MemoriaLargoplazo: contexto de interacciones
#     previas similares recuperado por similitud semántica.
#   + Acepta session_id como parámetro para multi-sesión.
#   + Retorna campo "razonamiento" con la traza ReAct completa.
# ─────────────────────────────────────────────────────────────

import os
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

from prompts import SYSTEM_PROMPT, FALLBACK_MSG, construir_prompt_usuario
from validacion import filtrar_por_score, tiene_evidencia, construir_contexto
from ingesta import buscar_pubmed
from memoria import MemoriaCortoplazo, MemoriaLargoplazo
from planificador import Planificador, Accion

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
TOP_K     = int(os.getenv("TOP_K_RESULTS", "4"))

# ── Inicializar LLM ────────────────────────────────────────────────────────
# temperature=0 - determinístico. En dominio médico la creatividad es riesgo.
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# ── Instancias globales de memoria y planificador ──────────────────────────
# Se inicializan una sola vez al importar el módulo.
memoria_cp  = MemoriaCortoplazo()
memoria_lp  = MemoriaLargoplazo()
planificador = Planificador()


def consultar(
    pregunta: str,
    nivel: str,
    vector_store: Chroma,
    session_id: str = "default"
) -> dict:
    """
    Pipeline RAG con ciclo ReAct, memoria dual y planificación adaptativa.

    Fases del ciclo (gobernadas por planificador.py):
      1. RECUPERAR         ChromaDB busca top-K fragmentos por similitud.
      2. BUSCAR_PUBMED     (condicional) Si score < umbral, consulta PubMed.
      3. VALIDAR_FUENTES   Verifica que los fragmentos tengan fuentes acreditadas.
      4. GENERAR_RESPUESTA GPT-4 genera respuesta citada con contexto RAG.
         o ACTIVAR_FALLBACK Respuesta segura determinística sin LLM.
      5. FINALIZADO        Actualiza ambas memorias y retorna resultado.

    Memoria integrada:
      - Corto plazo: historial de la sesión actual (últimos 10 turnos).
      - Largo plazo: interacciones previas similares de otras sesiones.

    Args:
        pregunta:     Consulta del usuario.
        nivel:        "lego" o "profesional".
        vector_store: ChromaDB inicializado.
        session_id:   ID de sesión para persistencia de memoria.

    Returns:
        dict con: respuesta, fuentes, tiene_evidencia, razonamiento, session_id
    """

    print(f"\n[AGENTE] Consulta: '{pregunta}' | Nivel: {nivel} | Sesión: {session_id}")

    # ── PASO 0: Inicializar plan ReAct ─────────────────────────────────────
    estado = planificador.iniciar(pregunta, nivel)

    # ── PASO 0b: Cargar contexto de memoria ────────────────────────────────
    # Corto plazo: historial de la sesión actual
    historial_sesion = memoria_cp.formatear_para_prompt(session_id)

    # Largo plazo: recuperar interacciones pasadas similares (umbral 0.80)
    contexto_lp = ""
    pasadas = memoria_lp.recuperar_similares(pregunta)
    if pasadas:
        estado.log(
            f"Memoria largo plazo: {len(pasadas)} interacción(es) similar(es) "
            f"recuperada(s) (similitud >= 0.80)."
        )
        contexto_lp = "\n".join(
            f"[Contexto previo | similitud {p['similitud']}]: {p['contenido'][:300]}"
            for p in pasadas
        )

    # ── PASO 1: Recuperación semántica (ChromaDB) ──────────────────────────
    resultados_raw = vector_store.similarity_search_with_score(
        query=pregunta,
        k=TOP_K
    )
    print(f"[AGENTE] Fragmentos recuperados: {len(resultados_raw)}")

    validos      = filtrar_por_score(resultados_raw)
    score_prom   = (
        sum(s for _, s in validos) / len(validos) if validos else 0.0
    )

    # ── Decisión del planificador tras recuperación ────────────────────────
    siguiente = planificador.tras_recuperacion(estado, score_prom, validos)

    # ── PASO 2 (condicional): Buscar en PubMed si score insuficiente ───────
    if siguiente == Accion.BUSCAR_PUBMED:
        texto_pubmed = buscar_pubmed(pregunta, max_resultados=5)
        encontro = bool(texto_pubmed and len(texto_pubmed) > 100)

        if encontro:
            # Agregar el abstract de PubMed al vector store temporalmente
            vector_store.add_texts(
                texts=[texto_pubmed[:2000]],
                metadatas=[{"source": "PubMed (búsqueda en tiempo real)", "page": "abstract"}]
            )
            # Reintentar recuperación con el nuevo contenido
            resultados_raw = vector_store.similarity_search_with_score(
                query=pregunta, k=TOP_K
            )
            validos   = filtrar_por_score(resultados_raw)
            score_prom = (
                sum(s for _, s in validos) / len(validos) if validos else 0.0
            )

        siguiente = planificador.tras_pubmed(estado, encontro and bool(validos))

    # ── PASO 3: Validación de fuentes ──────────────────────────────────────
    if siguiente == Accion.VALIDAR_FUENTES:
        fuentes_ok = tiene_evidencia(validos)
        siguiente  = planificador.tras_validacion(estado, fuentes_ok)

    # ── PASO 4a: FALLBACK ─────────────────────────────────────────────────
    if siguiente == Accion.ACTIVAR_FALLBACK:
        planificador.finalizar(estado, Accion.ACTIVAR_FALLBACK)
        # Guardar turno en memoria corto plazo (para trazabilidad)
        memoria_cp.guardar_turno(session_id, "usuario", pregunta)
        memoria_cp.guardar_turno(session_id, "agente", FALLBACK_MSG)

        return {
            "respuesta":       FALLBACK_MSG,
            "fuentes":         [],
            "tiene_evidencia": False,
            "razonamiento":    estado.razonamiento,
            "session_id":      session_id
        }

    # ── PASO 4b: Construir contexto y generar con LLM ─────────────────────
    contexto_rag, fuentes = construir_contexto(validos)

    # Enriquecer contexto con memoria de largo plazo si hay contexto previo
    if contexto_lp:
        contexto_rag = f"{contexto_lp}\n\n---\n\n{contexto_rag}"

    prompt_usuario = construir_prompt_usuario(
        consulta=pregunta,
        contexto_rag=contexto_rag,
        nivel=nivel,
        historial=historial_sesion      #  nuevo parámetro EP2
    )

    try:
        respuesta_llm = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_usuario)
        ])
        texto_respuesta = respuesta_llm.content
        print("[AGENTE] Respuesta generada por el LLM.")
    except Exception as e:
        print(f"[AGENTE] Error LLM: {e}")
        planificador.finalizar(estado, Accion.ACTIVAR_FALLBACK)
        return {
            "respuesta":       "Error interno al generar la respuesta. Inténtalo nuevamente.",
            "fuentes":         [],
            "tiene_evidencia": False,
            "razonamiento":    estado.razonamiento,
            "session_id":      session_id
        }

    # ── PASO 5: Actualizar memorias ────────────────────────────────────────
    planificador.finalizar(estado, Accion.GENERAR_RESPUESTA)

    # Corto plazo: guardar turno de esta interacción
    memoria_cp.guardar_turno(session_id, "usuario", pregunta)
    memoria_cp.guardar_turno(session_id, "agente", texto_respuesta)

    # Largo plazo: persistir interacción exitosa como embedding
    memoria_lp.guardar(
        session_id=session_id,
        consulta=pregunta,
        respuesta=texto_respuesta,
        metadata={"nivel": nivel, "fuentes": str(fuentes[:3])}
    )

    return {
        "respuesta":       texto_respuesta,
        "fuentes":         fuentes,
        "tiene_evidencia": True,
        "razonamiento":    estado.razonamiento,
        "session_id":      session_id
    }
