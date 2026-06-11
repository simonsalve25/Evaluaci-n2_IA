# ─────────────────────────────────────────────────────────────
# planificador.py  —  EP2: Planificación y toma de decisiones
#
# Módulo nuevo para la Evaluación Parcial N°2.
#
# Implementa el ciclo de razonamiento ReAct
# (Reasoning + Acting): el agente evalúa cada paso antes de
# ejecutar la siguiente herramienta, adaptando su
# comportamiento según el resultado anterior.
#
# Fases del ciclo:
#   THOUGHT     → Analiza el estado actual y decide qué hacer.
#   ACTION      → Ejecuta la herramienta seleccionada.
#   OBSERVATION → Interpreta el resultado y decide continuar.
#
# Estados posibles del plan:
#   INICIO → RECUPERAR → VERIFICAR_SCORE → VALIDAR_FUENTES
#          → GENERAR   o   FALLBACK
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()

UMBRAL_SCORE = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))


# ── Enumeración de acciones posibles ──────────────────────────

class Accion(str, Enum):
    RECUPERAR         = "RECUPERAR"          # Buscar en ChromaDB
    BUSCAR_PUBMED     = "BUSCAR_PUBMED"      # Consultar PubMed API
    VALIDAR_FUENTES   = "VALIDAR_FUENTES"    # Verificar fuentes acreditadas
    GENERAR_RESPUESTA = "GENERAR_RESPUESTA"  # Llamar al LLM
    ACTIVAR_FALLBACK  = "ACTIVAR_FALLBACK"   # Respuesta segura sin LLM
    FINALIZADO        = "FINALIZADO"         # Pipeline completo


# ── Estado del plan ───────────────────────────────────────────

@dataclass
class EstadoPlan:
    """
    Captura el estado completo del pipeline en cada paso del ciclo ReAct.
    Se pasa de herramienta en herramienta para que cada una pueda decidir
    si continuar, retroceder o activar el fallback.
    """
    consulta:          str
    nivel:             str
    accion_actual:     Accion            = Accion.RECUPERAR
    score_promedio:    float             = 0.0
    fragmentos_validos: list             = field(default_factory=list)
    fuentes_verificadas: bool            = False
    intentos_pubmed:   int               = 0
    razonamiento:      list[str]         = field(default_factory=list)  # trazabilidad ReAct

    def log(self, thought: str) -> None:
        """Registra un paso de razonamiento (Thought) para trazabilidad."""
        self.razonamiento.append(thought)
        print(f"[PLANIFICADOR][THOUGHT] {thought}")


# ── Planificador ReAct ────────────────────────────────────────

class Planificador:
    """
    Orquesta el pipeline MediSearch usando el patrón ReAct.

    Cada método decide_* evalúa el estado actual (Observation)
    y emite la siguiente Acción (Action), registrando el
    razonamiento (Thought) para trazabilidad completa.

    Flujo principal:
        RECUPERAR
            └─ score >= umbral  → VALIDAR_FUENTES
            └─ score < umbral, intentos < 1 → BUSCAR_PUBMED
            └─ score < umbral, intentos >= 1 → ACTIVAR_FALLBACK
        BUSCAR_PUBMED
            └─ encontró resultados → RECUPERAR (reintento con contexto)
            └─ sin resultados      → ACTIVAR_FALLBACK
        VALIDAR_FUENTES
            └─ fuentes OK   → GENERAR_RESPUESTA
            └─ fuentes FAIL → ACTIVAR_FALLBACK
        GENERAR_RESPUESTA / ACTIVAR_FALLBACK → FINALIZADO
    """

    def __init__(self):
        self.umbral = UMBRAL_SCORE

    def iniciar(self, consulta: str, nivel: str) -> EstadoPlan:
        """Crea el estado inicial del plan para una nueva consulta."""
        estado = EstadoPlan(consulta=consulta, nivel=nivel)
        estado.log(
            f"Nueva consulta recibida: '{consulta}' | Nivel: {nivel}. "
            f"Comenzando por recuperación semántica en ChromaDB."
        )
        return estado

    # ── Después de ChromaDB ────────────────────────────────────
    def tras_recuperacion(
        self,
        estado: EstadoPlan,
        score_promedio: float,
        fragmentos_validos: list
    ) -> Accion:
        """
        Evalúa el resultado de la búsqueda en ChromaDB.
        Decide si continuar, ir a PubMed, o activar fallback.
        """
        estado.score_promedio     = score_promedio
        estado.fragmentos_validos = fragmentos_validos

        if score_promedio >= self.umbral and fragmentos_validos:
            estado.log(
                f"Score promedio {score_promedio:.2f} >= umbral {self.umbral}. "
                f"{len(fragmentos_validos)} fragmentos válidos. "
                f"Procediendo a validación de fuentes."
            )
            estado.accion_actual = Accion.VALIDAR_FUENTES

        elif estado.intentos_pubmed < 1:
            estado.log(
                f"Score promedio {score_promedio:.2f} < umbral {self.umbral}. "
                f"Intentando complementar con PubMed API antes del fallback."
            )
            estado.accion_actual  = Accion.BUSCAR_PUBMED
            estado.intentos_pubmed += 1

        else:
            estado.log(
                f"Score {score_promedio:.2f} insuficiente y ya se consultó PubMed. "
                f"Activando fallback seguro."
            )
            estado.accion_actual = Accion.ACTIVAR_FALLBACK

        return estado.accion_actual

    # ── Después de PubMed ─────────────────────────────────────
    def tras_pubmed(
        self,
        estado: EstadoPlan,
        encontro_resultados: bool
    ) -> Accion:
        """
        Evalúa el resultado de la consulta a PubMed.
        Si encontró artículos, reintenta recuperación RAG con contexto enriquecido.
        """
        if encontro_resultados:
            estado.log(
                "PubMed retornó resultados. Reintentando recuperación RAG "
                "con contexto enriquecido."
            )
            estado.accion_actual = Accion.RECUPERAR
        else:
            estado.log(
                "PubMed sin resultados para esta consulta. "
                "Evidencia insuficiente en todas las fuentes → fallback."
            )
            estado.accion_actual = Accion.ACTIVAR_FALLBACK

        return estado.accion_actual

    # ── Después de validar fuentes ────────────────────────────
    def tras_validacion(
        self,
        estado: EstadoPlan,
        fuentes_ok: bool
    ) -> Accion:
        """
        Evalúa si las fuentes superaron la validación de acreditación.
        """
        estado.fuentes_verificadas = fuentes_ok

        if fuentes_ok:
            estado.log(
                "Fuentes validadas correctamente. "
                "Procediendo a generación de respuesta con GPT-4."
            )
            estado.accion_actual = Accion.GENERAR_RESPUESTA
        else:
            estado.log(
                "Fuentes no superaron la validación. "
                "No es seguro generar respuesta → activando fallback."
            )
            estado.accion_actual = Accion.ACTIVAR_FALLBACK

        return estado.accion_actual

    # ── Al finalizar ──────────────────────────────────────────
    def finalizar(self, estado: EstadoPlan, accion: Accion) -> None:
        estado.accion_actual = Accion.FINALIZADO
        estado.log(
            f"Pipeline completado. Acción final ejecutada: {accion.value}. "
            f"Total de pasos de razonamiento: {len(estado.razonamiento)}."
        )
