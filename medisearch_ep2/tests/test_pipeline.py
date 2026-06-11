# ─────────────────────────────────────────────────────────────
# tests/test_pipeline.py  —  EP2
# Pruebas unitarias del sistema MediSearch (EP1 + nuevas EP2)
#
# Ejecutar:
#   pytest tests/ -v
#   pytest tests/ -v --html=reporte_pruebas.html
#
# Cobertura EP2:
#   - MemoriaCortoplazo: guardar, ventana, limpiar, formatear
#   - MemoriaLargoplazo: guardar y recuperar (mock)
#   - Planificador ReAct: todas las transiciones de estado
#   - prompts.py: historial incluido en prompt
# ─────────────────────────────────────────────────────────────

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from validacion import filtrar_por_score, tiene_evidencia, construir_contexto
from prompts import construir_prompt_usuario, FALLBACK_MSG, SYSTEM_PROMPT
from memoria import MemoriaCortoplazo
from planificador import Planificador, Accion



# Helpers


class MockDoc:
    def __init__(self, contenido, source="test_source.pdf", page=1):
        self.page_content = contenido
        self.metadata = {"source": source, "page": page}



# EP1: Pruebas de validacion.py  (sin cambios, regresión)

def test_filtrar_por_score_con_validos():
    doc1 = MockDoc("La diabetes tipo 2 está relacionada con resistencia a insulina.")
    doc2 = MockDoc("La hipertensión afecta el sistema cardiovascular.")
    doc3 = MockDoc("El cáncer de próstata es el más frecuente en hombres.")
    resultados = [(doc1, 0.91), (doc2, 0.80), (doc3, 0.60)]
    validos = filtrar_por_score(resultados, umbral=0.75)
    assert len(validos) == 2
    assert validos[0][1] == 0.91
    print("test_filtrar_por_score_con_validos: PASSED")


def test_filtrar_por_score_sin_validos():
    doc = MockDoc("Contenido irrelevante.")
    resultados = [(doc, 0.40), (doc, 0.55)]
    validos = filtrar_por_score(resultados, umbral=0.75)
    assert len(validos) == 0
    print("test_filtrar_por_score_sin_validos: PASSED")


def test_tiene_evidencia_true():
    doc = MockDoc("Evidencia científica.")
    assert tiene_evidencia([(doc, 0.90)]) is True
    print("test_tiene_evidencia_true: PASSED")


def test_tiene_evidencia_false():
    assert tiene_evidencia([]) is False
    print("test_tiene_evidencia_false: PASSED")


def test_construir_contexto_formato():
    doc = MockDoc("La resistencia a la insulina...", source="Smith_2021.pdf", page=4)
    contexto, fuentes = construir_contexto([(doc, 0.88)])
    assert "Smith_2021.pdf" in contexto
    assert "0.88" in contexto
    assert fuentes == ["Smith_2021.pdf"]
    print("test_construir_contexto_formato: PASSED")


def test_construir_contexto_fuentes_unicas():
    doc1 = MockDoc("Fragmento 1.", source="articulo_A.pdf")
    doc2 = MockDoc("Fragmento 2.", source="articulo_A.pdf")
    doc3 = MockDoc("Fragmento 3.", source="articulo_B.pdf")
    _, fuentes = construir_contexto([(doc1, 0.90), (doc2, 0.85), (doc3, 0.80)])
    assert len(fuentes) == 2
    print("test_construir_contexto_fuentes_unicas: PASSED")


def test_system_prompt_contiene_reglas_clave():
    assert "RAG" in SYSTEM_PROMPT
    assert "NUNCA inventes" in SYSTEM_PROMPT
    assert "Autor, Año, Revista" in SYSTEM_PROMPT
    assert "profesional de la salud" in SYSTEM_PROMPT
    print(" test_system_prompt_contiene_reglas_clave: PASSED")


def test_fallback_msg_menciona_profesional():
    assert "profesional de la salud" in FALLBACK_MSG
    print("test_fallback_msg_menciona_profesional: PASSED")


def test_prompt_usuario_lego():
    prompt = construir_prompt_usuario("¿Qué es la diabetes?", "Contexto.", "lego")
    assert "accesible" in prompt.lower()
    print("test_prompt_usuario_lego: PASSED")


def test_prompt_usuario_profesional():
    prompt = construir_prompt_usuario("¿Qué es la diabetes?", "Contexto.", "profesional")
    assert "técnico" in prompt.lower()
    print("test_prompt_usuario_profesional: PASSED")


def test_prompts_diferentes_segun_nivel():
    p_lego = construir_prompt_usuario("¿Qué es la hipertensión?", "ctx", "lego")
    p_prof = construir_prompt_usuario("¿Qué es la hipertensión?", "ctx", "profesional")
    assert p_lego != p_prof
    print("test_prompts_diferentes_segun_nivel: PASSED")



# EP2: Pruebas de MemoriaCortoplazo (memoria.py)


class TestMemoriaCortoplazo:
    """Pruebas de la memoria de corto plazo en modo RAM (sin Redis)."""

    def setup_method(self):
        self.mem = MemoriaCortoplazo()
        # Forzar modo RAM para pruebas sin Redis
        self.mem._redis = None
        self.mem._local = {}

    def test_guardar_y_recuperar_historial(self):
        """Debe guardar y recuperar turnos correctamente."""
        self.mem.guardar_turno("s1", "usuario", "¿Qué es la diabetes?")
        self.mem.guardar_turno("s1", "agente",  "La diabetes es...")
        hist = self.mem.obtener_historial("s1")
        assert len(hist) == 2
        assert hist[0]["rol"] == "usuario"
        assert hist[1]["rol"] == "agente"
        print("test_guardar_y_recuperar_historial: PASSED")

    def test_ventana_deslizante_limita_a_10(self):
        """Con más de 10 turnos, solo deben quedar los últimos 10."""
        for i in range(15):
            self.mem.guardar_turno("s2", "usuario", f"Pregunta {i}")
        hist = self.mem.obtener_historial("s2")
        assert len(hist) == 10
        assert "14" in hist[-1]["contenido"]
        print("test_ventana_deslizante_limita_a_10: PASSED")

    def test_sesion_vacia_retorna_lista_vacia(self):
        """Una sesión inexistente debe retornar lista vacía."""
        hist = self.mem.obtener_historial("inexistente")
        assert hist == []
        print("test_sesion_vacia_retorna_lista_vacia: PASSED")

    def test_limpiar_sesion(self):
        """Después de limpiar, el historial debe estar vacío."""
        self.mem.guardar_turno("s3", "usuario", "Hola")
        self.mem.limpiar_sesion("s3")
        assert self.mem.obtener_historial("s3") == []
        print("test_limpiar_sesion: PASSED")

    def test_formatear_historial_vacio(self):
        """Historial vacío debe indicar que no hay historial previo."""
        texto = self.mem.formatear_para_prompt("s4")
        assert "Sin historial" in texto
        print("test_formatear_historial_vacio: PASSED")

    def test_formatear_historial_con_turnos(self):
        """El historial formateado debe contener prefijos Usuario/Agente."""
        self.mem.guardar_turno("s5", "usuario", "Hola")
        self.mem.guardar_turno("s5", "agente",  "¿En qué te ayudo?")
        texto = self.mem.formatear_para_prompt("s5")
        assert "Usuario: Hola" in texto
        assert "Agente:" in texto
        print("test_formatear_historial_con_turnos: PASSED")

    def test_sesiones_independientes(self):
        """Dos sesiones distintas no deben compartir historial."""
        self.mem.guardar_turno("sA", "usuario", "Pregunta A")
        self.mem.guardar_turno("sB", "usuario", "Pregunta B")
        assert self.mem.obtener_historial("sA")[0]["contenido"] == "Pregunta A"
        assert self.mem.obtener_historial("sB")[0]["contenido"] == "Pregunta B"
        print("test_sesiones_independientes: PASSED")


# EP2: Pruebas del Planificador ReAct (planificador.py)


class TestPlanificador:
    """Pruebas de todas las transiciones del ciclo ReAct."""

    def setup_method(self):
        self.pl = Planificador()

    def test_iniciar_crea_estado_correcto(self):
        """El estado inicial debe tener acción RECUPERAR."""
        estado = self.pl.iniciar("¿Qué es la diabetes?", "lego")
        assert estado.accion_actual == Accion.RECUPERAR
        assert estado.consulta == "¿Qué es la diabetes?"
        assert len(estado.razonamiento) == 1
        print("test_iniciar_crea_estado_correcto: PASSED")

    def test_tras_recuperacion_score_alto_va_a_validar(self):
        """Score >= 0.75 con fragmentos → debe ir a VALIDAR_FUENTES."""
        estado  = self.pl.iniciar("consulta", "lego")
        accion  = self.pl.tras_recuperacion(estado, 0.88, [("doc", 0.88)])
        assert accion == Accion.VALIDAR_FUENTES
        print(" test_tras_recuperacion_score_alto_va_a_validar: PASSED")

    def test_tras_recuperacion_score_bajo_va_a_pubmed(self):
        """Score < 0.75 en primer intento → debe ir a BUSCAR_PUBMED."""
        estado = self.pl.iniciar("consulta", "lego")
        accion = self.pl.tras_recuperacion(estado, 0.50, [])
        assert accion == Accion.BUSCAR_PUBMED
        print("test_tras_recuperacion_score_bajo_va_a_pubmed: PASSED")

    def test_tras_recuperacion_score_bajo_segundo_intento_fallback(self):
        """Score < 0.75 después de ya haber consultado PubMed → FALLBACK."""
        estado = self.pl.iniciar("consulta", "lego")
        estado.intentos_pubmed = 1          # simular que ya se intentó PubMed
        accion = self.pl.tras_recuperacion(estado, 0.40, [])
        assert accion == Accion.ACTIVAR_FALLBACK
        print("test_tras_recuperacion_score_bajo_segundo_intento_fallback: PASSED")

    def test_tras_pubmed_con_resultados_reintenta_recuperacion(self):
        """PubMed con resultados → debe reintentar RECUPERAR."""
        estado = self.pl.iniciar("consulta", "lego")
        accion = self.pl.tras_pubmed(estado, encontro_resultados=True)
        assert accion == Accion.RECUPERAR
        print("test_tras_pubmed_con_resultados_reintenta_recuperacion: PASSED")

    def test_tras_pubmed_sin_resultados_activa_fallback(self):
        """PubMed sin resultados → FALLBACK."""
        estado = self.pl.iniciar("consulta", "lego")
        accion = self.pl.tras_pubmed(estado, encontro_resultados=False)
        assert accion == Accion.ACTIVAR_FALLBACK
        print("test_tras_pubmed_sin_resultados_activa_fallback: PASSED")

    def test_tras_validacion_ok_genera_respuesta(self):
        """Fuentes OK → GENERAR_RESPUESTA."""
        estado = self.pl.iniciar("consulta", "lego")
        accion = self.pl.tras_validacion(estado, fuentes_ok=True)
        assert accion == Accion.GENERAR_RESPUESTA
        print("test_tras_validacion_ok_genera_respuesta: PASSED")

    def test_tras_validacion_fail_activa_fallback(self):
        """Fuentes FAIL → ACTIVAR_FALLBACK."""
        estado = self.pl.iniciar("consulta", "lego")
        accion = self.pl.tras_validacion(estado, fuentes_ok=False)
        assert accion == Accion.ACTIVAR_FALLBACK
        print("test_tras_validacion_fail_activa_fallback: PASSED")

    def test_razonamiento_registra_todos_los_pasos(self):
        """El razonamiento debe registrar al menos un Thought por paso ejecutado."""
        estado = self.pl.iniciar("consulta", "lego")
        self.pl.tras_recuperacion(estado, 0.88, [("doc", 0.88)])
        self.pl.tras_validacion(estado, fuentes_ok=True)
        self.pl.finalizar(estado, Accion.GENERAR_RESPUESTA)
        # inicio + recuperacion + validacion + finalizar = 4 pasos mínimo
        assert len(estado.razonamiento) >= 4
        print(" test_razonamiento_registra_todos_los_pasos: PASSED")



# EP2: Prompts con historial (prompts.py)


def test_prompt_incluye_historial_cuando_existe():
    """Si hay historial, el prompt debe incluirlo."""
    historial = "Usuario: ¿Qué es la glucosa?\nAgente: La glucosa es..."
    prompt = construir_prompt_usuario("¿Y la insulina?", "ctx", "lego", historial)
    assert historial in prompt
    print("test_prompt_incluye_historial_cuando_existe: PASSED")


def test_prompt_sin_historial_no_incluye_seccion():
    """Sin historial, el prompt no debe incluir la sección de historial."""
    prompt = construir_prompt_usuario("¿Qué es la diabetes?", "ctx", "lego", "")
    assert "Historial de la conversación" not in prompt
    print("test_prompt_sin_historial_no_incluye_seccion: PASSED")


def test_prompt_historial_sin_sesion_previa_no_incluye():
    """Historial 'Sin historial previo.' no debe aparecer en el prompt."""
    prompt = construir_prompt_usuario(
        "consulta", "ctx", "lego", "Sin historial previo en esta sesión."
    )
    assert "Historial de la conversación" not in prompt
    print("test_prompt_historial_sin_sesion_previa_no_incluye: PASSED")
