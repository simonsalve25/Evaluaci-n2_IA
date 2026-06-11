# ─────────────────────────────────────────────────────────────
# memoria.py  —  EP2: Memoria de corto y largo plazo
#
# Módulo nuevo para la Evaluación Parcial N°2.
# Implementa dos capas de memoria:
#
#   MemoriaCortoplazo  — Historial de conversación por sesión.
#                        Almacenado en RAM (dict) con ventana de
#                        los últimos 10 turnos. Si se configura
#                        REDIS_URL en .env, persiste en Redis.
#
#   MemoriaLargoplazo  — Interacciones exitosas guardadas como
#                        embeddings en ChromaDB (colección
#                        separada: "agent_memory").
#                        Permite recuperar contexto de sesiones
#                        anteriores por similitud semántica.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import hashlib
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

REDIS_URL   = os.getenv("REDIS_URL", "")
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./medisearch_db")
WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW", "10"))   # turnos máximos en corto plazo


#  MEMORIA DE CORTO PLAZO


class MemoriaCortoplazo:
    """
    Gestiona el historial de conversación dentro de una sesión activa.

    Comportamiento:
    - Guarda los últimos WINDOW_SIZE (default 10) turnos por session_id.
    - Backend: Redis si REDIS_URL está configurado, RAM en caso contrario.
    - Cada turno = {"rol": "usuario"|"agente", "contenido": str}

    Uso:
        mem = MemoriaCortoplazo()
        mem.guardar_turno("ses1", "usuario", "¿Qué es la diabetes?")
        mem.guardar_turno("ses1", "agente",  "La diabetes es...")
        historial = mem.obtener_historial("ses1")
        contexto  = mem.formatear_para_prompt("ses1")
    """

    def __init__(self):
        self._local: dict[str, list] = {}
        self._redis = None

        if REDIS_URL:
            try:
                import redis
                self._redis = redis.from_url(REDIS_URL, decode_responses=True)
                self._redis.ping()
                print("[MEMORIA CP] Redis conectado:", REDIS_URL)
            except Exception as e:
                print(f"[MEMORIA CP] Redis no disponible ({e}). Usando RAM.")
                self._redis = None

    # ── clave ─────────────────────────────────────────────────
    @staticmethod
    def _clave(session_id: str) -> str:
        return f"medisearch:memoria:{session_id}"

    # ── leer historial ────────────────────────────────────────
    def obtener_historial(self, session_id: str) -> list[dict]:
        if self._redis:
            raw = self._redis.get(self._clave(session_id))
            return json.loads(raw) if raw else []
        return self._local.get(session_id, [])

    # ── guardar turno ─────────────────────────────────────────
    def guardar_turno(self, session_id: str, rol: str, contenido: str) -> None:
        """
        Añade un turno y mantiene solo los últimos WINDOW_SIZE.

        Args:
            session_id: ID único de la sesión.
            rol:        "usuario" o "agente".
            contenido:  Texto del mensaje.
        """
        historial = self.obtener_historial(session_id)
        historial.append({"rol": rol, "contenido": contenido})
        historial = historial[-WINDOW_SIZE:]          # ventana deslizante

        if self._redis:
            ttl = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
            self._redis.setex(
                self._clave(session_id),
                ttl,
                json.dumps(historial, ensure_ascii=False)
            )
        else:
            self._local[session_id] = historial

    # ── borrar sesión ─────────────────────────────────────────
    def limpiar_sesion(self, session_id: str) -> None:
        if self._redis:
            self._redis.delete(self._clave(session_id))
        else:
            self._local.pop(session_id, None)

    # ── formato para prompt ───────────────────────────────────
    def formatear_para_prompt(self, session_id: str) -> str:
        """
        Convierte el historial en texto plano para incluir en el prompt.
        Retorna 'Sin historial previo.' si la sesión está vacía.
        """
        historial = self.obtener_historial(session_id)
        if not historial:
            return "Sin historial previo en esta sesión."
        lineas = []
        for turno in historial:
            prefijo = "Usuario" if turno["rol"] == "usuario" else "Agente"
            lineas.append(f"{prefijo}: {turno['contenido']}")
        return "\n".join(lineas)



#  MEMORIA DE LARGO PLAZO


class MemoriaLargoplazo:
    """
    Almacena interacciones exitosas como vectores en ChromaDB.

    Sólo se persisten respuestas con evidencia (tiene_evidencia=True).
    Al recuperar, se busca por similitud semántica para enriquecer
    el contexto de nuevas consultas relacionadas.

    Usa la misma colección ChromaDB del proyecto pero en una
    colección separada llamada "agent_memory", para no mezclar
    los artículos médicos con el historial del agente.

    Uso:
        mem_lp = MemoriaLargoplazo()
        mem_lp.guardar(session_id, consulta, respuesta)
        pasadas = mem_lp.recuperar_similares(nueva_consulta)
    """

    _COLLECTION = "agent_memory"

    def __init__(self):
        try:
            import chromadb
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self._embedder = HuggingFaceEmbeddings(
                model_name=os.getenv("EMBEDDINGS_MODEL",
                                     "dmis-lab/biobert-base-cased-v1.2"),
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            cliente = chromadb.PersistentClient(path=CHROMA_PATH)
            try:
                self._col = cliente.get_collection(self._COLLECTION)
            except Exception:
                self._col = cliente.create_collection(
                    name=self._COLLECTION,
                    metadata={"hnsw:space": "cosine"}
                )
            print(f"[MEMORIA LP] ChromaDB lista. "
                  f"Interacciones almacenadas: {self._col.count()}")
            self._disponible = True
        except Exception as e:
            print(f"[MEMORIA LP] No disponible ({e}). Largo plazo desactivado.")
            self._disponible = False

    # ── guardar interacción ───────────────────────────────────
    def guardar(
        self,
        session_id: str,
        consulta: str,
        respuesta: str,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Guarda una interacción exitosa como embedding.
        Deduplica por hash de (session_id + consulta).
        """
        if not self._disponible:
            return

        doc_id  = hashlib.md5(f"{session_id}:{consulta}".encode()).hexdigest()
        vector  = self._embedder.embed_query(consulta)
        meta    = {
            "session_id": session_id,
            "consulta":   consulta[:200],
            **(metadata or {})
        }
        documento = f"P: {consulta}\nR: {respuesta}"

        # Eliminar versión anterior si existe (upsert manual)
        try:
            self._col.delete(ids=[doc_id])
        except Exception:
            pass

        self._col.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[documento],
            metadatas=[meta]
        )

    # ── recuperar similares ───────────────────────────────────
    def recuperar_similares(
        self,
        consulta: str,
        top_k: int = 2,
        umbral: float = 0.80
    ) -> list[dict]:
        """
        Busca interacciones pasadas similares a la consulta actual.

        Returns:
            Lista de dicts con 'contenido' y 'similitud'.
            Solo retorna resultados con similitud >= umbral.
        """
        if not self._disponible or self._col.count() == 0:
            return []

        vector = self._embedder.embed_query(consulta)
        resultados = self._col.query(
            query_embeddings=[vector],
            n_results=min(top_k, self._col.count()),
            include=["documents", "distances"]
        )

        salida = []
        for doc, dist in zip(
            resultados["documents"][0],
            resultados["distances"][0]
        ):
            similitud = round(1 - dist, 3)
            if similitud >= umbral:
                salida.append({"contenido": doc, "similitud": similitud})
        return salida

    # ── estadísticas ──────────────────────────────────────────
    def total_almacenado(self) -> int:
        return self._col.count() if self._disponible else 0
