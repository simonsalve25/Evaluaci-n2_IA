# ─────────────────────────────────────────────────────────────
# main.py  —  EP2: API REST con soporte multi-sesión y endpoints
#             de memoria
#
# Cambios respecto a EP1:
#   + ConsultaEntrada incluye session_id opcional.
#   + ConsultaSalida incluye razonamiento (traza ReAct) y session_id.
#   + Nuevo endpoint GET /sesion/{session_id} — historial de sesión.
#   + Nuevo endpoint DELETE /sesion/{session_id} — limpiar sesión.
#   + Nuevo endpoint GET /memoria/stats — estadísticas de memoria LP.
# ─────────────────────────────────────────────────────────────

import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from ingesta import cargar_pdfs, dividir_documentos
from vector_store import crear_vector_store, cargar_vector_store, estado_vector_store
from agente import consultar, memoria_cp, memoria_lp

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./medisearch_db")
PDF_DIR     = "./articulos"

# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediSearch API",
    description="Motor de búsqueda médico con LLM, RAG, memoria dual y planificación ReAct — EP2",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"]
)

vector_store = None

@app.on_event("startup")
def inicializar_sistema():
    global vector_store
    if os.path.exists(CHROMA_PATH):
        print("[STARTUP] Cargando base vectorial existente...")
        vector_store = cargar_vector_store()
    else:
        print("[STARTUP] Base no encontrada. Procesando PDFs...")
        docs       = cargar_pdfs(PDF_DIR)
        fragmentos = dividir_documentos(docs)
        vector_store = crear_vector_store(fragmentos if fragmentos else [])


# ── Modelos Pydantic ───────────────────────────────────────────────────────

class ConsultaEntrada(BaseModel):
    pregunta:   str = Field(..., min_length=3, description="Consulta médica del usuario")
    nivel:      str = Field("lego", pattern="^(lego|profesional)$")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="ID de sesión (se genera automáticamente si no se provee)"
    )

class ConsultaSalida(BaseModel):
    respuesta:       str
    fuentes:         list[str]
    tiene_evidencia: bool
    razonamiento:    list[str]    # ← NUEVO EP2: traza ReAct
    session_id:      str          # ← NUEVO EP2


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Estado"])
def health_check():
    return {"status": "MediSearch v2.0 activo (EP2)", "version": "2.0.0"}


@app.get("/estado", tags=["Estado"])
def estado():
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")
    info = estado_vector_store(vector_store)
    info["memoria_largo_plazo"] = memoria_lp.total_almacenado()
    return info


@app.post("/consultar", response_model=ConsultaSalida, tags=["Consultas"])
def endpoint_consultar(consulta: ConsultaEntrada):
    """
    Endpoint principal con pipeline RAG + ReAct + memoria.
    Retorna la respuesta, fuentes, indicador de evidencia,
    traza de razonamiento del planificador y el session_id.
    """
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Base vectorial no inicializada")

    resultado = consultar(
        pregunta=consulta.pregunta,
        nivel=consulta.nivel,
        vector_store=vector_store,
        session_id=consulta.session_id
    )
    return resultado


@app.post("/reindexar", tags=["Administración"])
def reindexar():
    global vector_store
    docs       = cargar_pdfs(PDF_DIR)
    fragmentos = dividir_documentos(docs)
    vector_store = crear_vector_store(fragmentos)
    return {"mensaje": f"Re-indexación completada. Fragmentos: {len(fragmentos)}"}


# ── NUEVOS ENDPOINTS EP2 ───────────────────────────────────────────────────

@app.get("/sesion/{session_id}", tags=["Memoria"])
def obtener_sesion(session_id: str):
    """Retorna el historial de conversación de una sesión."""
    historial = memoria_cp.obtener_historial(session_id)
    return {"session_id": session_id, "turnos": len(historial), "historial": historial}


@app.delete("/sesion/{session_id}", tags=["Memoria"])
def limpiar_sesion(session_id: str):
    """Elimina el historial de conversación de una sesión."""
    memoria_cp.limpiar_sesion(session_id)
    return {"mensaje": f"Sesión {session_id} eliminada correctamente."}


@app.get("/memoria/stats", tags=["Memoria"])
def stats_memoria():
    """Estadísticas de ambas capas de memoria."""
    return {
        "memoria_largo_plazo_interacciones": memoria_lp.total_almacenado(),
        "descripcion": "Interacciones exitosas almacenadas como embeddings en ChromaDB"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
