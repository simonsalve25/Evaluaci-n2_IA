// ─────────────────────────────────────────────────────────────
// app.js — MediSearch Frontend EP2
// EP2: session_id persistente + sección de razonamiento ReAct
// ─────────────────────────────────────────────────────────────

const API_URL = "http://localhost:8000";

// Generar o recuperar session_id para la pestaña actual
let SESSION_ID = sessionStorage.getItem("medisearch_session");
if (!SESSION_ID) {
  SESSION_ID = crypto.randomUUID();
  sessionStorage.setItem("medisearch_session", SESSION_ID);
}

// ── Referencias DOM ───────────────────────────────────────────
const elPregunta       = () => document.getElementById("pregunta");
const elNivel          = () => document.getElementById("nivel");
const elBtnText        = () => document.getElementById("btn-text");
const elBtnSpinner     = () => document.getElementById("btn-spinner");
const elBtn            = () => document.getElementById("btn-consultar");
const elBtnLimpiar     = () => document.getElementById("btn-limpiar");
const elResultado      = () => document.getElementById("resultado-section");
const elBadge          = () => document.getElementById("badge-evidencia");
const elBadgeText      = () => document.getElementById("badge-text");
const elRespuesta      = () => document.getElementById("respuesta-texto");
const elFuentesCard    = () => document.getElementById("fuentes-card");
const elFuentesLista   = () => document.getElementById("fuentes-lista");
const elRazonCard      = () => document.getElementById("razon-card");
const elRazonLista     = () => document.getElementById("razon-lista");
const elSessionLabel   = () => document.getElementById("session-label");
const elError          = () => document.getElementById("error-box");


// ── Inicialización ────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Mostrar session_id truncado
  elSessionLabel().textContent = `Sesión: ${SESSION_ID.slice(0, 8)}…`;

  elPregunta().addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) consultar();
  });
});


// ── Función principal ─────────────────────────────────────────
async function consultar() {
  const pregunta = elPregunta().value.trim();
  const nivel    = elNivel().value;

  if (!pregunta || pregunta.length < 3) {
    elPregunta().focus();
    return;
  }

  setLoading(true);
  ocultarResultado();

  try {
    const response = await fetch(`${API_URL}/consultar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pregunta, nivel, session_id: SESSION_ID })
    });

    if (!response.ok) throw new Error(`Error del servidor: ${response.status}`);

    const data = await response.json();
    mostrarResultado(data);

  } catch (error) {
    console.error("[MEDISEARCH] Error:", error);
    mostrarError();
  } finally {
    setLoading(false);
  }
}


// ── Limpiar sesión ────────────────────────────────────────────
async function limpiarSesion() {
  try {
    await fetch(`${API_URL}/sesion/${SESSION_ID}`, { method: "DELETE" });
    SESSION_ID = crypto.randomUUID();
    sessionStorage.setItem("medisearch_session", SESSION_ID);
    elSessionLabel().textContent = `Sesión: ${SESSION_ID.slice(0, 8)}…`;
    ocultarResultado();
    elPregunta().value = "";
    console.log("[MEDISEARCH] Sesión reiniciada:", SESSION_ID);
  } catch (e) {
    console.error("[MEDISEARCH] Error al limpiar sesión:", e);
  }
}


// ── Actualizar UI ─────────────────────────────────────────────
function mostrarResultado(data) {
  elRespuesta().textContent = data.respuesta;

  // Badge de evidencia
  const badge = elBadge();
  badge.classList.remove("badge-ok", "badge-err");
  if (data.tiene_evidencia) {
    badge.classList.add("badge-ok");
    elBadgeText().textContent = "Evidencia científica encontrada";
  } else {
    badge.classList.add("badge-err");
    elBadgeText().textContent = "Sin evidencia suficiente";
  }

  // Fuentes
  if (data.tiene_evidencia && data.fuentes?.length > 0) {
    elFuentesLista().innerHTML = data.fuentes
      .map(f => `<li>📎 ${f}</li>`).join("");
    elFuentesCard().classList.remove("oculto");
  } else {
    elFuentesCard().classList.add("oculto");
  }

  // Razonamiento ReAct (EP2)
  if (data.razonamiento?.length > 0) {
    elRazonLista().innerHTML = data.razonamiento
      .map((t, i) => `<li><span class="razon-num">T${i + 1}</span>${t}</li>`)
      .join("");
    elRazonCard().classList.remove("oculto");
  } else {
    elRazonCard().classList.add("oculto");
  }

  elResultado().classList.remove("oculto");
  elError().classList.add("oculto");
  elResultado().scrollIntoView({ behavior: "smooth", block: "start" });
}

function ocultarResultado() {
  elResultado().classList.add("oculto");
  elError().classList.add("oculto");
}

function mostrarError() {
  elError().classList.remove("oculto");
  elResultado().classList.add("oculto");
}

function setLoading(loading) {
  elBtn().disabled = loading;
  elBtnText().textContent = loading ? "Buscando evidencia..." : "Buscar evidencia";
  elBtnSpinner().classList.toggle("oculto", !loading);
}
