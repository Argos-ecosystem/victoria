import os
import time
import json
import logging
import random
import re
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
import hashlib
import openai

# ======================
# CONFIG & SETUP
# ======================

load_dotenv()

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VictoriaWorker")

# Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_EVENTS = os.getenv("MONGO_DB_EVENTS", "omniguard")
MONGO_DB_CACHE = os.getenv("MONGO_DB_CACHE", "victoria")
EVENTS_COLL = os.getenv("MONGO_COLL_NAME", "events")
CACHE_COLL = "victoria_cache"
PROMPT_ANALYSIS = os.getenv("PROMPT_ANALYSIS", "Eres un analista de eventos; resume los siguientes grupos de eventos de forma concisa, directa y sin explicaciones.")

# Modelos por tipo
MODEL_ACTUAL = os.getenv("MODEL_ACTUAL", "gpt-4o")
MODEL_TRES   = os.getenv("MODEL_TRES",   "gpt-4o")
MODEL_DIA    = os.getenv("MODEL_DIA",    "gpt-4o")

try:
    mongo = MongoClient(MONGO_URI)
    db_events = mongo[MONGO_DB_EVENTS]
    db_cache  = mongo[MONGO_DB_CACHE]
    col_events = db_events[EVENTS_COLL]
    col_cache  = db_cache[CACHE_COLL]
    logger.info(f"Connected to MongoDB. Events DB: {MONGO_DB_EVENTS}, Cache DB: {MONGO_DB_CACHE}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    exit(1)

if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    logger.error("OPENAI_API_KEY not found. LLM analysis will fail.")

# ======================
# HELPERS (Migrated)
# ======================

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s) # Keep alphanumeric and spaces
    return s.strip()

def fingerprint(text: str) -> str:
    """Creates a deterministic hash for a given text."""
    if not text:
        return None
    return hashlib.sha256(normalize_text(text).encode('utf-8')).hexdigest()

def group_similar_events(events, threshold=0.95):
    if not events:
        return []
    groups = []
    seen = {}
    for evt in events:
        txt = evt.get("text", "") or evt.get("msg", "") or evt.get("description", "")
        if not txt:
            continue
        
        # Truncar para ahorrar tokens y memoria
        txt = txt[:500]

        fp = fingerprint(txt)
        if not fp: # Should not happen if txt is not empty
            continue

        if fp in seen:
            seen[fp]["count"] += 1
        else:
            grupo = {
                "sample_text": txt,
                "count": 1
            }
            groups.append(grupo)
            seen[fp] = grupo

    return groups

def read_last_event():
    """
    Obtiene el último registro crudo de la colección general.
    """
    doc = col_events.find().sort("timestamp", -1).limit(1)
    ultimo = next(doc, None)

    if not ultimo:
        return None

    ts = ultimo.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

    if isinstance(ts, datetime):
        ultimo["timestamp"] = ts.isoformat()

    ultimo.pop("_id", None)
    return ultimo

def limpiar_para_alexa(texto):
    if not texto:
        return "Sin información."

    texto = re.sub(r"\*\*(.*?)\*\*", r"\1", texto)
    texto = re.sub(r"\*(.*?)\*", r"\1", texto)
    texto = texto.replace("<", "").replace(">", "")
    texto = texto.replace("&", " y ")
    texto = texto.replace('"', "").replace("'", "")
    texto = texto.replace("\n- ", ". ").replace("\n* ", ". ")
    texto = texto.replace("\n1. ", ". ")
    texto = re.sub(r"\n+", " ", texto)

    return texto.strip()


# =======================
# CACHE
# =======================

def leer_cache(tipo):
    return col_cache.find_one({"tipo": tipo})


def guardar_cache(tipo, texto, events_hash):
    col_cache.update_one(
        {"tipo": tipo},
        {
            "$set": {
                "tipo": tipo,
                "texto": texto,
                "events_hash": events_hash,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        },
        upsert=True
    )

# =======================
# LLM
# =======================

def analizar(eventos, modelo):
    if not eventos:
        return "No hubo eventos relevantes en este periodo."
    
    if not client:
        return "Error: cliente de OpenAI no inicializado."

    # Ordenar por importancia (count) y limitar a top 100 para no explotar el context window
    if len(eventos) > 0 and "count" in eventos[0]:
        eventos.sort(key=lambda x: x["count"], reverse=True)
        eventos = eventos[:100]
    elif len(eventos) > 100:
        # Fallback si no son grupos agrupados, crude slice
        eventos = eventos[-100:]

    logger.info(f"Analizando {len(eventos)} grupos de eventos con {modelo}.")

    try:
        response = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": PROMPT_ANALYSIS},
                {"role": "user", "content": f"Eventos agrupados:\n{json.dumps(eventos, ensure_ascii=False)}"}
            ]
        )
        texto = response.choices[0].message.content
        return limpiar_para_alexa(texto)
    except Exception as e:
        logger.error(f"Error llamando a OpenAI API: {e}")
        return "Error procesando eventos."

# ======================
# WORKER LOGIC
# ======================

# =======================
# LOGICA DE RE-CÁLCULO
# =======================

def procesar_si_cambia(tipo, eventos, modelo):
    if not eventos:
        logger.warning(f"⚠️ No hay eventos para procesar para '{tipo}'. Saltando.")
        return

    # Usar sort_keys=True para garantizar hash determinista
    events_hash = hashlib.sha256(json.dumps(eventos, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    cache_prev = leer_cache(tipo)

    if cache_prev and cache_prev.get("events_hash") == events_hash:
        logger.info(f"✅ {tipo.upper()} sin cambios (hash: {events_hash[:7]}). Usando caché.")
        return

    logger.info(f"🔄 {tipo.upper()} con cambios detectados (hash: {events_hash[:7]}) → recalculando con {modelo}...")
    texto = analizar(eventos, modelo)
    
    logger.info(f"📝 Resultado {tipo.upper()}: {texto[:100]}...")
    
    guardar_cache(tipo, texto, events_hash)

    logger.info(f"🟢 {tipo.upper()} OK.")

def procesar_actual_desde_general():
    ultimo = read_last_event()
    if not ultimo:
        logger.warning("🔴 ACTUAL sin eventos en la colección general → Guardando estado vacío.")
        texto = "No hay eventos registrados aún."
        events_hash = "no_events"
        guardar_cache("actual", texto, events_hash)
        logger.info("🟢 ACTUAL (vacío) OK.")
        return

    texto = ultimo.get("text") or ultimo.get("msg") or ultimo.get("mensaje") or ultimo.get("description")
    if not texto:
        texto = json.dumps(ultimo, ensure_ascii=False)

    events_hash = hashlib.sha256(json.dumps(ultimo, ensure_ascii=False, default=str, sort_keys=True).encode('utf-8')).hexdigest()
    
    logger.info("🟣 ACTUAL se toma del último registro en la colección general.")
    texto_limpio = limpiar_para_alexa(texto)
    
    logger.info(f"📝 Resultado ACTUAL: {texto_limpio}")

    guardar_cache("actual", texto_limpio, events_hash)

    logger.info("🟢 ACTUAL OK.")

def read_last_n_events(n):
    # Obtener los últimos N eventos (orden descendente primero)
    docs = col_events.find().sort("timestamp", -1).limit(n)

    eventos = []
    for d in docs:
        ts = d.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

        d["timestamp"] = ts.isoformat()
        d.pop("_id", None)
        eventos.append(d)

    # Revertir a orden cronológico para el análisis
    # (aunque group_similar no le importa, es mejor ser consistentes)
    eventos.reverse()
    return eventos


# =======================
# MAIN LOOP
# =======================

def main():
    logger.info("🔥 Victoria PreCalculator ULTRA ONLINE (cada 5 minutos)")

    while True:
        try:
            print("\n=========================")
            logger.info("🔄 Ejecutando ciclo ULTRA")
            print("=========================")

            # 1) Actual (5 min)
            procesar_actual_desde_general()

            # 2) Tres horas -> Ahora "Short Term" (últimos 200 eventos)
            ev_tres = read_last_n_events(200)
            logger.info(f"🔎 TRES (Last 200): Encontrados {len(ev_tres)} eventos.")
            procesar_si_cambia("tres", group_similar_events(ev_tres), MODEL_TRES)

            # 3) Día -> Ahora "Long Term" (últimos 1000 eventos)
            ev_dia = read_last_n_events(1000)
            logger.info(f"🔎 DIA (Last 1000): Encontrados {len(ev_dia)} eventos.")
            procesar_si_cambia("dia", group_similar_events(ev_dia), MODEL_DIA)

            # 4) Ayer -> DISABLED per user request
            # (Logic removed)

        except Exception as e:
            logger.error(f"❌ ERROR GENERAL EN CICLO PRINCIPAL: {e}", exc_info=True)

        logger.info("⏳ Durmiendo 5 minutos...\n")
        time.sleep(300)


if __name__ == "__main__":
    main()