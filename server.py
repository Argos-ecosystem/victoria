from flask import Flask, request
from flask_cors import CORS
from pymongo import MongoClient
import os
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# ===== CONFIG =====
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_CACHE = os.getenv("MONGO_DB_CACHE", "victoria")
CACHE_COLL = "victoria_cache"
MONGO_DB_EVENTS = os.getenv("MONGO_DB_EVENTS", "omnistatus")
EVENTS_COLL = os.getenv("MONGO_COLL_NAME", "events")
APIKEY = os.getenv("VICTORIA_APIKEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

try:
    SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "200"))
except ValueError:
    SUMMARY_MAX_CHARS = 200

mongo = MongoClient(MONGO_URI)
col_cache = mongo[MONGO_DB_CACHE][CACHE_COLL]
daily_col = col_cache # Fallback for backwards compatibility with report_blocks_3h
col_events = mongo[MONGO_DB_EVENTS][EVENTS_COLL]

import openai
if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# ======================
# HELPERS
# ======================

def is_apikey_valid(req) -> bool:
    requested_key = req.args.get("apikey")
    return bool(APIKEY) and bool(requested_key) and requested_key == APIKEY

def sanitize(text: str) -> str:
    if not text:
        return "No information available."
    return (
        text.replace("<", "")
            .replace(">", "")
            .replace("&", " and ")
            .replace('"', "")
            .replace("'", "")
            .strip()
    )

def normalize_event(doc: dict) -> dict:
    event = dict(doc)
    event.pop("_id", None)
    if "timestamp" in event and hasattr(event["timestamp"], "isoformat"):
        event["timestamp"] = event["timestamp"].isoformat()
    return event

def event_fingerprint(event: dict) -> str:
    comparable = {k: v for k, v in event.items() if k != "timestamp"}
    return json.dumps(comparable, ensure_ascii=False, sort_keys=True, default=str)

def build_analysis_prompt(user_prompt: str) -> str:
    base_prompt = os.getenv(
        "PROMPT_ANALYSIS",
        "Eres un analista de eventos; identifica hechos relevantes y genera un resumen muy breve, directo y sin explicaciones extensas."
    )
    return (
        f"{base_prompt}\n"
        f"Consulta del usuario: {user_prompt}\n"
        f"Responde en espanol, en aproximadamente {SUMMARY_MAX_CHARS} caracteres. "
        "Prioriza lo mas relevante, no inventes datos y considera que los eventos ya vienen ordenados cronologicamente y sin duplicados."
    )

# ======================
# ENDPOINTS
# ======================

@app.get("/report/blocks/3h")
def report_blocks_3h():
    if not is_apikey_valid(request):
        return {"error": "Invalid apikey"}, 403

    # date=YYYY-MM-DD (UTC)
    day = request.args.get("date")
    granularity = request.args.get("granularity", "3h") # '3h' or 'detailed'
    
    if not day:
        day = datetime.now(timezone.utc).date().isoformat()

    doc = daily_col.find_one({"date": day})
    items = []
    
    field_name = "blocks_detailed" if granularity == "detailed" else "blocks"

    if doc and field_name in doc:
        blocks = doc[field_name]
        # blocks is a dict: {"00": {...}} or {"00:10": {...}}
        
        for key in sorted(blocks.keys()):
            b_data = blocks[key]
            
            # Construct ISO timestamp
            if len(key) == 2: # "00", "03"
                block_ts_iso = f"{day}T{key}:00:00+00:00"
            elif len(key) == 5: # "09:10"
                block_ts_iso = f"{day}T{key}:00+00:00"
            else:
                block_ts_iso = f"{day}T00:00:00+00:00" # Fallback

            items.append({
                "block": block_ts_iso,
                "texto": sanitize(b_data.get("text", "")),
                "score": b_data.get("score", 0),
                "events_hash": f"count:{b_data.get('events_count', 0)}",
                "events_detail": b_data.get("events_detail", []), 
                "is_current": (b_data.get("status") == "active")
            })

    return {
        "day": day, 
        "granularity": granularity,
        "daily_summary": doc.get("daily_summary", "Sin resumen disponible."), 
        "daily_score": doc.get("daily_score", 0.0),
        "items": items
    }

# ======================
# LLM ON-DEMAND API
# ======================
@app.post("/analyze/on-demand")
def analyze_on_demand():
    if not is_apikey_valid(request):
        return {"error": "Invalid apikey"}, 403

    data = request.get_json() or {}
    try:
        minutes = int(data.get("minutes", 60))
    except (TypeError, ValueError):
        return {"error": "minutes must be an integer"}, 400

    if minutes <= 0:
        return {"error": "minutes must be greater than zero"}, 400

    custom_prompt = data.get("prompt") or "Resume los eventos relevantes de forma breve."
    analysis_prompt = build_analysis_prompt(custom_prompt)

    if not client:
         return {"error": "OpenAI API Key is missing or invalid in server."}, 500

    # Read events from last N minutes
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    
    docs = col_events.find({"timestamp": {"$gte": cutoff}}).sort("timestamp", 1)

    events = []
    seen = set()
    duplicates_removed = 0
    for d in docs:
        event = normalize_event(d)
        fingerprint = event_fingerprint(event)
        if fingerprint in seen:
            duplicates_removed += 1
            continue
        seen.add(fingerprint)
        events.append(event)
        
    if not events:
        return {"result": "No hay eventos registrados en este rango de tiempo.", "events_count": 0}
        
    # Limit events to avoid context window explosion
    if len(events) > 100:
        events = events[-100:]
        
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": analysis_prompt},
                {"role": "user", "content": f"Eventos:\n{json.dumps(events, ensure_ascii=False)}"}
            ],
            max_tokens=120
        )
        texto = response.choices[0].message.content
        return {
            "minutes": minutes,
            "events_count": len(events),
            "duplicates_removed": duplicates_removed,
            "result": sanitize(texto)
        }
    except Exception as e:
        return {"error": str(e)}, 500

# ======================
# MAIN
# ======================

if __name__ == "__main__":
    print("Victoria Server running on port 8888 🦊")
    app.run(host="0.0.0.0", port=8888)
