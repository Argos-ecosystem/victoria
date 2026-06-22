import argparse
import json
import math
import os
import queue
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
import warnings

import openai
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.(auth|oauth2)")

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_VOICE_MODEL = os.getenv("OPENAI_VOICE_MODEL", "gpt-4o-mini")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TRANSCRIBE_MODEL = os.getenv("GEMINI_TRANSCRIBE_MODEL", "gemini-2.5-flash")
OMNI_URL = os.getenv("OMNI_URL", "http://localhost:8001/analyze/custom")
RECORDING_SOUND = os.getenv("OMNI_RECORDING_SOUND", "/System/Library/Sounds/Ping.aiff")
PROCESSING_MESSAGE = os.getenv(
    "OMNI_PROCESSING_MESSAGE",
    "Consultando el sistema central de datos.",
).strip()
PROCESSING_VOICE = os.getenv("OMNI_PROCESSING_VOICE", "Paulina").strip()
PROCESSING_RATE = os.getenv("OMNI_PROCESSING_RATE", "175").strip()
RECORDING_PROMPT_MESSAGE = os.getenv("OMNI_RECORDING_PROMPT_MESSAGE", "Habla al escuchar el pip.").strip()
RECORDING_PROMPT_VOICE = os.getenv("OMNI_RECORDING_PROMPT_VOICE", PROCESSING_VOICE).strip()
RECORDING_PROMPT_RATE = os.getenv("OMNI_RECORDING_PROMPT_RATE", "190").strip()
VOICE_UI_ENABLED = os.getenv("OMNI_VOICE_UI", "1").strip().lower() not in {"0", "false", "no", "off"}
VOICE_UI_WIDTH = int(os.getenv("OMNI_VOICE_UI_WIDTH", "1280"))
VOICE_UI_HEIGHT = int(os.getenv("OMNI_VOICE_UI_HEIGHT", "720"))
FACE_UI_ENABLED = os.getenv("OMNI_FACE_UI", "1").strip().lower() not in {"0", "false", "no", "off"}
FACE_CAMERA_INDEX = int(os.getenv("OMNI_FACE_CAMERA_INDEX", "0"))
FACE_UI_SCALE = float(os.getenv("OMNI_FACE_UI_SCALE", "0.85"))
FACE_PANEL_WIDTH = int(os.getenv("OMNI_FACE_PANEL_WIDTH", "420"))
FACE_PANEL_HEIGHT = int(os.getenv("OMNI_FACE_PANEL_HEIGHT", "260"))
FACE_DETECT_SCALE_FACTOR = float(os.getenv("OMNI_FACE_DETECT_SCALE_FACTOR", "1.12"))
FACE_DETECT_MIN_NEIGHBORS = int(os.getenv("OMNI_FACE_DETECT_MIN_NEIGHBORS", "10"))
FACE_DETECT_MIN_SIZE = int(os.getenv("OMNI_FACE_DETECT_MIN_SIZE", "80"))
FACE_MEMORY_ENABLED = os.getenv("OMNI_FACE_MEMORY", "1").strip().lower() not in {"0", "false", "no", "off"}
FACE_MEMORY_DIR = os.getenv("OMNI_FACE_MEMORY_DIR", ".victoria_face_memory")
FACE_MEMORY_COLLECTION = os.getenv("OMNI_FACE_MEMORY_COLLECTION", "victoria_faces")
FACE_MEMORY_BACKEND = os.getenv("OMNI_FACE_MEMORY_BACKEND", "opencv").strip().lower()
FACE_MEMORY_MODEL = os.getenv("OMNI_FACE_MEMORY_MODEL", "clip-ViT-B-32")
FACE_MEMORY_SAMPLES = int(os.getenv("OMNI_FACE_MEMORY_SAMPLES", "3"))
FACE_MEMORY_THRESHOLD = float(os.getenv("OMNI_FACE_MEMORY_THRESHOLD", "0.30"))
FACE_MEMORY_COOLDOWN_SECONDS = float(os.getenv("OMNI_FACE_MEMORY_COOLDOWN_SECONDS", "20"))
FACE_MEMORY_ASK_ENROLL = os.getenv("OMNI_FACE_MEMORY_ASK_ENROLL", "1").strip().lower() not in {"0", "false", "no", "off"}
OMNI_API_MAX_CHARS = os.getenv("OMNI_API_MAX_CHARS", "200")
OMNI_API_MAX_CHARS = OMNI_API_MAX_CHARS.strip()
_mic_index_raw = os.getenv("OMNI_MIC_INDEX", "").strip()
MIC_INDEX = int(_mic_index_raw) if _mic_index_raw else None
OPENAI_REQUEST_TIMEOUT_SECONDS = float(os.getenv("OMNI_OPENAI_TIMEOUT_SECONDS", "45"))
OMNI_REQUEST_TIMEOUT_SECONDS = float(os.getenv("OMNI_REQUEST_TIMEOUT_SECONDS", "60"))
GOOGLE_ASR_TIMEOUT_SECONDS = float(os.getenv("OMNI_GOOGLE_ASR_TIMEOUT_SECONDS", "8"))
WAKEWORD_LISTEN_TIMEOUT_SECONDS = float(os.getenv("OMNI_WAKEWORD_LISTEN_TIMEOUT_SECONDS", "1"))
WAKEWORD_PHRASE_TIME_LIMIT_SECONDS = float(os.getenv("OMNI_WAKEWORD_PHRASE_TIME_LIMIT_SECONDS", "3"))
WAKEWORD_ERROR_SLEEP_SECONDS = float(os.getenv("OMNI_WAKEWORD_ERROR_SLEEP_SECONDS", "0.5"))
WAKEWORD_SAMPLE_RATE = int(os.getenv("OMNI_WAKEWORD_SAMPLE_RATE", "16000"))
WAKEWORD_MIN_RMS = float(os.getenv("OMNI_WAKEWORD_MIN_RMS", "35"))
WAKEWORD_MIN_PEAK = int(os.getenv("OMNI_WAKEWORD_MIN_PEAK", "350"))
RECORDING_TIMEOUT_PADDING_SECONDS = float(os.getenv("OMNI_RECORDING_TIMEOUT_PADDING_SECONDS", "3"))
SOUND_TIMEOUT_SECONDS = float(os.getenv("OMNI_SOUND_TIMEOUT_SECONDS", "3"))
TTS_PLAYBACK_TIMEOUT_PADDING_SECONDS = float(os.getenv("OMNI_TTS_PLAYBACK_TIMEOUT_PADDING_SECONDS", "10"))
GOOGLE_ASR_LANGUAGES = [
    lang.strip()
    for lang in os.getenv("OMNI_GOOGLE_ASR_LANGUAGES", "es-CL,es-ES,es-419").split(",")
    if lang.strip()
]
LOCAL_ASR_MODEL = os.getenv("OMNI_LOCAL_ASR_MODEL", "tiny")
LOCAL_ASR_TIMEOUT_SECONDS = int(os.getenv("OMNI_LOCAL_ASR_TIMEOUT_SECONDS", "60"))
DEFAULT_QUERY_MINUTES = int(os.getenv("OMNI_DEFAULT_QUERY_MINUTES", "60"))
BAD_TRANSCRIPT_MARKERS = (
    "[NO_SPEECH]",
    "amara.org",
    "subtitulos realizados",
    "subtítulos realizados",
    "gracias por ver",
    "biberón",
    "biberon",
    "canal de subtítulos",
    "canal de subtitulos",
    "iglesia de jesucristo",
    "santos de los últimos días",
    "santos de los ultimos dias",
)

if not OPENAI_API_KEY:
    print("Error: falta OPENAI_API_KEY en el archivo .env.")
    raise SystemExit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_REQUEST_TIMEOUT_SECONDS)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if genai is not None and GEMINI_API_KEY else None
WAKE_SOUND_FILE = "/System/Library/Sounds/Glass.aiff"
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "coral")
OPENAI_TTS_INSTRUCTIONS = os.getenv(
    "OPENAI_TTS_INSTRUCTIONS",
    (
        "Habla en espanol latinoamericano natural, con acento chileno suave y cercano. "
        "Manten una voz calida, clara, calmada y conversacional. Evita sonar como locucion neutra corporativa."
    ),
)
OPENAI_TTS_SPEED = float(os.getenv("OPENAI_TTS_SPEED", "1.03"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "consultar_omnistatus",
            "description": "Consulta los eventos recientes de Omnistatus y devuelve un analisis breve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutos": {
                        "type": "integer",
                        "description": (
                            "Rango de tiempo en minutos inferido desde lo que pidio el usuario. "
                            "Ejemplos: ultimos 15 minutos = 15, ultima hora = 60, dos horas = 120, "
                            "tres horas = 180, hoy o el dia = 1440, ayer = 2880. "
                            "Si el usuario no menciona tiempo, usa el valor por defecto indicado por el sistema."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Texto exacto dicho por el usuario, sin resumir ni reescribir.",
                    },
                },
                "required": ["minutos", "prompt"],
            },
        },
    }
]

def normalize_omni_url(url: str) -> str:
    if url.endswith("/analyze/on-demand"):
        return url[: -len("/analyze/on-demand")] + "/analyze/custom"
    return url


def minutes_to_hours(minutes: int) -> int:
    hours = max(1, (max(1, minutes) + 59) // 60)
    return min(hours, 168)


def format_omnistatus_response(data: dict) -> str:
    msg = data.get("msg") or data.get("result")
    if msg:
        return msg
    return json.dumps(data, ensure_ascii=False)


def build_api_prompt(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not OMNI_API_MAX_CHARS:
        return prompt

    try:
        max_chars = int(OMNI_API_MAX_CHARS)
    except ValueError:
        print(f"Aviso: OMNI_API_MAX_CHARS invalido: {OMNI_API_MAX_CHARS!r}.")
        return prompt

    if max_chars <= 0:
        return prompt

    return f"{prompt}\n\nResponde en maximo {max_chars} caracteres."


def consultar_omnistatus(minutos: int, prompt: str) -> str:
    hours = minutes_to_hours(minutos)
    api_prompt = build_api_prompt(prompt)
    print(f"[Function Call] hours={hours} | minutos={minutos} | prompt='{api_prompt}'")

    payload = {"hours": hours, "prompt": api_prompt}
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }

    try:
        response = requests.post(
            normalize_omni_url(OMNI_URL),
            json=payload,
            headers=headers,
            timeout=OMNI_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return format_omnistatus_response(data)
    except Exception as e:
        return f"Error en la consulta a Omnistatus: {e}"


def play_recording_sound():
    if not RECORDING_SOUND or not os.path.exists(RECORDING_SOUND):
        return

    try:
        subprocess.run(["afplay", RECORDING_SOUND], check=False, timeout=SOUND_TIMEOUT_SECONDS)
    except Exception as e:
        print(f"Aviso: no se pudo reproducir el sonido de grabacion: {e}")


def speak_system_message(message, voice="", rate=""):
    message = (message or "").strip()
    if not message:
        return

    command = ["say"]
    if voice:
        command.extend(["-v", voice])
    if rate:
        command.extend(["-r", rate])
    command.append(message)

    try:
        subprocess.run(command, check=False, timeout=max(3.0, len(message) * 0.08))
    except Exception as e:
        print(f"Aviso: no se pudo reproducir el aviso de voz: {e}")


def speak_recording_prompt():
    speak_system_message(
        RECORDING_PROMPT_MESSAGE,
        voice=RECORDING_PROMPT_VOICE,
        rate=RECORDING_PROMPT_RATE,
    )


def start_processing_message():
    if not PROCESSING_MESSAGE:
        return None

    command = ["say"]
    if PROCESSING_VOICE:
        command.extend(["-v", PROCESSING_VOICE])
    if PROCESSING_RATE:
        command.extend(["-r", PROCESSING_RATE])
    command.append(PROCESSING_MESSAGE)

    try:
        print(f"Victoria: {PROCESSING_MESSAGE}")
        return subprocess.Popen(command)
    except Exception as e:
        print(f"Aviso: no se pudo reproducir el mensaje de espera: {e}")
        return None


def stop_processing_message(process):
    if process is None or process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=1)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


class VoiceStatusController:
    def __init__(self, enabled=True, face_enabled=True):
        self.enabled = enabled
        self.face_enabled = face_enabled
        self.events = queue.Queue()
        self.process = None

    def start(self):
        if not self.enabled:
            return

        try:
            child_env = os.environ.copy()
            child_env["OMNI_FACE_UI"] = "1" if self.face_enabled else "0"
            child_env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            child_env.setdefault("TOKENIZERS_PARALLELISM", "false")
            self.process = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--status-window-child"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                env=child_env,
            )
            threading.Thread(target=self._read_events, daemon=True).start()
            self.set_state("idle", "Victoria", "Di Victoria para comenzar")
        except Exception as e:
            self.process = None
            print(f"Aviso: no se pudo abrir la ventana visual: {e}")

    def _read_events(self):
        if self.process is None or self.process.stdout is None:
            return

        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.events.put(json.loads(line))
            except json.JSONDecodeError:
                print(line)

    def send_command(self, payload):
        if self.process is None or self.process.poll() is not None or self.process.stdin is None:
            return False

        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            return True
        except Exception:
            self.close()
            return False

    def set_state(self, state, title=None, subtitle=None, duration=None, spectrum=None):
        if self.process is None or self.process.poll() is not None or self.process.stdin is None:
            return

        payload = {"state": state}
        if title is not None:
            payload["title"] = title
        if subtitle is not None:
            payload["subtitle"] = subtitle
        if duration is not None:
            payload["duration"] = duration
        if spectrum is not None:
            payload["spectrum"] = spectrum

        self.send_command(payload)

    def enroll_face(self, name):
        return self.send_command({"action": "enroll_face", "name": name})

    def delete_face(self, name):
        return self.send_command({"action": "delete_face", "name": name})

    def list_faces(self):
        return self.send_command({"action": "list_faces"})

    def wait_for_delete_result(self, timeout_seconds=8):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=0.25)
            except queue.Empty:
                continue
            if event.get("type") in {"face_deleted", "face_delete_error"}:
                return event
        return {"type": "face_delete_error", "error": "timeout"}

    def wait_for_face_list(self, timeout_seconds=5):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=0.25)
            except queue.Empty:
                continue
            if event.get("type") == "face_list":
                return event.get("names", [])
        return []

    def get_delete_request(self):
        found = False
        kept_events = []
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event.get("type") == "delete_enroll_requested":
                found = True
            else:
                kept_events.append(event)
        for event in kept_events:
            self.events.put(event)
        return found

    def get_enroll_request(self):
        found = False
        kept_events = []
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event.get("type") == "enroll_requested":
                found = True
            else:
                kept_events.append(event)
        for event in kept_events:
            self.events.put(event)
        return found

    def get_latest_recognized_name(self):
        recognized_name = None
        kept_events = []
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event.get("type") == "face_recognized":
                recognized_name = event.get("name") or recognized_name
            else:
                kept_events.append(event)

        for event in kept_events:
            self.events.put(event)
        return recognized_name

    def wait_for_enroll_result(self, timeout_seconds=12):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=0.25)
            except queue.Empty:
                continue
            if event.get("type") in {"face_enrolled", "face_enroll_error"}:
                return event
        return {"type": "face_enroll_error", "error": "timeout"}

    def close(self):
        if self.process is None:
            return

        try:
            if self.process.stdin is not None and self.process.poll() is None:
                self.process.stdin.write(json.dumps({"state": "quit"}) + "\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            self.process.wait(timeout=1)
        except Exception:
            try:
                self.process.terminate()
            except Exception:
                pass
        finally:
            self.process = None


status_window = None
_greeted_names: set = set()


def set_voice_status(state, title=None, subtitle=None, duration=None, spectrum=None):
    if status_window is not None:
        status_window.set_state(
            state,
            title=title,
            subtitle=subtitle,
            duration=duration,
            spectrum=spectrum,
        )


def victoria_voice_spectrum(text, bins=44):
    source = text or "Victoria"
    values = []
    seed = sum((index + 1) * ord(char) for index, char in enumerate(source))
    for index in range(bins):
        char_value = ord(source[index % len(source)])
        harmonic = math.sin((seed * 0.017 + index * 0.73) * math.tau)
        formant = math.sin((char_value * 0.011 + index * 0.19) * math.tau)
        envelope = math.sin(math.pi * (index + 0.5) / bins)
        value = 0.18 + (0.50 + 0.28 * harmonic + 0.22 * formant) * envelope
        values.append(max(0.12, min(1.0, value)))
    return values


def status_spectrum(state_name, title="", subtitle=""):
    seed_text = f"{state_name}:{title}:{subtitle}"
    scale_by_state = {
        "idle": 0.45,
        "listening": 0.62,
        "recording": 0.85,
        "processing": 0.70,
        "speaking": 1.00,
        "error": 0.55,
    }
    scale = scale_by_state.get(state_name, 0.65)
    return [max(0.08, min(1.0, value * scale)) for value in victoria_voice_spectrum(seed_text)]


def animated_bars(base, state_name):
    t = time.monotonic()
    amplitude = {
        "idle": 0.05, "listening": 0.09, "recording": 0.16,
        "processing": 0.07, "speaking": 0.20, "error": 0.06,
    }.get(state_name, 0.07)
    result = []
    for i, v in enumerate(base):
        wave = amplitude * math.sin(t * (1.1 + i * 0.18) + i * 0.55)
        result.append(max(0.08, min(1.0, v + wave)))
    return result


def lerp_color(c1, c2, t):
    """Interpolate between two (r,g,b) float tuples."""
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


def emit_child_event(event):
    try:
        print(json.dumps(event), flush=True)
    except Exception:
        pass


class FaceMemory:
    def __init__(self):
        self.enabled = False
        self.loading = False
        self.error = ""
        self.model = None
        self.collection = None

        if not FACE_MEMORY_ENABLED:
            return

        self.loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            import chromadb

            self.image_cls = None
            if FACE_MEMORY_BACKEND == "clip":
                from PIL import Image
                from sentence_transformers import SentenceTransformer

                self.image_cls = Image
                self.model = SentenceTransformer(FACE_MEMORY_MODEL)
            client = chromadb.PersistentClient(path=FACE_MEMORY_DIR)
            self.collection = client.get_or_create_collection(
                FACE_MEMORY_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            self.enabled = True
            emit_child_event({"type": "face_memory_ready"})
        except Exception as e:
            self.error = str(e)
            emit_child_event({"type": "face_memory_error", "error": self.error})
        finally:
            self.loading = False

    def embed_crop(self, crop_bgr, cv2):
        if not self.enabled or crop_bgr is None or crop_bgr.size == 0:
            return None

        if FACE_MEMORY_BACKEND != "clip":
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            vector = resized.astype(np.float32).reshape(-1) / 255.0
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(-1).astype(np.float32)
            hist_sum = float(hist.sum())
            if hist_sum > 0:
                hist = hist / hist_sum
            embedding = np.concatenate([vector, hist])
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding = embedding / norm
            return embedding.tolist()

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = self.image_cls.fromarray(rgb)
        embedding = self.model.encode([image], normalize_embeddings=True)[0]
        return np.asarray(embedding, dtype=np.float32).tolist()

    def remember(self, name, crops, cv2):
        if not self.enabled:
            if self.loading:
                return False, "memoria facial cargando modelo CLIP"
            return False, self.error or "memoria facial no disponible"

        clean_name = " ".join((name or "").strip().split())
        if not clean_name:
            return False, "nombre vacio"

        embeddings = []
        ids = []
        metadatas = []
        documents = []
        for index, crop in enumerate(crops):
            embedding = self.embed_crop(crop, cv2)
            if embedding is None:
                continue
            embeddings.append(embedding)
            ids.append(f"{clean_name}-{uuid.uuid4()}")
            metadatas.append({"name": clean_name, "sample": index + 1})
            documents.append(clean_name)

        if not embeddings:
            return False, "no se pudieron generar embeddings"

        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        return True, clean_name

    def recognize(self, crop_bgr, cv2):
        if not self.enabled:
            return None

        try:
            if self.collection.count() <= 0:
                return None
            embedding = self.embed_crop(crop_bgr, cv2)
            if embedding is None:
                return None
            result = self.collection.query(query_embeddings=[embedding], n_results=1)
            distances = result.get("distances") or []
            metadatas = result.get("metadatas") or []
            if not distances or not distances[0] or not metadatas or not metadatas[0]:
                return None
            distance = float(distances[0][0])
            if distance > FACE_MEMORY_THRESHOLD:
                return None
            name = metadatas[0][0].get("name")
            if not name:
                return None
            return {"name": name, "distance": distance}
        except Exception as e:
            emit_child_event({"type": "face_memory_error", "error": str(e)})
            return None

    def delete(self, name):
        if not self.enabled:
            return False, self.error or "memoria facial no disponible"
        clean_name = " ".join((name or "").strip().split())
        if not clean_name:
            return False, "nombre vacio"
        try:
            results = self.collection.get(where={"name": clean_name})
            ids = results.get("ids") or []
            if not ids:
                return False, f"no encontre a {clean_name} en memoria"
            self.collection.delete(ids=ids)
            return True, clean_name
        except Exception as e:
            return False, str(e)

    def list_names(self):
        if not self.enabled:
            return []
        try:
            results = self.collection.get()
            metadatas = results.get("metadatas") or []
            return sorted({m.get("name") for m in metadatas if m.get("name")})
        except Exception:
            return []


def run_cocoa_status_window_child(tk_error):
    try:
        import objc
        from AppKit import (
            NSApplication,
            NSApplicationActivationPolicyRegular,
            NSBackingStoreBuffered,
            NSBezierPath,
            NSColor,
            NSCompositeSourceOver,
            NSFont,
            NSFontAttributeName,
            NSForegroundColorAttributeName,
            NSInsetRect,
            NSImage,
            NSMakeRect,
            NSMutableParagraphStyle,
            NSParagraphStyleAttributeName,
            NSView,
            NSWindow,
            NSWindowStyleMaskClosable,
            NSWindowStyleMaskMiniaturizable,
            NSWindowStyleMaskTitled,
        )
        from Foundation import NSData, NSObject, NSString, NSTimer
    except Exception as e:
        print(f"Aviso: no se pudo abrir la ventana visual. Tk fallo: {tk_error}; Cocoa fallo: {e}", file=sys.stderr)
        return 1

    messages = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    threading.Thread(target=read_stdin, daemon=True).start()

    class WaveView(NSView):
        def initWithFrame_(self, frame):
            self = objc.super(WaveView, self).initWithFrame_(frame)
            if self is None:
                return None

            self.state = {
                "name": "idle",
                "title": "Victoria",
                "subtitle": "Di Victoria para comenzar",
                "duration": None,
                "spectrum": victoria_voice_spectrum("Victoria"),
                "started_at": time.monotonic(),
            }
            self.palette = {
                "idle": ((0.31, 0.55, 1.0), (0.49, 0.66, 1.0)),
                "listening": ((0.12, 0.82, 0.65), (0.49, 0.96, 0.83)),
                "recording": ((1.0, 0.30, 0.43), (1.0, 0.60, 0.68)),
                "processing": ((0.96, 0.77, 0.27), (1.0, 0.88, 0.54)),
                "speaking": ((0.69, 0.42, 1.0), (0.82, 0.66, 1.0)),
                "error": ((1.0, 0.42, 0.29), (1.0, 0.70, 0.60)),
            }
            self.cv2 = None
            self.camera = None
            self.faceDetector = None
            self.cameraImage = None
            self.cameraFaces = []
            self.latestFaceCrops = []
            self.cameraSize = (0, 0)
            self.lastCameraAt = 0
            self.lastRecognitionAt = 0
            self.lastRecognizedName = ""
            self.pendingEnrollName = ""
            self.pendingEnrollCrops = []
            self.cameraError = ""
            self.faceMemory = FaceMemory()
            # lerp color state
            self.currentPrimary = (0.31, 0.55, 1.0)
            self.currentSecondary = (0.49, 0.66, 1.0)
            # names panel
            self.showNamesPanel = False
            self.showingNames = []
            self.nameItemRects = []
            if FACE_UI_ENABLED:
                self.startCamera()
            return self

        def startCamera(self):
            try:
                import cv2

                cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                detector = cv2.CascadeClassifier(cascade_path)
                if detector.empty():
                    self.cameraError = "sin detector"
                    return

                camera = cv2.VideoCapture(FACE_CAMERA_INDEX)
                if not camera.isOpened():
                    self.cameraError = f"camara {FACE_CAMERA_INDEX} no disponible"
                    return

                self.cv2 = cv2
                self.faceDetector = detector
                self.camera = camera
            except Exception as e:
                self.cameraError = str(e)

        def updateCamera(self):
            if self.camera is None or self.cv2 is None:
                return
            now = time.monotonic()
            if now - self.lastCameraAt < 0.12:
                return
            self.lastCameraAt = now

            ok, frame = self.camera.read()
            if not ok:
                return

            if FACE_UI_SCALE and FACE_UI_SCALE != 1:
                frame = self.cv2.resize(frame, None, fx=FACE_UI_SCALE, fy=FACE_UI_SCALE, interpolation=self.cv2.INTER_AREA)

            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            gray = self.cv2.equalizeHist(gray)
            faces = self.faceDetector.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECT_SCALE_FACTOR,
                minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
                minSize=(FACE_DETECT_MIN_SIZE, FACE_DETECT_MIN_SIZE),
            )

            ok, encoded = self.cv2.imencode(".png", frame)
            if not ok:
                return

            height, width = frame.shape[:2]
            self.cameraSize = (width, height)
            self.cameraFaces = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
            self.latestFaceCrops = []
            for x, y, w, h in self.cameraFaces:
                pad_x = int(w * 0.12)
                pad_y = int(h * 0.16)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(width, x + w + pad_x)
                y2 = min(height, y + h + pad_y)
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size:
                    self.latestFaceCrops.append(crop)

            if self.pendingEnrollName and self.latestFaceCrops:
                self.pendingEnrollCrops.append(self.latestFaceCrops[0])
                if len(self.pendingEnrollCrops) >= FACE_MEMORY_SAMPLES:
                    ok_memory, result = self.faceMemory.remember(
                        self.pendingEnrollName,
                        self.pendingEnrollCrops[:FACE_MEMORY_SAMPLES],
                        self.cv2,
                    )
                    if ok_memory:
                        emit_child_event({"type": "face_enrolled", "name": result})
                    else:
                        emit_child_event({"type": "face_enroll_error", "error": result})
                    self.pendingEnrollName = ""
                    self.pendingEnrollCrops = []

            if self.latestFaceCrops and time.monotonic() - self.lastRecognitionAt >= FACE_MEMORY_COOLDOWN_SECONDS:
                recognition = self.faceMemory.recognize(self.latestFaceCrops[0], self.cv2)
                self.lastRecognitionAt = time.monotonic()
                if recognition and recognition["name"] != self.lastRecognizedName:
                    self.lastRecognizedName = recognition["name"]
                    emit_child_event(
                        {
                            "type": "face_recognized",
                            "name": recognition["name"],
                            "distance": recognition["distance"],
                        }
                    )
            image_data = encoded.tobytes()
            ns_data = NSData.dataWithBytes_length_(image_data, len(image_data))
            self.cameraImage = NSImage.alloc().initWithData_(ns_data)

        def closeCamera(self):
            if self.camera is not None:
                self.camera.release()
                self.camera = None

        def drawCameraPanel_withPrimary_(self, panel_rect, primary):
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.02, 0.03, 0.04, 1.0).setFill()
            NSBezierPath.bezierPathWithRect_(panel_rect).fill()
            NSColor.colorWithCalibratedWhite_alpha_(0.18, 1.0).setStroke()
            border = NSBezierPath.bezierPathWithRect_(panel_rect)
            border.setLineWidth_(1)
            border.stroke()

            if not FACE_UI_ENABLED:
                return

            if self.cameraImage is None:
                label = self.cameraError or "iniciando camara"
                self.drawText_inRect_size_bold_color_(
                    label,
                    panel_rect,
                    11,
                    False,
                    NSColor.colorWithCalibratedWhite_alpha_(0.72, 1.0),
                )
                return

            image_width, image_height = self.cameraSize
            if not image_width or not image_height:
                return

            image_ratio = image_width / image_height
            panel_ratio = panel_rect.size.width / panel_rect.size.height
            if image_ratio > panel_ratio:
                draw_width = panel_rect.size.width
                draw_height = draw_width / image_ratio
            else:
                draw_height = panel_rect.size.height
                draw_width = draw_height * image_ratio

            draw_x = panel_rect.origin.x + (panel_rect.size.width - draw_width) / 2
            draw_y = panel_rect.origin.y + (panel_rect.size.height - draw_height) / 2
            draw_rect = NSMakeRect(draw_x, draw_y, draw_width, draw_height)
            self.cameraImage.drawInRect_fromRect_operation_fraction_(draw_rect, NSMakeRect(0, 0, image_width, image_height), NSCompositeSourceOver, 1.0)

            x_scale = draw_width / image_width
            y_scale = draw_height / image_height
            NSColor.redColor().setStroke()
            for face_x, face_y, face_width, face_height in self.cameraFaces:
                rect_x = draw_x + face_x * x_scale
                rect_y = draw_y + (image_height - face_y - face_height) * y_scale
                face_rect = NSMakeRect(rect_x, rect_y, face_width * x_scale, face_height * y_scale)
                path = NSBezierPath.bezierPathWithRect_(face_rect)
                path.setLineWidth_(2.5)
                path.stroke()

        def applyMessage_(self, message):
            if message.get("state") == "quit":
                self.closeCamera()
                NSApplication.sharedApplication().terminate_(None)
                return
            if message.get("action") == "enroll_face":
                name = " ".join((message.get("name") or "").strip().split())
                if not name:
                    emit_child_event({"type": "face_enroll_error", "error": "nombre vacio"})
                    return
                self.pendingEnrollName = name
                self.pendingEnrollCrops = []
                self.state["name"] = "processing"
                self.state["title"] = "Recordando rostro"
                self.state["subtitle"] = f"Tomando {FACE_MEMORY_SAMPLES} fotos de {name}"
                self.state["started_at"] = time.monotonic()
                return
            if message.get("action") == "delete_face":
                name = " ".join((message.get("name") or "").strip().split())
                ok, result = self.faceMemory.delete(name)
                if ok:
                    emit_child_event({"type": "face_deleted", "name": result})
                else:
                    emit_child_event({"type": "face_delete_error", "error": result})
                return
            if message.get("action") == "list_faces":
                names = self.faceMemory.list_names()
                emit_child_event({"type": "face_list", "names": names})
                return

            self.state["name"] = message.get("state", self.state["name"])
            self.state["title"] = message.get("title", self.state["title"])
            self.state["subtitle"] = message.get("subtitle", self.state["subtitle"])
            self.state["duration"] = message.get("duration")
            self.state["spectrum"] = message.get("spectrum", self.state["spectrum"])
            self.state["started_at"] = time.monotonic()

        def tick_(self, timer):
            while True:
                try:
                    self.applyMessage_(messages.get_nowait())
                except queue.Empty:
                    break
            self.updateCamera()
            self.setNeedsDisplay_(True)

        def drawText_inRect_size_bold_color_(self, text, rect, size, bold, color):
            style = NSMutableParagraphStyle.alloc().init()
            style.setAlignment_(1)
            font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
            attrs = {
                NSFontAttributeName: font,
                NSForegroundColorAttributeName: color,
                NSParagraphStyleAttributeName: style,
            }
            NSString.stringWithString_(text).drawInRect_withAttributes_(rect, attrs)

        def drawEqualizerInRect_spectrum_primary_secondary_(self, rect, spectrum, primary, secondary):
            gap = 5
            bar_count = min(16, len(spectrum))
            bar_width = max(4, (rect.size.width - gap * (bar_count - 1)) / bar_count)
            baseline = rect.origin.y + 8
            max_height = rect.size.height - 16
            start = max(0, (len(spectrum) - bar_count) // 2)

            for index in range(bar_count):
                value = spectrum[start + index]
                x = rect.origin.x + index * (bar_width + gap)
                bar_height = max(8, value * max_height)
                primary.setFill()
                NSBezierPath.bezierPathWithRect_(NSMakeRect(x, baseline, bar_width, bar_height)).fill()
                secondary.setFill()
                NSBezierPath.bezierPathWithRect_(NSMakeRect(x, baseline + bar_height + 3, bar_width, 6)).fill()

        def drawAIFaceInRect_primary_secondary_(self, rect, primary, secondary):
            center_x = rect.origin.x + rect.size.width / 2
            center_y = rect.origin.y + rect.size.height / 2
            face_size = min(rect.size.width, rect.size.height) * 0.82
            face_rect = NSMakeRect(center_x - face_size / 2, center_y - face_size / 2, face_size, face_size)
            state_name = self.state["name"]

            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.04, 0.06, 0.07, 1.0).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(face_rect).fill()
            primary.setStroke()
            outer = NSBezierPath.bezierPathWithOvalInRect_(face_rect)
            outer.setLineWidth_(7 if state_name in {"recording", "speaking"} else 4)
            outer.stroke()

            inner = NSInsetRect(face_rect, face_size * 0.12, face_size * 0.12)
            secondary.setStroke()
            inner_path = NSBezierPath.bezierPathWithOvalInRect_(inner)
            inner_path.setLineWidth_(1.5)
            inner_path.stroke()

            eye_width = face_size * 0.18
            eye_height = face_size * 0.08
            if state_name == "listening":
                eye_width = face_size * 0.28
                eye_height = face_size * 0.12
            elif state_name == "recording":
                eye_width = face_size * 0.16
                eye_height = face_size * 0.16
            elif state_name == "speaking":
                eye_width = face_size * 0.24
                eye_height = face_size * 0.075
            elif state_name == "processing":
                eye_width = face_size * 0.11
                eye_height = face_size * 0.11
            elif state_name == "error":
                eye_width = face_size * 0.20
                eye_height = face_size * 0.055

            eye_y = center_y + face_size * 0.08
            left_eye = NSMakeRect(center_x - face_size * 0.25, eye_y, eye_width, eye_height)
            right_eye = NSMakeRect(center_x + face_size * 0.07, eye_y, eye_width, eye_height)
            secondary.setFill()
            if state_name in {"processing", "recording"}:
                NSBezierPath.bezierPathWithOvalInRect_(left_eye).fill()
                NSBezierPath.bezierPathWithOvalInRect_(right_eye).fill()
            else:
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(left_eye, eye_height / 2, eye_height / 2).fill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(right_eye, eye_height / 2, eye_height / 2).fill()

            mouth_width = face_size * 0.36
            mouth_height = face_size * 0.08
            if state_name == "recording":
                mouth_width = face_size * 0.24
                mouth_height = face_size * 0.24
            elif state_name == "speaking":
                mouth_width = face_size * 0.52
                mouth_height = face_size * 0.18
            elif state_name == "processing":
                mouth_width = face_size * 0.24
                mouth_height = face_size * 0.035
            elif state_name == "error":
                mouth_width = face_size * 0.34
                mouth_height = face_size * 0.08
            mouth = NSMakeRect(center_x - mouth_width / 2, center_y - face_size * 0.20, mouth_width, mouth_height)
            if state_name in {"recording", "speaking"}:
                primary.setFill()
                NSBezierPath.bezierPathWithOvalInRect_(mouth).fill()
            else:
                primary.setStroke()
                smile = NSBezierPath.bezierPath()
                smile.setLineWidth_(max(3, face_size * 0.018))
                smile.moveToPoint_((center_x - mouth_width / 2, center_y - face_size * 0.17))
                smile.curveToPoint_controlPoint1_controlPoint2_(
                    (center_x + mouth_width / 2, center_y - face_size * 0.17),
                    (center_x - mouth_width * 0.22, center_y - face_size * 0.31),
                    (center_x + mouth_width * 0.22, center_y - face_size * 0.31),
                )
                smile.stroke()

            if state_name == "error":
                secondary.setFill()
                dot_size = face_size * 0.032
                for offset in (-0.06, 0.06):
                    dot = NSMakeRect(
                        center_x + face_size * offset - dot_size / 2,
                        center_y - face_size * 0.35,
                        dot_size,
                        dot_size,
                    )
                    NSBezierPath.bezierPathWithOvalInRect_(dot).fill()
            elif state_name == "processing":
                secondary.setFill()
                dot_size = face_size * 0.045
                for offset in (-0.12, 0.0, 0.12):
                    dot = NSMakeRect(
                        center_x + face_size * offset - dot_size / 2,
                        center_y - face_size * 0.34,
                        dot_size,
                        dot_size,
                    )
                    NSBezierPath.bezierPathWithOvalInRect_(dot).fill()
            elif state_name == "listening":
                secondary.setStroke()
                arc = NSBezierPath.bezierPathWithOvalInRect_(NSInsetRect(face_rect, -face_size * 0.06, -face_size * 0.06))
                arc.setLineWidth_(1.5)
                arc.stroke()

        def enrollButtonClicked_(self, sender):
            emit_child_event({"type": "enroll_requested"})

        def deleteButtonClicked_(self, sender):
            self.showingNames = self.faceMemory.list_names()
            self.showNamesPanel = True
            self.nameItemRects = []
            self.setNeedsDisplay_(True)

        def mouseDown_(self, event):
            if not self.showNamesPanel:
                return
            pt = self.convertPoint_fromView_(event.locationInWindow(), None)
            for (x1, y1, x2, y2, name) in self.nameItemRects:
                if x1 <= pt.x <= x2 and y1 <= pt.y <= y2:
                    ok, result = self.faceMemory.delete(name)
                    if ok:
                        emit_child_event({"type": "face_deleted", "name": result})
                    else:
                        emit_child_event({"type": "face_delete_error", "error": result})
                    self.showingNames = self.faceMemory.list_names()
                    self.nameItemRects = []
                    self.setNeedsDisplay_(True)
                    return
            self.showNamesPanel = False
            self.showingNames = []
            self.nameItemRects = []
            self.setNeedsDisplay_(True)

        def _draw_names_panel(self, width, height, primary, secondary):
            self.nameItemRects = []
            panel_w = min(500, width * 0.55)
            panel_h = min(500, max(200, 80 + len(self.showingNames) * 54))
            px = (width - panel_w) / 2
            py = (height - panel_h) / 2
            panel = NSMakeRect(px, py, panel_w, panel_h)

            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.06, 0.08, 0.10, 0.96).setFill()
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(panel, 18, 18)
            path.fill()
            primary.setStroke()
            path.setLineWidth_(1.5)
            path.stroke()

            self.drawText_inRect_size_bold_color_(
                "Enrolamientos guardados",
                NSMakeRect(px, py + panel_h - 46, panel_w, 30),
                18, True,
                NSColor.colorWithCalibratedWhite_alpha_(0.96, 1.0),
            )

            if not self.showingNames:
                self.drawText_inRect_size_bold_color_(
                    "No hay personas guardadas.",
                    NSMakeRect(px, py + panel_h / 2 - 12, panel_w, 24),
                    14, False,
                    NSColor.colorWithCalibratedWhite_alpha_(0.60, 1.0),
                )
                return

            item_h = 44
            for i, name in enumerate(self.showingNames):
                item_y = py + panel_h - 80 - i * (item_h + 8)
                item_rect = NSMakeRect(px + 20, item_y, panel_w - 40, item_h)
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.12, 0.15, 0.18, 1.0).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(item_rect, 10, 10).fill()

                self.drawText_inRect_size_bold_color_(
                    name,
                    NSMakeRect(px + 36, item_y + 12, panel_w - 120, item_h - 12),
                    15, False,
                    NSColor.colorWithCalibratedWhite_alpha_(0.92, 1.0),
                )
                btn_rect = NSMakeRect(px + panel_w - 68, item_y + 8, 42, 28)
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.8, 0.18, 0.22, 1.0).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(btn_rect, 6, 6).fill()
                self.drawText_inRect_size_bold_color_(
                    "✕",
                    btn_rect,
                    14, True,
                    NSColor.colorWithCalibratedWhite_alpha_(1.0, 1.0),
                )
                self.nameItemRects.append((
                    px + 20, item_y, px + panel_w - 20, item_y + item_h, name
                ))

            self.drawText_inRect_size_bold_color_(
                "Toca afuera para cerrar",
                NSMakeRect(px, py + 10, panel_w, 20),
                11, False,
                NSColor.colorWithCalibratedWhite_alpha_(0.40, 1.0),
            )

        def drawRect_(self, rect):
            bounds = self.bounds()
            width = bounds.size.width
            height = bounds.size.height

            target_p, target_s = self.palette.get(self.state["name"], self.palette["idle"])
            self.currentPrimary = lerp_color(self.currentPrimary, target_p, 0.14)
            self.currentSecondary = lerp_color(self.currentSecondary, target_s, 0.14)
            primary = NSColor.colorWithCalibratedRed_green_blue_alpha_(*self.currentPrimary, 1.0)
            secondary = NSColor.colorWithCalibratedRed_green_blue_alpha_(*self.currentSecondary, 1.0)

            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.06, 0.08, 0.09, 1.0).setFill()
            NSBezierPath.bezierPathWithRect_(bounds).fill()

            self.drawText_inRect_size_bold_color_(
                self.state["title"],
                NSMakeRect(0, height - 55, width, 30),
                22, True,
                NSColor.colorWithCalibratedWhite_alpha_(0.96, 1.0),
            )
            self.drawText_inRect_size_bold_color_(
                self.state["subtitle"],
                NSMakeRect(0, height - 83, width, 24),
                13, False,
                NSColor.colorWithCalibratedWhite_alpha_(0.72, 1.0),
            )

            if self.state["name"] == "speaking":
                base_spectrum = self.state.get("spectrum") or victoria_voice_spectrum("Victoria")
            else:
                base_spectrum = status_spectrum(self.state["name"], self.state["title"], self.state["subtitle"])
            spectrum = animated_bars(base_spectrum, self.state["name"])

            margin = 70
            face_top_limit = height - 118
            face_bottom_limit = 285
            face_area_height = max(260, face_top_limit - face_bottom_limit)
            center_size = min(390, face_area_height)
            center_rect = NSMakeRect(
                (width - center_size) / 2,
                face_bottom_limit + (face_area_height - center_size) / 2,
                center_size,
                center_size,
            )
            self.drawAIFaceInRect_primary_secondary_(center_rect, primary, secondary)
            self.drawText_inRect_size_bold_color_(
                self.state["subtitle"],
                NSMakeRect(width * 0.22, center_rect.origin.y - 38, width * 0.56, 30),
                17, False, secondary,
            )

            eq_width = max(250, min(330, (width - center_size - 280) / 2))
            eq_height = 205
            eq_y = 54
            left_eq = NSMakeRect(margin, eq_y, eq_width, eq_height)
            right_eq_x = width - margin - eq_width

            if FACE_UI_ENABLED:
                panel_margin = 18
                panel_width = min(FACE_PANEL_WIDTH, max(320, width * 0.34))
                panel_height = min(FACE_PANEL_HEIGHT, max(200, height * 0.36))
                panel_rect = NSMakeRect(width - panel_width - panel_margin, height - panel_height - 34, panel_width, panel_height)
                self.drawCameraPanel_withPrimary_(panel_rect, primary)

            right_eq = NSMakeRect(right_eq_x, eq_y, eq_width, eq_height)
            self.drawEqualizerInRect_spectrum_primary_secondary_(left_eq, spectrum, primary, secondary)
            self.drawEqualizerInRect_spectrum_primary_secondary_(right_eq, list(reversed(spectrum)), primary, secondary)

            if self.showNamesPanel:
                self._draw_names_panel(width, height, primary, secondary)

            human_label_width = max(160, right_eq.origin.x - (left_eq.origin.x + left_eq.size.width) - 40)
            human_label_x = left_eq.origin.x + left_eq.size.width + 20
            self.drawText_inRect_size_bold_color_(
                "Humano",
                NSMakeRect(human_label_x, 124, human_label_width, 42),
                30, True,
                NSColor.colorWithCalibratedWhite_alpha_(0.92, 1.0),
            )

    class AppDelegate(NSObject):
        def applicationShouldTerminateAfterLastWindowClosed_(self, app):
            return True

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)

    style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(200, 200, VOICE_UI_WIDTH, VOICE_UI_HEIGHT),
        style,
        NSBackingStoreBuffered,
        False,
    )
    window.setTitle_("Victoria")
    window.setLevel_(3)
    view = WaveView.alloc().initWithFrame_(NSMakeRect(0, 0, VOICE_UI_WIDTH, VOICE_UI_HEIGHT))
    window.setContentView_(view)

    window.center()
    window.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(0.15, view, "tick:", None, True)
    app.run()
    return 0


def run_status_window_child():
    try:
        import tkinter as tk
    except Exception as e:
        return run_cocoa_status_window_child(e)

    messages = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    threading.Thread(target=read_stdin, daemon=True).start()

    root = tk.Tk()
    root.title("Victoria")
    root.geometry(f"{VOICE_UI_WIDTH}x{VOICE_UI_HEIGHT}")
    root.resizable(False, False)
    root.configure(bg="#101418")
    root.attributes("-topmost", True)

    canvas = tk.Canvas(root, width=VOICE_UI_WIDTH, height=VOICE_UI_HEIGHT, bg="#101418", highlightthickness=0)
    canvas.pack(fill="both", expand=True)


    state = {
        "name": "idle",
        "title": "Victoria",
        "subtitle": "Di Victoria para comenzar",
        "duration": None,
        "spectrum": victoria_voice_spectrum("Victoria"),
        "started_at": time.monotonic(),
    }
    palette = {
        "idle": ("#4f8cff", "#7da9ff"),
        "listening": ("#1fd1a5", "#7cf4d4"),
        "recording": ("#ff4d6d", "#ff9aae"),
        "processing": ("#f6c445", "#ffe08a"),
        "speaking": ("#b06cff", "#d1a8ff"),
        "error": ("#ff6b4a", "#ffb199"),
    }

    face_memory_tk = FaceMemory() if FACE_MEMORY_ENABLED else None

    # lerp color state (as 0-255 ints for easy hex conversion)
    def hex_to_rgb(h):
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))

    def rgb_to_hex(r, g, b):
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    idle_p, idle_s = palette["idle"]
    lerp = {
        "pr": hex_to_rgb(idle_p), "sr": hex_to_rgb(idle_s),
    }

    # names panel state
    names_panel = {"show": False, "names": [], "item_rects": []}

    def apply_message(message):
        if message.get("state") == "quit":
            root.destroy()
            return
        if message.get("action") == "delete_face":
            name = " ".join((message.get("name") or "").strip().split())
            if face_memory_tk:
                ok, result = face_memory_tk.delete(name)
                if ok:
                    emit_child_event({"type": "face_deleted", "name": result})
                else:
                    emit_child_event({"type": "face_delete_error", "error": result})
            else:
                emit_child_event({"type": "face_delete_error", "error": "memoria facial no disponible"})
            return
        if message.get("action") == "list_faces":
            names = face_memory_tk.list_names() if face_memory_tk else []
            emit_child_event({"type": "face_list", "names": names})
            return

        state["name"] = message.get("state", state["name"])
        state["title"] = message.get("title", state["title"])
        state["subtitle"] = message.get("subtitle", state["subtitle"])
        state["duration"] = message.get("duration")
        state["spectrum"] = message.get("spectrum", state["spectrum"])
        state["started_at"] = time.monotonic()

    def drain_messages():
        while True:
            try:
                apply_message(messages.get_nowait())
            except queue.Empty:
                break

    def on_canvas_click(event):
        if not names_panel["show"]:
            return
        for (x1, y1, x2, y2, name) in names_panel["item_rects"]:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                if face_memory_tk:
                    ok, result = face_memory_tk.delete(name)
                    if ok:
                        emit_child_event({"type": "face_deleted", "name": result})
                    else:
                        emit_child_event({"type": "face_delete_error", "error": result})
                names_panel["names"] = face_memory_tk.list_names() if face_memory_tk else []
                names_panel["item_rects"] = []
                return
        names_panel["show"] = False

    canvas.bind("<Button-1>", on_canvas_click)

    def draw():
        drain_messages()
        canvas.delete("all")
        width = VOICE_UI_WIDTH
        height = VOICE_UI_HEIGHT

        # lerp colors toward target
        tp, ts = palette.get(state["name"], palette["idle"])
        tr, tg, tb = hex_to_rgb(tp)
        sr2, sg2, sb2 = hex_to_rgb(ts)
        alpha = 0.14
        pr, pg, pb = lerp["pr"]
        srr, srg, srb = lerp["sr"]
        lerp["pr"] = (pr + (tr - pr) * alpha, pg + (tg - pg) * alpha, pb + (tb - pb) * alpha)
        lerp["sr"] = (srr + (sr2 - srr) * alpha, srg + (sg2 - srg) * alpha, srb + (sb2 - srb) * alpha)
        primary = rgb_to_hex(*lerp["pr"])
        secondary = rgb_to_hex(*lerp["sr"])

        canvas.create_text(width / 2, 38, text=state["title"], fill="#f5f7fa", font=("Helvetica", 22, "bold"))
        canvas.create_text(width / 2, 68, text=state["subtitle"], fill="#aeb7c2", font=("Helvetica", 13))

        if state["name"] == "speaking":
            base_spectrum = state.get("spectrum") or victoria_voice_spectrum("Victoria")
        else:
            base_spectrum = status_spectrum(state["name"], state["title"], state["subtitle"])
        spectrum = animated_bars(base_spectrum, state["name"])

        center_x = width / 2
        center_y = height * 0.39
        face_size = min(390, height * 0.52)
        canvas.create_oval(
            center_x - face_size / 2, center_y - face_size / 2,
            center_x + face_size / 2, center_y + face_size / 2,
            fill="#0a0f12", outline=primary, width=4,
        )
        inset = face_size * 0.13
        canvas.create_oval(
            center_x - face_size / 2 + inset, center_y - face_size / 2 + inset,
            center_x + face_size / 2 - inset, center_y + face_size / 2 - inset,
            outline=secondary, width=2,
        )
        eye_w = face_size * 0.18
        eye_h = face_size * 0.08
        eye_y = center_y - face_size * 0.08
        canvas.create_oval(center_x - face_size * 0.25, eye_y, center_x - face_size * 0.25 + eye_w, eye_y + eye_h, fill=secondary, outline="")
        canvas.create_oval(center_x + face_size * 0.07, eye_y, center_x + face_size * 0.07 + eye_w, eye_y + eye_h, fill=secondary, outline="")
        if state["name"] in {"recording", "speaking"}:
            canvas.create_oval(center_x - face_size * 0.16, center_y + face_size * 0.16, center_x + face_size * 0.16, center_y + face_size * 0.34, fill=primary, outline="")
        else:
            canvas.create_arc(
                center_x - face_size * 0.24, center_y + face_size * 0.08,
                center_x + face_size * 0.24, center_y + face_size * 0.38,
                start=200, extent=140, style="arc", outline=primary, width=5,
            )
        canvas.create_text(width / 2, center_y + face_size * 0.55, text=state["subtitle"], fill=secondary, font=("Helvetica", 17))
        canvas.create_text(width / 2, height - 112, text="Humano", fill="#eef2f6", font=("Helvetica", 28, "bold"))

        def draw_equalizer(x0, y0, eq_width, eq_height, values):
            bar_count = min(16, len(values))
            gap = 5
            bar_width = max(4, (eq_width - gap * (bar_count - 1)) / bar_count)
            for index in range(bar_count):
                bar_height = max(8, values[index] * (eq_height - 16))
                x = x0 + index * (bar_width + gap)
                canvas.create_rectangle(x, y0 + eq_height - bar_height, x + bar_width, y0 + eq_height, fill=primary, outline="")
                canvas.create_rectangle(x, y0 + eq_height - bar_height - 8, x + bar_width, y0 + eq_height - bar_height - 3, fill=secondary, outline="")

        eq_w = max(250, min(330, (width - face_size - 280) / 2))
        eq_h = 205
        eq_y = height - eq_h - 54
        draw_equalizer(70, eq_y, eq_w, eq_h, spectrum[:16])
        draw_equalizer(width - 70 - eq_w, eq_y, eq_w, eq_h, list(reversed(spectrum[:16])))

        # names panel overlay
        if names_panel["show"]:
            names_panel["item_rects"] = []
            pw, ph = min(500, width * 0.55), 0
            names = names_panel["names"]
            ph = min(500, max(180, 80 + len(names) * 54))
            px, py = (width - pw) / 2, (height - ph) / 2
            canvas.create_rectangle(px, py, px + pw, py + ph, fill="#0f1318", outline=primary, width=2)
            canvas.create_text(px + pw / 2, py + 28, text="Enrolamientos guardados", fill="#f5f7fa", font=("Helvetica", 17, "bold"))
            if not names:
                canvas.create_text(px + pw / 2, py + ph / 2, text="No hay personas guardadas.", fill="#6b7280", font=("Helvetica", 14))
            for i, name in enumerate(names):
                iy = py + 58 + i * 52
                canvas.create_rectangle(px + 16, iy, px + pw - 16, iy + 42, fill="#1a2030", outline="")
                canvas.create_text(px + 36, iy + 21, text=name, fill="#e5e7eb", font=("Helvetica", 14), anchor="w")
                bx1, by1, bx2, by2 = px + pw - 66, iy + 7, px + pw - 22, iy + 35
                canvas.create_rectangle(bx1, by1, bx2, by2, fill="#cc2233", outline="")
                canvas.create_text((bx1 + bx2) / 2, (by1 + by2) / 2, text="✕", fill="white", font=("Helvetica", 13, "bold"))
                names_panel["item_rects"].append((px + 16, iy, px + pw - 16, iy + 42, name))
            canvas.create_text(px + pw / 2, py + ph - 14, text="Toca afuera para cerrar", fill="#374151", font=("Helvetica", 11))

        root.after(150, draw)

    draw()
    root.mainloop()
    return 0


def play_wake_sound():
    try:
        subprocess.run(["afplay", WAKE_SOUND_FILE], check=True, timeout=SOUND_TIMEOUT_SECONDS)
    except Exception:
        # Si falla el sonido del sistema, se ignora y se continúa
        pass


def enter_activation_pressed():
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except Exception:
        return False

    if not readable:
        return False

    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return True


def activate_from_enter():
    print("Victoria activada por Enter.")
    set_voice_status("idle", "Victoria", "Activada por Enter")
    play_recording_sound()


def greet_recognized_face_if_any():
    if status_window is None:
        return
    name = status_window.get_latest_recognized_name()
    if not name or name in _greeted_names:
        return
    _greeted_names.add(name)
    print(f"Victoria reconoce a: {name}")
    set_voice_status("speaking", f"Hola {name}", "Rostro reconocido")
    speak_system_message(f"Hola {name}. ¿Necesitas consultar algo?", voice=PROCESSING_VOICE, rate=RECORDING_PROMPT_RATE)
    set_voice_status("listening", "Victoria en escucha", "Di Victoria o presiona Enter")


def wait_for_wakeword_sounddevice(recognizer):
    print("Wake word usando sounddevice. Di 'Victoria' o presiona Enter para despertarme.")
    set_voice_status("listening", "Victoria en escucha", "Di Victoria o presiona Enter")
    consecutive_errors = 0

    while True:
        try:
            if status_window and status_window.get_enroll_request():
                return "enroll"
            if status_window and status_window.get_delete_request():
                return "delete_enroll"

            if enter_activation_pressed():
                activate_from_enter()
                return "wakeword"

            frames = int(WAKEWORD_PHRASE_TIME_LIMIT_SECONDS * WAKEWORD_SAMPLE_RATE)
            recording = sd.rec(frames, samplerate=WAKEWORD_SAMPLE_RATE, channels=1, dtype="int16", device=MIC_INDEX)
            wait_for_recording(WAKEWORD_PHRASE_TIME_LIMIT_SECONDS + RECORDING_TIMEOUT_PADDING_SECONDS)
            audio = np.asarray(recording, dtype=np.int16).reshape(-1)
            if not audio.size:
                continue

            rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
            peak = int(np.max(np.abs(audio)))
            if rms < WAKEWORD_MIN_RMS and peak < WAKEWORD_MIN_PEAK:
                greet_recognized_face_if_any()
                time.sleep(0.1)
                continue

            audio_data = sr.AudioData(audio.tobytes(), WAKEWORD_SAMPLE_RATE, 2)
            text = recognizer.recognize_google(audio_data, language="es-ES").lower()
            consecutive_errors = 0
            if "victoria" in text:
                print("Victoria despierta.")
                set_voice_status("idle", "Victoria despierta", "Preparando grabacion")
                play_recording_sound()
                return "wakeword"
        except sr.UnknownValueError:
            continue
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors == 1 or consecutive_errors % 5 == 0:
                print(f"\nAviso: fallo temporal al escuchar la palabra de activacion con sounddevice ({e}). Sigo escuchando.")
            time.sleep(WAKEWORD_ERROR_SLEEP_SECONDS)


def wait_for_wakeword():
    if sr is None:
        set_voice_status("idle", "Victoria", "Presiona Enter para hablar")
        input("\nSpeechRecognition no esta instalado. Presiona Enter para hablar...")
        return "wakeword"

    recognizer = sr.Recognizer()
    recognizer.operation_timeout = GOOGLE_ASR_TIMEOUT_SECONDS
    print("\nEn reposo. Di 'Victoria' o presiona Enter para despertarme. Ctrl+C para salir.")
    set_voice_status("listening", "Victoria en escucha", "Di Victoria o presiona Enter")

    try:
        microphone = sr.Microphone(device_index=MIC_INDEX)
    except Exception as e:
        print(f"\nAviso: no se pudo usar PyAudio para wake word ({e}). Uso sounddevice como fallback.")
        return wait_for_wakeword_sounddevice(recognizer)

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        consecutive_errors = 0

        while True:
            try:
                if status_window and status_window.get_enroll_request():
                    return "enroll"
                if status_window and status_window.get_delete_request():
                    return "delete_enroll"

                if enter_activation_pressed():
                    activate_from_enter()
                    return "wakeword"

                audio = recognizer.listen(
                    source,
                    timeout=WAKEWORD_LISTEN_TIMEOUT_SECONDS,
                    phrase_time_limit=WAKEWORD_PHRASE_TIME_LIMIT_SECONDS,
                )
                text = recognizer.recognize_google(audio, language="es-ES").lower()
                consecutive_errors = 0
                if "victoria" in text:
                    print("Victoria despierta.")
                    set_voice_status("idle", "Victoria despierta", "Preparando grabacion")
                    play_recording_sound()
                    return "wakeword"
            except sr.WaitTimeoutError:
                greet_recognized_face_if_any()
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors == 1 or consecutive_errors % 5 == 0:
                    print(f"\nAviso: fallo temporal al escuchar la palabra de activacion ({e}). Sigo escuchando.")
                time.sleep(WAKEWORD_ERROR_SLEEP_SECONDS)


def is_media_trigger_key(key) -> bool:
    key_name = getattr(key, "name", "") or str(key)
    key_name = key_name.lower()
    return any(
        token in key_name
        for token in (
            "media",
            "media_play_pause",
            "play_pause",
            "media_play",
            "media_next",
            "media_previous",
            "volume_up",
            "volume_down",
            "volume_mute",
        )
    )


def describe_key(key) -> str:
    key_name = getattr(key, "name", None)
    return f"{key!r} name={key_name!r}"


def debug_keys():
    if keyboard is None:
        print("pynput no esta instalado; no puedo escuchar teclas del manos libres.")
        return

    print("Presiona el boton del manos libres. Ctrl+C para salir.")

    def on_press(key):
        trigger = "TRIGGER" if is_media_trigger_key(key) else "ignore"
        print(f"{trigger}: {describe_key(key)}", flush=True)
        return None

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def wait_for_media_button():
    if keyboard is None:
        print("\npynput no esta instalado; usando Enter como fallback.")
        set_voice_status("idle", "Victoria", "Presiona Enter para grabar")
        input("Presiona Enter para grabar...")
        return

    pressed = threading.Event()

    def on_press(key):
        if is_media_trigger_key(key):
            pressed.set()
            return False
        return None

    print("\nPresiona el boton del manos libres para grabar. Enter tambien sirve.")
    set_voice_status("idle", "Victoria", "Presiona el boton o Enter")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        while not pressed.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable:
                line = sys.stdin.readline()
                if line == "":
                    raise EOFError
                pressed.set()
    finally:
        listener.stop()
        listener.join(timeout=1)


def wait_for_record_trigger(trigger: str):
    if trigger == "media":
        wait_for_media_button()
        return
    set_voice_status("idle", "Victoria", "Presiona Enter para grabar")
    input("\nPresiona Enter para grabar...")


def record_audio(filename="temp_query.wav", duration=5, fs=44100):
    print(f"\nPreparando grabacion por {duration} segundos. Habla al escuchar el pip.")
    speak_recording_prompt()
    play_recording_sound()
    set_voice_status("recording", "Grabando", "Habla al escuchar el pip", duration=duration)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16", device=MIC_INDEX)
    wait_for_recording(duration + RECORDING_TIMEOUT_PADDING_SECONDS)
    sf.write(filename, recording, fs)
    play_recording_sound()
    set_voice_status("processing", "Procesando", "Transcribiendo audio")
    print("Audio grabado.")
    return filename


def transcribe_audio_openai(filename):
    print(f"Transcribiendo audio con OpenAI ({OPENAI_TRANSCRIBE_MODEL})...")
    try:
        if audio_is_too_quiet(filename):
            print("Audio demasiado bajo para transcribir.")
            return ""

        prepared_filename = prepare_audio_for_asr(filename, suffix="openai")
        with open(prepared_filename, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=OPENAI_TRANSCRIBE_MODEL,
                file=audio_file,
                language="es",
                prompt=(
                    "Victoria es una asistente para consultar eventos recientes. "
                    "El usuario suele pedir resumenes, eventos, incidentes, reportes, "
                    "ultimos minutos, ultimas horas, hoy o ayer."
                ),
            )
        text = clean_transcript(response.text)
        print(f"Tu (OpenAI): '{text}'")
        return text
    except Exception as e:
        print(f"Error en transcripcion OpenAI: {e}")
        return ""


def prepare_audio_for_asr(filename, suffix="asr"):
    try:
        audio, sample_rate = sf.read(filename, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = np.nan_to_num(audio)
        audio = audio - float(np.mean(audio))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 0.0001:
            return filename

        audio = audio / peak * 0.92
        abs_audio = np.abs(audio)
        threshold = max(0.008, float(np.percentile(abs_audio, 90)) * 0.10)
        voiced = np.flatnonzero(abs_audio > threshold)
        if voiced.size:
            padding = int(sample_rate * 0.35)
            start = max(0, int(voiced[0]) - padding)
            end = min(len(audio), int(voiced[-1]) + padding)
            audio = audio[start:end]

        prepared_filename = os.path.splitext(filename)[0] + f"_{suffix}.wav"
        sf.write(prepared_filename, audio, sample_rate, subtype="PCM_16")
        return prepared_filename
    except Exception as e:
        print(f"Aviso: no se pudo preparar audio para ASR: {e}")
        return filename


def audio_is_too_quiet(filename):
    try:
        audio, _ = sf.read(filename, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.nan_to_num(audio)
        if not audio.size:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        return rms < 0.0015 and peak < 0.02
    except Exception as e:
        print(f"Aviso: no se pudo medir volumen de audio: {e}")
        return False


def clean_transcript(text):
    text = (text or "").strip().strip('"').strip("'").strip()
    if not text:
        return ""

    lowered = text.lower()
    if any(marker.lower() in lowered for marker in BAD_TRANSCRIPT_MARKERS):
        return ""

    return text


def transcribe_audio_gemini(filename):
    if gemini_client is None:
        print("ASR Gemini no disponible: instala google-genai y define GEMINI_API_KEY.")
        return ""

    print(f"Transcribiendo audio con Gemini Flash ({GEMINI_TRANSCRIBE_MODEL})...")
    try:
        if audio_is_too_quiet(filename):
            print("Audio demasiado bajo para transcribir.")
            return ""

        prepared_filename = prepare_audio_for_asr(filename, suffix="gemini")
        audio_file = gemini_client.files.upload(file=prepared_filename)
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_TRANSCRIBE_MODEL,
                config=types.GenerateContentConfig(temperature=0) if types else None,
                contents=[
                    (
                        "Transcribe exactamente la voz humana en este audio. El idioma esperado es español "
                        "latinoamericano, probablemente chileno. No traduzcas, no corrijas la intención y no "
                        "agregues comentarios. Si no hay habla clara, devuelve exactamente [NO_SPEECH]. "
                        "Contexto: el usuario le habla a Victoria para pedir resumenes de eventos, incidentes, "
                        "reportes o actividad reciente; puede decir frases como revisa los eventos, dame un "
                        "resumen, ultimos minutos, ultimas horas, hoy o ayer. "
                        "Devuelve solo la transcripción."
                    ),
                    audio_file,
                ],
            )
            text = clean_transcript(response.text)
            print(f"Tu: '{text}'")
            return text
        finally:
            gemini_client.files.delete(name=audio_file.name)
    except Exception as e:
        print(f"Error en transcripcion Gemini: {e}")
        return ""


def pick_google_transcript(result):
    alternatives = result.get("alternative") or []
    if not alternatives:
        return ""

    best = max(alternatives, key=lambda item: item.get("confidence", 0.0))
    return clean_transcript(best.get("transcript"))


def transcribe_audio_google(filename):
    if sr is None:
        print("ASR Google no disponible: instala SpeechRecognition.")
        return ""

    print("Transcribiendo audio con Google ASR optimizado...")
    recognizer = sr.Recognizer()
    recognizer.operation_timeout = GOOGLE_ASR_TIMEOUT_SECONDS
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 180
    prepared_filename = prepare_audio_for_asr(filename, suffix="google")

    try:
        with sr.AudioFile(prepared_filename) as source:
            audio = recognizer.record(source)

        for language in GOOGLE_ASR_LANGUAGES:
            try:
                result = recognizer.recognize_google(audio, language=language, show_all=True)
                text = pick_google_transcript(result)
                if text:
                    print(f"Tu ({language}): '{text}'")
                    return text
            except sr.UnknownValueError:
                continue

        print("Google ASR no entendio el audio.")
        return ""
    except sr.UnknownValueError:
        print("Google ASR no entendio el audio.")
        return ""
    except Exception as e:
        print(f"Error en Google ASR: {e}")
        return ""


def transcribe_audio_local_whisper(filename):
    global local_whisper_model

    if WhisperModel is None:
        print("Whisper local no disponible: instala faster-whisper.")
        return ""

    print(f"Transcribiendo localmente con faster-whisper ({LOCAL_ASR_MODEL})...")
    try:
        if local_whisper_model is None:
            local_whisper_model = WhisperModel(LOCAL_ASR_MODEL, device="auto", compute_type="auto")

        segments, _ = local_whisper_model.transcribe(filename, language="es", vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        print(f"Tu: '{text}'")
        return text
    except Exception as e:
        print(f"Whisper local fallo: {e}")
        return ""


def transcribe_audio_local_sphinx(filename):
    if sr is None:
        print("ASR local no disponible: instala SpeechRecognition y pocketsphinx.")
        return ""

    print("Transcribiendo audio localmente con Sphinx...")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_sphinx(audio, language="es-ES").strip()
        print(f"Tu: '{text}'")
        return text
    except Exception as e:
        print(f"ASR local fallo: {e}")
        return ""


def transcribe_audio_local(filename):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"ASR local no termino dentro de {LOCAL_ASR_TIMEOUT_SECONDS}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(LOCAL_ASR_TIMEOUT_SECONDS)

    try:
        text = transcribe_audio_local_whisper(filename)
        if text:
            return text

        return transcribe_audio_local_sphinx(filename)
    except TimeoutError as e:
        print(str(e))
        return ""
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def wait_for_recording(timeout_seconds):
    done = threading.Event()
    error = []

    def wait_for_device():
        try:
            sd.wait()
        except Exception as e:
            error.append(e)
        finally:
            done.set()

    thread = threading.Thread(target=wait_for_device, daemon=True)
    thread.start()
    if not done.wait(timeout_seconds):
        sd.stop()
        raise TimeoutError(f"La grabacion no termino dentro de {timeout_seconds:.1f}s")

    if error:
        raise error[0]


ASR_PROVIDERS = ("auto", "openai", "gemini", "google", "local")


def get_auto_asr_providers():
    providers = []
    if gemini_client is not None:
        providers.append("gemini")
    if sr is not None:
        providers.append("google")
    if WhisperModel is not None:
        providers.append("local")
    providers.append("openai")
    return providers


def transcribe_audio_with_provider(filename, asr_provider):
    if asr_provider == "gemini":
        return transcribe_audio_gemini(filename)
    if asr_provider == "google":
        return transcribe_audio_google(filename)
    if asr_provider == "local":
        return transcribe_audio_local(filename)
    return transcribe_audio_openai(filename)


def transcribe_audio(filename, asr_provider):
    if asr_provider == "auto":
        for provider in get_auto_asr_providers():
            text = transcribe_audio_with_provider(filename, provider)
            if text:
                return text
        return ""

    return transcribe_audio_with_provider(filename, asr_provider)


def run_tool_call(tool_call, user_prompt: str):
    if tool_call.function.name != "consultar_omnistatus":
        return f"Tool desconocida: {tool_call.function.name}"

    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        return f"Argumentos invalidos para tool call: {e}"

    minutos = int(args.get("minutos", 60))
    prompt = user_prompt.strip()
    if not prompt:
        prompt = (args.get("prompt") or "").strip()
    return consultar_omnistatus(minutos=minutos, prompt=prompt)


def ask_openai(prompt_text, minutes=60):
    print(f"Analizando intencion con OpenAI ({OPENAI_VOICE_MODEL}) y function calling...")

    messages = [
        {
            "role": "system",
            "content": (
                "Eres Victoria, una asistente de voz inteligente, concisa y conversacional. "
                "Si el usuario pregunta por eventos, reportes, resumenes, incidentes o actividad reciente, "
                "usa la herramienta consultar_omnistatus. Debes inferir el parametro minutos desde el texto: "
                "15 minutos = 15, media hora = 30, una hora = 60, dos horas = 120, tres horas = 180, "
                "hoy o el dia = 1440, ayer = 2880. Si el usuario no especifica tiempo, "
                f"asume {minutes} minutos. Cuando llames la herramienta, el campo prompt debe ser exactamente "
                "el texto del usuario, sin resumirlo, corregirlo ni reescribirlo. "
                "Tus respuestas finales deben ser cortas y naturales para voz."
            ),
        },
        {"role": "user", "content": prompt_text},
    ]

    try:
        for _ in range(4):
            response = client.chat.completions.create(
                model=OPENAI_VOICE_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                result_text = (message.content or "No logre estructurar una respuesta.").strip()
                print(f"Victoria: {result_text}")
                return result_text

            for tool_call in message.tool_calls:
                tool_result = run_tool_call(tool_call, prompt_text)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        return "Necesité demasiados pasos para responder. Intenta hacer la consulta más directa."
    except Exception as e:
        print(f"Error al consultar OpenAI: {e}")
        return "Hubo un error al comunicarme con el cerebro de Victoria."


def speak_text(text):
    print("Generando voz y reproduciendo...")
    try:
        tts_filename = "temp_response.mp3"
        with client.audio.speech.with_streaming_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            instructions=OPENAI_TTS_INSTRUCTIONS,
            speed=OPENAI_TTS_SPEED,
            timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response.stream_to_file(tts_filename)
        playback_timeout = max(10.0, len(text) * 0.09 + TTS_PLAYBACK_TIMEOUT_PADDING_SECONDS)
        subprocess.run(["afplay", tts_filename], check=False, timeout=playback_timeout)
    except Exception as e:
        print(f"Error en TTS: {e}")


def maybe_greet_recognized_person():
    if status_window is None:
        return None

    # Si ya saludamos a alguien durante el wakeword, ese nombre cuenta
    if _greeted_names:
        return next(iter(_greeted_names))

    name = status_window.get_latest_recognized_name()
    if not name:
        return None

    _greeted_names.add(name)
    greeting = f"Hola {name}."
    print(f"Victoria reconoce a: {name}")
    set_voice_status("speaking", f"Hola {name}", "Rostro reconocido")
    speak_system_message(greeting, voice=PROCESSING_VOICE, rate=RECORDING_PROMPT_RATE)
    return name


def _record_short_answer(asr_provider, duration=4):
    play_recording_sound()
    try:
        tmp = "temp_enroll_answer.wav"
        recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype="int16", device=MIC_INDEX)
        wait_for_recording(duration + RECORDING_TIMEOUT_PADDING_SECONDS)
        sf.write(tmp, recording, 44100)
    except Exception as e:
        print(f"Error grabando respuesta: {e}")
        return ""
    play_recording_sound()
    return transcribe_audio(tmp, asr_provider).strip().strip(".,!?¿¡").strip()


def _is_affirmative(text):
    lowered = text.lower()
    return any(w in lowered for w in ("sí", "si", "yes", "dale", "claro", "actualiza", "actualizar", "ok", "bueno"))


def _do_enroll(name, updating=False):
    set_voice_status("processing", "Recordando rostro", f"Tomando {FACE_MEMORY_SAMPLES} fotos de {name}")
    if not status_window.enroll_face(name):
        message = f"No pude acceder a la camara para guardar tu imagen, {name}. Asegurate de estar bien visible e intentalo de nuevo."
        print(f"Victoria: {message}")
        set_voice_status("error", "Error de camara", "Verifica que la camara este activa")
        speak_system_message(message, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
        return
    event = status_window.wait_for_enroll_result(timeout_seconds=18)
    if event.get("type") == "face_enrolled":
        remembered_name = event.get("name") or name
        if updating:
            message = (
                f"Listo, {remembered_name}. Tu imagen ha sido actualizada correctamente. "
                "La proxima vez que te vea te reconocere mejor."
            )
        else:
            message = (
                f"Listo, {remembered_name}. He guardado tu imagen con exito. "
                "La proxima vez que te vea te saludare por tu nombre."
            )
        print(f"Victoria: {message}")
        set_voice_status("speaking", "Imagen guardada" if not updating else "Imagen actualizada", remembered_name)
        speak_system_message(message, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
    else:
        error = event.get("error") or "error desconocido"
        message = (
            f"Lo siento, {name}. No pude guardar tu imagen esta vez. "
            "Asegurate de estar de frente a la camara con buena iluminacion e intentalo de nuevo."
        )
        print(f"Victoria: {message} (detalle: {error})")
        set_voice_status("error", "No pude guardar la imagen", "Reintenta con la cara visible")
        speak_system_message(message, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)


def _delete_enroll_by_voice(asr_provider="openai"):
    if status_window is None:
        return

    status_window.list_faces()
    names = status_window.wait_for_face_list()
    if not names:
        message = "No hay personas guardadas en memoria."
        speak_system_message(message, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
        set_voice_status("idle", "Victoria", "Lista para la siguiente consulta")
        return

    names_str = ", ".join(names)
    prompt = f"Tengo guardados: {names_str}. Di el nombre que quieres borrar."
    set_voice_status("processing", "Borrar enrolamiento", names_str)
    speak_system_message(prompt, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
    set_voice_status("recording", "Escuchando nombre", "Di el nombre a borrar", duration=4)
    raw = _record_short_answer(asr_provider)
    name = " ".join(raw.strip().strip(".,!?¿¡").split()[:3])
    if not name:
        speak_system_message("No entendi el nombre. Operacion cancelada.", voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
        set_voice_status("idle", "Victoria", "Lista para la siguiente consulta")
        return

    set_voice_status("processing", "Borrando", f"Eliminando a {name}")
    status_window.delete_face(name)
    event = status_window.wait_for_delete_result()
    if event.get("type") == "face_deleted":
        deleted_name = event.get("name") or name
        message = f"Listo. He eliminado a {deleted_name} de mi memoria. Ya no lo reconocere."
        speak_system_message(message, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
        set_voice_status("idle", "Victoria", "Lista para la siguiente consulta")
    else:
        error = event.get("error") or "error desconocido"
        message = f"No pude borrar a {name}. {error}."
        speak_system_message(message, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
        set_voice_status("error", "No pude borrar", error)


def _say_farewell():
    message = "Hasta luego. Fue un placer ayudarte."
    print(f"Victoria: {message}")
    set_voice_status("speaking", "Hasta luego", "")
    speak_system_message(message, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
    set_voice_status("idle", "Victoria", "Lista para la siguiente consulta")


def _ask_name_and_enroll(asr_provider, updating=False):
    prompt = "Di tu nombre."
    set_voice_status("recording", "Escuchando nombre", "Di tu nombre ahora", duration=4)
    speak_system_message(prompt, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
    raw_name = _record_short_answer(asr_provider)
    name = " ".join(raw_name.strip().strip(".,!?¿¡").split()[:3])
    if not name:
        message = "No pude entender tu nombre. Puedes intentarlo la proxima vez."
        speak_system_message(message, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
        set_voice_status("idle", "Victoria", "Lista para la siguiente consulta")
        return
    print(f"Nombre reconocido: '{name}'")
    _do_enroll(name, updating=updating)


def maybe_enroll_person_after_response(recognized_name, asr_provider="openai"):
    if not FACE_MEMORY_ASK_ENROLL or status_window is None:
        return

    if recognized_name:
        prompt = f"¿Quieres actualizar tu imagen, {recognized_name}?"
        set_voice_status("processing", "Memoria facial", "¿Actualizar imagen?")
        speak_system_message(prompt, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
        set_voice_status("recording", "Escuchando respuesta", "Di si o no", duration=4)
        answer = _record_short_answer(asr_provider)
        if not _is_affirmative(answer):
            _say_farewell()
            return
        _ask_name_and_enroll(asr_provider, updating=True)
        return

    prompt = "¿Quieres que guarde tu imagen para recordarte?"
    set_voice_status("processing", "Memoria facial", "¿Guardar imagen?")
    speak_system_message(prompt, voice=PROCESSING_VOICE, rate=PROCESSING_RATE)
    set_voice_status("recording", "Escuchando respuesta", "Di si o no", duration=4)
    answer = _record_short_answer(asr_provider)
    if not _is_affirmative(answer):
        _say_farewell()
        return
    _ask_name_and_enroll(asr_provider, updating=False)


def main():
    global status_window

    parser = argparse.ArgumentParser(description="Victoria Voice Interface")
    parser.add_argument("-d", "--duration", type=int, default=6, help="Duracion de la grabacion en segundos.")
    parser.add_argument("-m", "--minutes", type=int, default=DEFAULT_QUERY_MINUTES, help="Minutos por defecto para consultar eventos.")
    parser.add_argument(
        "--asr",
        choices=ASR_PROVIDERS,
        default=os.getenv("OMNI_ASR", "openai"),
        help="Motor de transcripcion. 'auto' prueba proveedores disponibles y cae a OpenAI.",
    )
    parser.add_argument(
        "--trigger",
        choices=["enter", "media"],
        default=os.getenv("OMNI_TRIGGER", "enter"),
        help="Como iniciar la grabacion cuando se usa --no-wakeword.",
    )
    parser.add_argument(
        "--debug-keys",
        action="store_true",
        help="Muestra las teclas que llegan desde el teclado o manos libres y sale con Ctrl+C.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="No abre la ventana visual.",
    )
    parser.add_argument(
        "--no-face-ui",
        action="store_true",
        help="No abre la ventana de webcam con deteccion de rostros.",
    )
    parser.add_argument(
        "--status-window-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-wakeword", action="store_true", help="Salta la palabra de activacion y graba directo.")
    args = parser.parse_args()

    if args.status_window_child:
        raise SystemExit(run_status_window_child())

    if args.debug_keys:
        debug_keys()
        return

    status_window = VoiceStatusController(
        enabled=VOICE_UI_ENABLED and not args.no_ui,
        face_enabled=FACE_UI_ENABLED and not args.no_face_ui,
    )
    status_window.start()

    print("========================================")
    print("Victoria Voice Interface Activada")
    print("========================================")
    print(f"Configuracion: escucha={args.duration}s, asr={args.asr}, trigger={args.trigger}.")

    try:
        while True:
            processing_process = None
            try:
                if args.no_wakeword:
                    wait_for_record_trigger(args.trigger)
                else:
                    wait_for_wakeword()

                audio_file = record_audio(duration=args.duration)
                processing_process = start_processing_message()
                set_voice_status("processing", "Procesando", "Transcribiendo y entendiendo tu pregunta")
                text = transcribe_audio(audio_file, args.asr)
                if not text.strip():
                    stop_processing_message(processing_process)
                    set_voice_status("error", "No escuche claro", "Reintenta: di Victoria o presiona Enter")
                    print("No se escucho nada claro. Intenta de nuevo.")
                    continue

                set_voice_status("processing", "Procesando consulta", "Consultando Omnistatus")
                response_text = ask_openai(text, minutes=args.minutes)
                stop_processing_message(processing_process)
                set_voice_status(
                    "speaking",
                    "Victoria hablando",
                    "Espectro de voz",
                    spectrum=victoria_voice_spectrum(response_text),
                )
                speak_text(response_text)
                set_voice_status("idle", "Victoria", "Lista para la siguiente consulta")
            except KeyboardInterrupt:
                stop_processing_message(processing_process)
                print("\nSaliendo de la interfaz de voz...")
                break
            except EOFError:
                stop_processing_message(processing_process)
                print("\nNo hay entrada interactiva disponible para iniciar la grabacion.")
                break
            except Exception as e:
                stop_processing_message(processing_process)
                set_voice_status("error", "Fallo la consulta", "Reintenta en unos segundos")
                print(f"\nError inesperado: {e}")
    finally:
        if status_window is not None:
            status_window.close()
            status_window = None


if __name__ == "__main__":
    main()
