import argparse
import json
import os
import select
import signal
import subprocess
import sys
import threading
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
VICTORIA_APIKEY = os.getenv("VICTORIA_APIKEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TRANSCRIBE_MODEL = os.getenv("GEMINI_TRANSCRIBE_MODEL", "gemini-2.5-flash")
VICTORIA_URL = os.getenv("VICTORIA_URL", "http://localhost:8888/analyze/custom")
RECORDING_SOUND = os.getenv("VICTORIA_RECORDING_SOUND", "/System/Library/Sounds/Ping.aiff")
VICTORIA_API_MAX_CHARS = os.getenv("VICTORIA_API_MAX_CHARS", "200").strip()
DEFAULT_QUERY_MINUTES = int(os.getenv("VICTORIA_DEFAULT_QUERY_MINUTES", "720"))
GOOGLE_ASR_LANGUAGES = [
    lang.strip()
    for lang in os.getenv("VICTORIA_GOOGLE_ASR_LANGUAGES", "es-CL,es-ES,es-419").split(",")
    if lang.strip()
]
LOCAL_ASR_MODEL = os.getenv("VICTORIA_LOCAL_ASR_MODEL", "tiny")
LOCAL_ASR_TIMEOUT_SECONDS = int(os.getenv("VICTORIA_LOCAL_ASR_TIMEOUT_SECONDS", "60"))
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

client = openai.OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
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
            "name": "consultar_servidor_victoria",
            "description": "Consulta la API configurada con la pregunta del usuario y devuelve una respuesta breve.",
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

def normalize_victoria_url(url: str) -> str:
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


def print_api_trace(method: str, url: str, payload: dict, response_data=None):
    if response_data is None:
        print("\n========== Omnistatus Request ==========")
        print(f"{method} {url}")
        print("Content-Type: application/json")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("========================================\n")
        return

    print("\n========== Omnistatus Response =========")
    print(json.dumps(response_data, ensure_ascii=False, indent=2))
    print("========================================\n")


def build_api_prompt(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not VICTORIA_API_MAX_CHARS:
        return prompt

    try:
        max_chars = int(VICTORIA_API_MAX_CHARS)
    except ValueError:
        print(f"Aviso: VICTORIA_API_MAX_CHARS invalido: {VICTORIA_API_MAX_CHARS!r}.")
        return prompt

    if max_chars <= 0:
        return prompt

    return f"{prompt}\n\nResponde en maximo {max_chars} caracteres."


def consultar_servidor_victoria(minutos: int, prompt: str) -> str:
    hours = minutes_to_hours(minutos)
    api_prompt = build_api_prompt(prompt)

    payload = {"hours": hours, "prompt": api_prompt}
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    url = normalize_victoria_url(VICTORIA_URL)

    try:
        print_api_trace("POST", url, payload)
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        print_api_trace("POST", url, payload, data)
        return format_omnistatus_response(data)
    except Exception as e:
        return f"Error en la consulta a Omnistatus: {e}"


def play_recording_sound():
    if not RECORDING_SOUND or not os.path.exists(RECORDING_SOUND):
        return

    try:
        subprocess.run(["afplay", RECORDING_SOUND], check=False)
    except Exception as e:
        print(f"Aviso: no se pudo reproducir el sonido de grabacion: {e}")


def play_wake_sound():
    try:
        subprocess.run(["afplay", WAKE_SOUND_FILE], check=True)
    except Exception:
        # Si falla el sonido del sistema, se ignora y se continúa
        pass


def wait_for_wakeword():
    if sr is None:
        input("\nSpeechRecognition no esta instalado. Presiona Enter para hablar...")
        return

    recognizer = sr.Recognizer()
    print("\nEn reposo. Di 'Victoria' para despertarme, o presiona Ctrl+C para salir.")

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        while True:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                text = recognizer.recognize_google(audio, language="es-ES").lower()
                if "victoria" in text:
                    print("Victoria despierta.")
                    play_recording_sound()
                    return
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                input(f"\nNo pude usar la palabra de activacion ({e}). Presiona Enter para hablar...")
                return


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
        input("Presiona Enter para grabar...")
        return

    pressed = threading.Event()

    def on_press(key):
        if is_media_trigger_key(key):
            pressed.set()
            return False
        return None

    print("\nPresiona el boton del manos libres para grabar. Enter tambien sirve.")
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
    input("\nPresiona Enter para grabar...")


def record_audio(filename="temp_query.wav", duration=5, fs=44100):
    print(f"\nGrabando por {duration} segundos. Habla ahora.")
    play_recording_sound()
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    sf.write(filename, recording, fs)
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
                    "Victoria es una asistente para consultar una API con preguntas del usuario. "
                    "El usuario puede pedir actividad reciente, registros, estado, presencia, "
                    "incidentes, reportes, resumenes, ultimos minutos, ultimas horas, hoy o ayer."
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
                        "Contexto: el usuario le habla a Victoria para consultar una API sobre actividad, "
                        "registros, estado, presencia, incidentes, reportes o periodos recientes; puede decir "
                        "frases con ultimos minutos, ultimas horas, hoy o ayer. "
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


def transcribe_audio(filename, asr_provider):
    return transcribe_audio_openai(filename)


def run_tool_call(tool_call, user_prompt: str):
    if tool_call.function.name != "consultar_servidor_victoria":
        return f"Tool desconocida: {tool_call.function.name}"

    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        return f"Argumentos invalidos para tool call: {e}"

    minutos = int(args.get("minutos", DEFAULT_QUERY_MINUTES))
    prompt = user_prompt.strip()
    if not prompt:
        prompt = (args.get("prompt") or "").strip()
    return consultar_servidor_victoria(minutos=minutos, prompt=prompt)


def ask_openai(prompt_text, minutes=DEFAULT_QUERY_MINUTES):
    print(f"Analizando intencion con OpenAI ({OPENAI_VOICE_MODEL}) y function calling...")

    messages = [
        {
            "role": "system",
            "content": (
                "Eres Victoria, una asistente de voz inteligente, concisa y conversacional. "
                "Si el usuario hace una consulta que deba resolverse con informacion del API configurada, "
                "usa la herramienta consultar_servidor_victoria. Esto incluye actividad reciente, registros, "
                "estado, presencia, reportes, resumenes, incidentes o preguntas sobre un periodo. "
                "Debes inferir el parametro minutos desde el texto: "
                "15 minutos = 15, media hora = 30, una hora = 60, dos horas = 120, tres horas = 180, "
                "hoy o el dia = 1440, ayer = 2880. Si el usuario no especifica tiempo, "
                f"asume {minutes} minutos (12 horas por defecto). Cuando llames la herramienta, el campo prompt debe ser exactamente "
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
        ) as response:
            response.stream_to_file(tts_filename)
        subprocess.run(["afplay", tts_filename], check=False)
    except Exception as e:
        print(f"Error en TTS: {e}")


def main():
    parser = argparse.ArgumentParser(description="Victoria Voice Interface")
    parser.add_argument("-d", "--duration", type=int, default=6, help="Duracion de la grabacion en segundos.")
    parser.add_argument(
        "-m",
        "--minutes",
        type=int,
        default=DEFAULT_QUERY_MINUTES,
        help="Minutos por defecto para consultar eventos cuando la frase no indica rango. Default: 720.",
    )
    parser.add_argument(
        "--asr",
        choices=["openai"],
        default=os.getenv("VICTORIA_ASR", "openai"),
    )
    parser.add_argument(
        "--trigger",
        choices=["enter", "media"],
        default=os.getenv("VICTORIA_TRIGGER", "enter"),
        help="Como iniciar la grabacion cuando se usa --no-wakeword.",
    )
    parser.add_argument(
        "--debug-keys",
        action="store_true",
        help="Muestra las teclas que llegan desde el teclado o manos libres y sale con Ctrl+C.",
    )
    parser.add_argument("--no-wakeword", action="store_true", help="Salta la palabra de activacion y graba directo.")
    args = parser.parse_args()

    if args.debug_keys:
        debug_keys()
        return

    print("========================================")
    print("Victoria Voice Interface Activada")
    print("========================================")
    print(f"Configuracion: escucha={args.duration}s, asr={args.asr}, trigger={args.trigger}.")

    while True:
        try:
            if args.no_wakeword:
                wait_for_record_trigger(args.trigger)
            else:
                wait_for_wakeword()

            audio_file = record_audio(duration=args.duration)
            text = transcribe_audio(audio_file, args.asr)
            if not text.strip():
                print("No se escucho nada claro. Intenta de nuevo.")
                continue

            response_text = ask_openai(text, minutes=args.minutes)
            speak_text(response_text)
        except KeyboardInterrupt:
            print("\nSaliendo de la interfaz de voz...")
            break
        except EOFError:
            print("\nNo hay entrada interactiva disponible para iniciar la grabacion.")
            break
        except Exception as e:
            print(f"\nError inesperado: {e}")


if __name__ == "__main__":
    main()
