import argparse
import json
import os
import signal
import subprocess
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


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_VOICE_MODEL = os.getenv("OPENAI_VOICE_MODEL", "gpt-4o-mini")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "coral")
OPENAI_TTS_INSTRUCTIONS = os.getenv(
    "OPENAI_TTS_INSTRUCTIONS",
    (
        "Habla en español latinoamericano natural, con acento chileno suave y cercano. "
        "Mantén una voz cálida, clara, calmada y conversacional. Evita sonar como locución neutra corporativa."
    ),
)
OPENAI_TTS_SPEED = float(os.getenv("OPENAI_TTS_SPEED", "1.03"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TRANSCRIBE_MODEL = os.getenv("GEMINI_TRANSCRIBE_MODEL", "gemini-2.5-flash")
VICTORIA_APIKEY = os.getenv("VICTORIA_APIKEY")
VICTORIA_URL = os.getenv("VICTORIA_URL", "http://localhost:8888/analyze/on-demand")
RECORDING_SOUND = os.getenv("VICTORIA_RECORDING_SOUND", "/System/Library/Sounds/Ping.aiff")
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
)

if not OPENAI_API_KEY:
    print("Error: falta OPENAI_API_KEY en el archivo .env.")
    raise SystemExit(1)

if not VICTORIA_APIKEY:
    print("Error: falta VICTORIA_APIKEY en el archivo .env.")
    raise SystemExit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if genai and GEMINI_API_KEY else None
local_whisper_model = None

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "consultar_servidor_victoria",
            "description": "Consulta los eventos recientes de Victoria y devuelve un analisis breve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutos": {
                        "type": "integer",
                        "description": "Rango de tiempo en minutos. Ejemplos: ultima hora = 60, dos horas = 120.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Instruccion clara para analizar los eventos segun lo que pidio el usuario.",
                    },
                },
                "required": ["minutos", "prompt"],
            },
        },
    }
]


def consultar_servidor_victoria(minutos: int, prompt: str) -> str:
    print(f"[Function Call] minutos={minutos} | prompt='{prompt}'")

    payload = {"minutes": minutos, "prompt": prompt}
    params = {"apikey": VICTORIA_APIKEY}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(VICTORIA_URL, params=params, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("result") or json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return f"Error en la consulta al servidor Victoria: {e}"


def play_recording_sound():
    if not RECORDING_SOUND or not os.path.exists(RECORDING_SOUND):
        return

    try:
        subprocess.run(["afplay", RECORDING_SOUND], check=False)
    except Exception as e:
        print(f"Aviso: no se pudo reproducir el sonido de grabacion: {e}")


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
        with open(filename, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=OPENAI_TRANSCRIBE_MODEL,
                file=audio_file,
                language="es",
            )
        text = response.text.strip()
        print(f"Tu: '{text}'")
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
    if asr_provider == "gemini":
        text = transcribe_audio_gemini(filename)
        if text:
            return text
        print("Usando Google ASR como fallback de transcripcion.")
        text = transcribe_audio_google(filename)
        if text:
            return text
        print("Usando OpenAI como fallback de transcripcion.")
        return transcribe_audio_openai(filename)

    if asr_provider == "google":
        text = transcribe_audio_google(filename)
        if text:
            return text
        print("Usando OpenAI como fallback de transcripcion.")
        return transcribe_audio_openai(filename)

    if asr_provider == "local":
        text = transcribe_audio_local(filename)
        if text:
            return text
        print("Usando OpenAI como fallback de transcripcion.")

    return transcribe_audio_openai(filename)


def run_tool_call(tool_call):
    if tool_call.function.name != "consultar_servidor_victoria":
        return f"Tool desconocida: {tool_call.function.name}"

    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        return f"Argumentos invalidos para tool call: {e}"

    minutos = int(args.get("minutos", 60))
    prompt = args.get("prompt") or "Resume los eventos relevantes de forma breve."
    return consultar_servidor_victoria(minutos=minutos, prompt=prompt)


def ask_openai(prompt_text, minutes=60):
    print(f"Analizando intencion con OpenAI ({OPENAI_VOICE_MODEL}) y function calling...")

    messages = [
        {
            "role": "system",
            "content": (
                "Eres Victoria, una asistente de voz inteligente, concisa y conversacional. "
                "Si el usuario pregunta por eventos, reportes, resumenes, incidentes o actividad reciente, "
                "usa la herramienta consultar_servidor_victoria. Si el usuario no especifica tiempo, "
                f"asume {minutes} minutos. Tus respuestas finales deben ser cortas y naturales para voz."
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
                tool_result = run_tool_call(tool_call)
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
        response = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            instructions=OPENAI_TTS_INSTRUCTIONS,
            speed=OPENAI_TTS_SPEED,
        )
        tts_filename = "temp_response.mp3"
        response.stream_to_file(tts_filename)
        subprocess.run(["afplay", tts_filename], check=False)
    except Exception as e:
        print(f"Error en TTS: {e}")


def main():
    parser = argparse.ArgumentParser(description="Victoria Voice Interface")
    parser.add_argument("-d", "--duration", type=int, default=6, help="Duracion de la grabacion en segundos.")
    parser.add_argument("-m", "--minutes", type=int, default=60, help="Minutos por defecto para consultar eventos.")
    parser.add_argument(
        "--asr",
        choices=["gemini", "google", "openai", "local"],
        default=os.getenv("VICTORIA_ASR", "gemini"),
    )
    parser.add_argument("--no-wakeword", action="store_true", help="Salta la palabra de activacion y graba directo.")
    args = parser.parse_args()

    print("========================================")
    print("Victoria Voice Interface Activada")
    print("========================================")
    print(f"Configuracion: escucha={args.duration}s, eventos={args.minutes}m, asr={args.asr}.")

    while True:
        try:
            if args.no_wakeword:
                input("\nPresiona Enter para grabar...")
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
        except Exception as e:
            print(f"\nError inesperado: {e}")


if __name__ == "__main__":
    main()
