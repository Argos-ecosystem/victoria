import os
import json
import sounddevice as sd
import soundfile as sf
import subprocess
from dotenv import load_dotenv
import openai
import numpy as np
import argparse
import requests
from google import genai
from google.genai import types
import speech_recognition as sr

# Configuración
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VICTORIA_APIKEY = os.getenv("VICTORIA_APIKEY")
VICTORIA_URL = os.getenv("VICTORIA_URL", "http://localhost:8888/analyze/on-demand")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ Error: Falta OPENAI_API_KEY en el archivo .env (Requerido para el Text-to-Speech)")
    exit(1)

if not GEMINI_API_KEY:
    print("❌ Error: Falta GEMINI_API_KEY en el archivo .env")
    exit(1)

if not VICTORIA_APIKEY:
    print("❌ Error: Falta VICTORIA_APIKEY en el archivo .env (Necesario para que la Tool llame a Victoria)")
    exit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# --- Definición de Tool para Gemini ---
def consultar_servidor_victoria(minutos: int, prompt: str) -> str:
    """
    Consulta la base de datos de eventos recientes del sistema a través del servidor.
    
    Args:
        minutos: El rango de tiempo en minutos. Extrae y calcula esto matemáticamente basado en lo que pide el usuario (ej: "última hora" = 60, "2 horas" = 120, "10 minutos" = 10).
        prompt: La instrucción que el servidor usará para analizar los eventos. Debe ser una directiva clara adaptada a lo que pidió el usuario (ej: "Haz un resumen general" o "Filtra incidentes de seguridad").
    """
    print(f"📡  [Function Call] Gemini extrajo parámetros -> minutos={minutos} | prompt='{prompt}'")
    
    payload = {"minutes": minutos, "prompt": prompt}
    params = {"apikey": VICTORIA_APIKEY}
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(VICTORIA_URL, params=params, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json().get("result", "Sin resultados.")
    except Exception as e:
        return f"Error en la consulta al servidor: {e}"

print("� Preparando la interfaz de voz de Victoria...")

def wait_for_wakeword():
    recognizer = sr.Recognizer()
    print("\n💤 En modo reposo. Di 'Victoria' para despertarme (o presiona Ctrl+C para salir)...")
    
    # Usamos el micrófono por defecto en modo escucha pasiva
    with sr.Microphone() as source:
        # Ajuste dinámico rápido para ignorar el ruido de fondo
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        while True:
            try:
                # Escucha en fragmentos cortos (máximo 3 segundos de habla)
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                # Motor gratuito de Google para escaneo rápido de la palabra clave
                texto = recognizer.recognize_google(audio, language="es-ES").lower()
                
                if "victoria" in texto:
                    print("✨ ¡Dígame, Sr. Aguilera!")
                    # Responde con la voz de Victoria antes de empezar a grabar
                    speak_text("Dígame, señor Aguilera.")
                    return
            except sr.WaitTimeoutError:
                continue  # Silencio, sigue esperando
            except sr.UnknownValueError:
                continue  # Hubo ruido pero no palabras claras, sigue esperando
            except Exception as e:
                # Fallback de seguridad por si no hay internet
                input(f"\n⚠️ Fallo en red ({e}). Presiona Enter para hablar...")
                return

def record_audio(filename="temp_query.wav", duration=5, fs=44100):
    print(f"\n🎙️  Escuchando por {duration} segundos... (¡Habla ahora!)")
    # Grabamos en 1 canal (mono) a 44100 Hz
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Esperar a que terminen los 5 segundos
    sf.write(filename, recording, fs)
    print("✅  Audio grabado.")
    return filename

def transcribe_audio(filename):
    print("⚡  Transcribiendo audio en la nube con Gemini 2.5 Flash...")
    try:
        # 1. Subir el archivo de audio a los servidores de Gemini
        audio_file = gemini_client.files.upload(file=filename)
        
        # 2. Pedirle que lo transcriba usando un modelo básico (sin las tools)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Transcribe exactamente lo que se dice en este audio en español. Solo devuelve la transcripción, sin comillas ni comentarios adicionales.", 
                audio_file
            ]
        )
        texto = response.text.strip()
        print(f"🗣️  Tú: '{texto}'")
        
        # 3. Limpiar/Borrar el archivo subido para no ocupar espacio en tu cuota de Google
        gemini_client.files.delete(name=audio_file.name)
        
        return texto
    except Exception as e:
        print(f"❌ Error en transcripción con Gemini: {e}")
        return ""


def ask_gemini(prompt_text, minutes=60):
    print("🧠  Analizando intención con Gemini (Auto Function Calling)...")
    try:
        # Iniciamos el chat y le damos permiso de usar su herramienta automáticamente
        config = types.GenerateContentConfig(
            system_instruction="Eres Victoria, una asistente de voz inteligente, muy concisa y conversacional. Tienes acceso a la herramienta 'consultar_servidor_victoria'. Cuando el usuario te pregunte por eventos, reportes o resúmenes, llama a esa función de forma automática para obtener la información. Si el usuario te hace una pregunta general, simplemente conversa. Tus respuestas finales deben ser muy cortas y naturales para leerse en voz alta.",
            tools=[consultar_servidor_victoria],
        )
        chat = gemini_client.chats.create(model="gemini-2.5-flash", config=config)
        
        mensaje = f"Pregunta del usuario: '{prompt_text}'.\n\n(Si necesitas usar tu herramienta para buscar eventos y el usuario no especifica el tiempo, asume {minutes} minutos)."
        response = chat.send_message(mensaje)
        
        result_text = response.text.strip() if response.text else "No logré estructurar una respuesta."
        print(f"🤖  Victoria: {result_text}")
        return result_text
    except Exception as e:
        print(f"❌ Error al consultar a Gemini: {e}")
        return "Hubo un error al comunicarme con el cerebro de Gemini."

def speak_text(text):
    print("🔊  Generando voz y reproduciendo...")
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova", # Voces disponibles: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        tts_filename = "temp_response.mp3"
        response.stream_to_file(tts_filename)
        
        # Reproducir en Mac usando afplay de forma síncrona
        subprocess.run(["afplay", tts_filename])
    except Exception as e:
        print(f"❌ Error en TTS: {e}")

def main():
    parser = argparse.ArgumentParser(description="Victoria Voice Interface")
    parser.add_argument("-d", "--duration", type=int, default=4, help="Duración de la grabación de voz en segundos (default: 4)")
    parser.add_argument("-m", "--minutes", type=int, default=60, help="Minutos de historial de eventos a consultar (default: 60)")
    args = parser.parse_args()

    print("========================================")
    print("🦊 Victoria Voice Interface Activada 🦊")
    print("========================================")
    print(f"⚙️  Configuración: {args.duration}s de escucha, {args.minutes}m de eventos.")
    
    while True:
        try:
            wait_for_wakeword()
            
            # 1. Grabar
            audio_file = record_audio(duration=args.duration)
            
            # 2. Transcribir
            text = transcribe_audio(audio_file)
            if not text.strip():
                print("⚠️  No se escuchó nada o el audio no fue claro. Intenta de nuevo.")
                continue
            
            # 3. Preguntar a Gemini
            respuesta = ask_gemini(text, minutes=args.minutes)
            
            # 4. Hablar
            speak_text(respuesta)
            
        except KeyboardInterrupt:
            print("\nSaliendo de la interfaz de voz...")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()
