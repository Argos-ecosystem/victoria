import os
import argparse
import json
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_URL = "http://localhost:8888/analyze/on-demand"
VICTORIA_URL = os.getenv("VICTORIA_URL", DEFAULT_URL)
VICTORIA_APIKEY = os.getenv("VICTORIA_APIKEY")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Victoria CLI - Consulta la API existente de Victoria"
    )
    parser.add_argument(
        "-p", "--prompt",
        help="Texto de consulta para Victoria. Si no se provee, se lee de stdin.",
        type=str,
    )
    parser.add_argument(
        "-m", "--minutes",
        help="Cantidad de minutos en el pasado a consultar (default 60).",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--url",
        help="URL del endpoint /analyze/on-demand",
        default=VICTORIA_URL,
    )
    parser.add_argument(
        "--raw",
        help="Imprime la respuesta JSON completa en lugar de solo el resultado.",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    apikey = VICTORIA_APIKEY or os.getenv("VICTORIA_APIKEY")
    if not apikey:
        print("❌ Error: falta la variable de entorno VICTORIA_APIKEY.")
        print("Define VICTORIA_APIKEY en tu .env o en el entorno antes de ejecutar el CLI.")
        return 1

    prompt = args.prompt
    if not prompt:
        try:
            prompt = input("Pregunta para Victoria: ").strip()
        except KeyboardInterrupt:
            print("\nCancelado.")
            return 1

    if not prompt:
        print("❌ El prompt no puede estar vacío.")
        return 1

    payload = {
        "minutes": args.minutes,
        "prompt": prompt,
    }
    params = {"apikey": apikey}
    headers = {"ngrok-skip-browser-warning": "true"}

    print(f"🌐 Consultando Victoria en {args.url}...")
    print(f"⏱️  Minutos: {args.minutes}")
    print(f"📝 Prompt: {prompt}\n")

    try:
        response = requests.post(args.url, params=params, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"❌ Error al conectar con la API de Victoria: {exc}")
        return 1
    except json.JSONDecodeError:
        print("❌ La respuesta no es JSON válido.")
        return 1

    if args.raw:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if "result" in data:
        print("==============================")
        print("Victoria dijo:")
        print(data["result"])
        print("==============================")
        if "events_count" in data:
            print(f"Eventos en rango: {data['events_count']}")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
