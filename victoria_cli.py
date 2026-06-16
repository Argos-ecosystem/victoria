import os
import argparse
import json
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_URL = "http://localhost:8888/analyze/custom"
VICTORIA_URL = os.getenv("VICTORIA_URL", DEFAULT_URL)
VICTORIA_APIKEY = os.getenv("VICTORIA_APIKEY")
VICTORIA_API_MAX_CHARS = os.getenv("VICTORIA_API_MAX_CHARS", "200").strip()
DEFAULT_QUERY_HOURS = int(os.getenv("VICTORIA_DEFAULT_QUERY_HOURS", "12"))


def normalize_victoria_url(url: str) -> str:
    if url.endswith("/analyze/on-demand"):
        return url[: -len("/analyze/on-demand")] + "/analyze/custom"
    return url


def minutes_to_hours(minutes: int) -> int:
    hours = max(1, (max(1, minutes) + 59) // 60)
    return min(hours, 168)


def clamp_hours(hours: int) -> int:
    return min(max(hours, 1), 168)


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


def build_payload(prompt: str, hours: int) -> dict:
    return {"hours": clamp_hours(hours), "prompt": build_api_prompt(prompt)}


def query_omnistatus(prompt: str, hours: int, url: str = VICTORIA_URL):
    payload = build_payload(prompt, hours)
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    target_url = normalize_victoria_url(url)
    response = requests.post(target_url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return target_url, payload, response.json()


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
        "--hours",
        help="Cantidad de horas hacia atras a consultar (1-168, default 12).",
        type=int,
    )
    parser.add_argument(
        "-m", "--minutes",
        help="Compatibilidad: minutos hacia atras; se convierten a horas.",
        type=int,
    )
    parser.add_argument(
        "--url",
        help="URL del endpoint /analyze/custom",
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

    if args.hours is not None:
        hours = clamp_hours(args.hours)
    elif args.minutes is not None:
        hours = minutes_to_hours(args.minutes)
    else:
        hours = clamp_hours(DEFAULT_QUERY_HOURS)

    try:
        url, payload, data = query_omnistatus(prompt, hours, args.url)
        print_api_trace("POST", url, payload)
        print_api_trace("POST", url, payload, data)
    except requests.RequestException as exc:
        print(f"❌ Error al conectar con la API de Omnistatus: {exc}")
        return 1
    except json.JSONDecodeError:
        print("❌ La respuesta no es JSON válido.")
        return 1

    if args.raw:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if "msg" in data or "result" in data:
        print("==============================")
        print("Victoria dijo:")
        print(data.get("msg") or data.get("result"))
        print("==============================")
        if "events_count" in data:
            print(f"Eventos en rango: {data['events_count']}")
        if "status" in data:
            print(f"Status: {data['status']}")
        if "score" in data:
            print(f"Score: {data['score']}")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
