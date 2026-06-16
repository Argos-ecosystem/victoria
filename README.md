# Victoria

Victoria is the voice and CLI client for Omnistatus event analysis.

Omnistatus owns the event storage, filtering, compression, model call, and final analysis response. Victoria only transcribes user intent, calls the Omnistatus API, and speaks or prints the response.

## Requirements

- Python 3.10+
- OpenAI API key for voice intent, transcription, and TTS
- Omnistatus API reachable through `VICTORIA_URL`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | API key for OpenAI voice features. |
| `OPENAI_VOICE_MODEL` | Model used for voice intent and tool calling. Default: `gpt-4o-mini`. |
| `OPENAI_TRANSCRIBE_MODEL` | Model used for audio transcription. Default: `whisper-1`. |
| `OPENAI_TTS_MODEL` | Model used for speech output. Default: `gpt-4o-mini-tts`. |
| `OPENAI_TTS_VOICE` | TTS voice. Default: `coral`. |
| `VICTORIA_URL` | Omnistatus endpoint. Example: `http://host:8001/analyze/custom`. |
| `VICTORIA_APIKEY` | Optional legacy API key. Not sent to `/analyze/custom`. |
| `VICTORIA_API_MAX_CHARS` | Max response length appended to the API prompt. Default: `200`; use `0` or empty to disable. |
| `VICTORIA_DEFAULT_QUERY_MINUTES` | Voice fallback range when the user does not specify time. Default: `720` (12 hours). |
| `VICTORIA_DEFAULT_QUERY_HOURS` | CLI fallback range when no `--hours` or `--minutes` is provided. Default: `12`. |

## Omnistatus API

Victoria calls:

```http
POST /analyze/custom
Content-Type: application/json
```

```json
{
  "hours": 6,
  "prompt": "Dime si hubo errores criticos en este periodo"
}
```

Parameters:

| Param | Type | Description |
| --- | --- | --- |
| `hours` | int, 1-168 | Hours backwards to search events in Mongo. |
| `prompt` | string | Custom analysis prompt. |

Expected response fields:

```json
{
  "status": "ok",
  "score": 0,
  "msg": "Resumen breve del analisis",
  "events_count": 0,
  "window_hours": 6
}
```

## CLI

```bash
python3 victoria_cli.py --hours 6 --prompt "Dime si hubo errores criticos en este periodo"
```

`--minutes` is still accepted for old commands and is converted to hours. If no range is provided, the CLI asks for the last 12 hours.

## Voice

```bash
python3 victoria_voice.py
```

The voice flow waits for the wake word `Victoria`, records the question, infers a time range, uses 12 hours when no range is clear, converts it to `hours`, calls Omnistatus, and speaks the `msg` field from the response.

To skip the wake word and record immediately:

```bash
python3 victoria_voice.py --no-wakeword
```

To start recording with a headset media button instead of Enter:

```bash
python3 victoria_voice.py --no-wakeword --trigger media
```

To check whether macOS exposes the headset button to Victoria:

```bash
python3 victoria_voice.py --debug-keys
```

Press the headset button once. If the output says `TRIGGER`, `--trigger media` should work. On macOS, this may require Accessibility permission for the terminal app. If the headset button is not exposed as a media key, Victoria falls back to Enter.
