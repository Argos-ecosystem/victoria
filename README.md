# Victoria Reports

Small set of scripts to pre-calculate short reports from MongoDB events using OpenAI and expose the cached results through a lightweight Flask API.

## Features
- Pulls events from MongoDB and groups similar entries with fast fingerprints.
- Generates summaries with OpenAI models for four time windows: current event, last 3 hours, last 24 hours, and yesterday.
- Caches results in MongoDB to avoid unnecessary recomputation and keeps a history collection.
- Exposes read-only HTTP endpoints (Spanish legacy paths plus English aliases) for voice assistants or other clients.

## Requirements
- Python 3.10+
- MongoDB accessible through `MONGO_URI`
- OpenAI API key with access to the configured models

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Environment Variables
Set these before running either service:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | API key for OpenAI. |
| `GEMINI_API_KEY` | API key for Gemini voice transcription. |
| `PROMPT_ANALYSIS` | System prompt used to steer the event analysis. |
| `MONGO_URI` | Mongo connection URI. |
| `MONGO_DB_NAME` | Database name. |
| `MONGO_COLL_NAME` | Collection containing raw events with a `timestamp` field. |
| `VICTORIA_APIKEY` | Shared secret for the Flask API. |

Optional tuning:
- `MODEL_ACTUAL` (default `gpt-4o-mini`)
- `MODEL_TRES` (default `gpt-4o-mini`)
- `MODEL_DIA` (default `gpt-4o-mini`)
- `MODEL_AYER` (default `gpt-4.1`)
- `OPENAI_MODEL` (default `gpt-4o-mini`, used by `/analyze/on-demand`)
- `OPENAI_VOICE_MODEL` (default `gpt-4o-mini`, used by `victoria_voice.py` for function calling)
- `OPENAI_TRANSCRIBE_MODEL` (default `whisper-1`)
- `OPENAI_TTS_MODEL` (default `gpt-4o-mini-tts`)
- `OPENAI_TTS_VOICE` (default `coral`)
- `OPENAI_TTS_INSTRUCTIONS` (default: Spanish Latin American voice with a soft Chilean accent)
- `OPENAI_TTS_SPEED` (default `1.03`)
- `GEMINI_TRANSCRIBE_MODEL` (default `gemini-2.5-flash`)
- `VICTORIA_ASR` (default `gemini`; options: `gemini`, `google`, `openai`, `local`)
- `VICTORIA_GOOGLE_ASR_LANGUAGES` (default `es-CL,es-ES,es-419`)
- `VICTORIA_LOCAL_ASR_MODEL` (default `tiny`; optional local faster-whisper model)
- `VICTORIA_LOCAL_ASR_TIMEOUT_SECONDS` (default `60`; after this, local ASR falls back)
- `VICTORIA_RECORDING_SOUND` (default `/System/Library/Sounds/Ping.aiff`)
- `CYCLE_SLEEP_SECONDS` (default `600`)
- `REQUEST_TIMEOUT_SECONDS` (default `40`)

## Running the pre-calculator
This loop fetches events, calls OpenAI when the input changes, and stores the results in `victoria_cache` plus a history in `victoria_cache_history`.
```bash
python preCalcultator.py
```

It uses the following cache types (stored under the Mongo field `tipo`): `actual`, `tres`, `dia`, `ayer`.

## Running the API server
Read-only server that returns the cached reports. Default port: `8888`.
```bash
python server.py
```

Endpoints (all expect `apikey` query parameter):
- `/informe_actual` and `/report/current`
- `/informe_tres` and `/report/three-hours`
- `/informe_dia` and `/report/day`
- `/informe_ayer` and `/report/yesterday`

## Running the voice interface
Victoria Voice records a short prompt, plays a system sound before recording, prepares the audio for ASR, transcribes it with Gemini 2.5 Flash by default, and uses `gpt-4o-mini` function calling to query `/analyze/on-demand` when the user asks about events.

Start the Flask API first, then run:
```bash
python3 victoria_voice.py --no-wakeword
```

To use a different ASR:
```bash
python3 victoria_voice.py --no-wakeword --asr openai
```

Gemini 2.5 Flash ASR is the default. Victoria normalizes/trims the audio before transcription and ignores obvious junk transcripts. Google ASR remains available with `--asr google`, and Victoria falls back to Google/OpenAI if Gemini fails. Function calling still runs through `gpt-4o-mini` after the audio is transcribed.

## Notes
- Hashes used for cache comparison are now deterministic (`sha256` over sorted JSON) so the process does not recompute unnecessarily after restarts.
- Mongo field names remain in Spanish (`tipo`, `texto`) for compatibility with existing data; code and logs are now in English.
