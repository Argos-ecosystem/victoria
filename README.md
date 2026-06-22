# Victoria

Victoria is the voice and CLI client for Omnistatus event analysis.

Omnistatus owns event storage, filtering, compression, the main analysis model, and the final analysis response. Victoria only captures user input, transcribes voice when needed, calls the Omnistatus API, and speaks or prints the response.

## Requirements

- Python 3.10+
- Omnistatus API reachable through `OMNI_URL`
- OpenAI API key only when using the voice interface

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment

| Variable | Purpose |
| --- | --- |
| `OMNI_URL` | Omnistatus endpoint. Example: `http://host:8001/analyze/custom`. |
| `OMNI_API_MAX_CHARS` | Max response length appended to the API prompt. Default: `200`; use `0` or empty to disable. |
| `OMNI_DEFAULT_QUERY_HOURS` | CLI fallback range when no `--hours` or `--minutes` is provided. Default: `12`. |
| `OMNI_DEFAULT_QUERY_MINUTES` | Voice fallback range when the user does not specify time. Default: `60`. |

### Voice-only environment

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | API key for voice intent routing, OpenAI transcription, and TTS. |
| `OPENAI_VOICE_MODEL` | Lightweight model used only to route voice intent/tool calls before asking Omnistatus. Default: `gpt-4o-mini`. |
| `OPENAI_TRANSCRIBE_MODEL` | OpenAI audio transcription model. Default: `whisper-1`. |
| `OPENAI_TTS_MODEL` | Model used for speech output. Default: `gpt-4o-mini-tts`. |
| `OPENAI_TTS_VOICE` | TTS voice. Default: `coral`. |
| `GEMINI_API_KEY` | Optional, only needed when using `--asr gemini` or `--asr auto` with Gemini enabled. |
| `GEMINI_TRANSCRIBE_MODEL` | Gemini transcription model. Default: `gemini-2.5-flash`. |
| `OMNI_ASR` | Voice transcription provider: `openai`, `gemini`, `google`, `local`, or `auto`. Default: `openai`. |
| `OMNI_GOOGLE_ASR_LANGUAGES` | Comma-separated Google ASR language fallbacks. Default: `es-CL,es-ES,es-419`. |
| `OMNI_WAKEWORD_LISTEN_TIMEOUT_SECONDS` | Wake-word microphone listen timeout before retrying. Default: `1`. |
| `OMNI_WAKEWORD_PHRASE_TIME_LIMIT_SECONDS` | Max seconds captured per wake-word phrase. Default: `3`. |
| `OMNI_WAKEWORD_ERROR_SLEEP_SECONDS` | Pause after transient wake-word recognition errors before listening again. Default: `0.5`. |
| `OMNI_WAKEWORD_SAMPLE_RATE` | Sample rate used by the sounddevice wake-word fallback when PyAudio is unavailable. Default: `16000`. |
| `OMNI_WAKEWORD_MIN_RMS` | Minimum RMS level before trying wake-word transcription with sounddevice. Default: `35`. |
| `OMNI_WAKEWORD_MIN_PEAK` | Minimum peak level before trying wake-word transcription with sounddevice. Default: `350`. |
| `OMNI_LOCAL_ASR_MODEL` | Optional faster-whisper local model name. Default: `tiny`. |
| `OMNI_LOCAL_ASR_TIMEOUT_SECONDS` | Local ASR timeout. Default: `60`. |
| `OMNI_RECORDING_SOUND` | macOS sound used at recording start/end. Default: `/System/Library/Sounds/Ping.aiff`. |
| `OMNI_PROCESSING_MESSAGE` | Spoken waiting message after recording ends. Default: `Consultando el sistema central de datos.` Empty disables it. |
| `OMNI_PROCESSING_VOICE` | macOS `say` voice for the waiting message. Default: `Paulina`. |
| `OMNI_PROCESSING_RATE` | macOS `say` speaking rate for the waiting message. Default: `175`. |
| `OMNI_RECORDING_PROMPT_MESSAGE` | Spoken cue before the recording-start sound. Default: `Habla al escuchar el pip.` Empty disables it. |
| `OMNI_RECORDING_PROMPT_VOICE` | macOS `say` voice for the recording cue. Default: same as `OMNI_PROCESSING_VOICE`. |
| `OMNI_RECORDING_PROMPT_RATE` | macOS `say` speaking rate for the recording cue. Default: `190`. |
| `OMNI_VOICE_UI` | Opens the visual spectrum window for the voice flow. Default: `1`; use `0` to disable. |
| `OMNI_VOICE_UI_WIDTH` | Width of the visual window. Default: `1280`. |
| `OMNI_VOICE_UI_HEIGHT` | Height of the visual window. Default: `720`. |
| `OMNI_FACE_UI` | Shows the integrated webcam panel with local face detection and red boxes. Default: `1`; use `0` to disable. |
| `OMNI_FACE_CAMERA_INDEX` | Webcam index used for face detection. Default: `0`. |
| `OMNI_FACE_UI_SCALE` | Webcam display scale. Default: `0.85`. |
| `OMNI_FACE_PANEL_WIDTH` | Integrated webcam panel width. Default: `420`. |
| `OMNI_FACE_PANEL_HEIGHT` | Integrated webcam panel height. Default: `260`. |
| `OMNI_FACE_DETECT_SCALE_FACTOR` | Haar cascade scan scale factor. Higher can reduce detections. Default: `1.12`. |
| `OMNI_FACE_DETECT_MIN_NEIGHBORS` | Haar cascade confidence-like neighbor threshold. Higher reduces false positives. Default: `10`. |
| `OMNI_FACE_DETECT_MIN_SIZE` | Minimum detected face size in pixels. Higher reduces small false positives. Default: `80`. |
| `OMNI_FACE_MEMORY` | Enables local face memory in ChromaDB. Default: `1`. |
| `OMNI_FACE_MEMORY_DIR` | Persistent ChromaDB directory. Default: `.victoria_face_memory`. |
| `OMNI_FACE_MEMORY_COLLECTION` | ChromaDB collection name. Default: `victoria_faces`. |
| `OMNI_FACE_MEMORY_BACKEND` | Face embedding backend: `opencv` for lightweight local embeddings, or `clip` for SentenceTransformers CLIP. Default: `opencv`. |
| `OMNI_FACE_MEMORY_MODEL` | SentenceTransformers CLIP model used when backend is `clip`. Default: `clip-ViT-B-32`. |
| `OMNI_FACE_MEMORY_SAMPLES` | Number of webcam face samples saved per person. Default: `3`. |
| `OMNI_FACE_MEMORY_THRESHOLD` | Max Chroma cosine distance accepted as a match. Lower is stricter. Default: `0.30`. |
| `OMNI_FACE_MEMORY_COOLDOWN_SECONDS` | Minimum seconds between face recognition checks. Default: `20`. |
| `OMNI_FACE_MEMORY_ASK_ENROLL` | Asks for a name after an unrecognized conversation and stores face samples. Default: `1`. |

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

The voice flow waits for the wake word `Victoria`, records the question, plays a configurable waiting message while processing, infers a time range, uses 12 hours when no range is clear, converts it to `hours`, calls Omnistatus, and speaks the `msg` field from the response.

By default the voice flow opens one large visual window with a central AI face that changes expression by state, side equalizers, a centered `Humano` label, and an integrated webcam panel that marks detected faces with red boxes.

If face memory dependencies are installed, Victoria stores three face crops as local embeddings in ChromaDB after asking for a name. On later sessions, recognized faces are greeted by name near the start of the interaction. Set `OMNI_FACE_MEMORY_BACKEND=clip` to use CLIP embeddings; the first run may need to download a large model.

To skip the wake word and record immediately:

```bash
python3 victoria_voice.py --no-wakeword
```

To run without the visual window:

```bash
python3 victoria_voice.py --no-ui
```

To run without the integrated webcam face panel:

```bash
python3 victoria_voice.py --no-face-ui
```

To try available transcription providers before falling back to OpenAI:

```bash
python3 victoria_voice.py --asr auto
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
