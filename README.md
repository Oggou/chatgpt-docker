# chatgpt-docker (Backend API for LawyerGPT UI)

Minimal FastAPI backend intended to run locally (or in Docker) and serve a simple chat endpoint to `lawyergpt-ui`.

This backend also contains the “stabilization” primitives used during development (e.g., an authority system frame and optional prompt anchoring). The proprietary scoring/stabilization module (formerly “LSTR”) is **not** included in this public workspace.

## What this service provides

- **Chat API**: returns a model response for a given prompt.
- **Stabilization mode**: supports “anchored” vs “raw” generation modes.
- **CORS enabled**: configured for local development UI usage.

## Requirements

- An OpenAI API key available in your environment as **`OPENAI_API_KEY`**
  - The OpenAI Python SDK reads this automatically.
- Python 3.10+ (for local run) or Docker.

## API

### `GET /chat`

Query parameters:

- `q` (string, required): the user prompt
- `mode` (optional): `anchored` (default) or `raw`
- `include_compare` (optional, boolean): when true, also returns the alternate response for side-by-side comparison

Response shape (stable fields):

- `reply`: the selected response (based on `mode`)
- `anchored_response`: present when computed (null otherwise)
- `raw_response`: present when computed (null otherwise)

Example:

```bash
curl "http://localhost:8000/chat?q=Explain%20consideration%20in%20contract%20law&mode=anchored"
```

## Running locally (no Docker)

Install dependencies:

```bash
pip install -r requirements.txt
```

Export your key and run:

```bash
export OPENAI_API_KEY="..."
python app.py
```

The API will listen on `http://localhost:8000`.

## Running with Docker

Build:

```bash
docker build -t chatgpt-docker .
```

Run (publish port 8000 and pass your key):

```bash
docker run --rm -p 8000:8000 -e OPENAI_API_KEY="..." chatgpt-docker
```

## Using with `lawyergpt-ui`

1. Start this backend on `http://localhost:8000`.
2. Start the UI (`npm run dev`) from the `lawyergpt-ui` folder.
3. The UI will call `GET /chat?q=...` and read the `reply` field.

## Security notes

- **Never commit API keys**. Keep secrets in environment variables or a secret manager.
- CORS is currently wide-open for local development. Tighten `allow_origins` before deploying anywhere beyond localhost.

