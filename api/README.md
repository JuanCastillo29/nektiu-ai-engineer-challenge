# Backend (FastAPI)

Backend de partida del reto. Hoy solo llama al modelo; tú lo conviertes en un asistente RAG (ver los `TODO` en `app.py`).

## Requisitos

- Python 3.10+
- Una API key de OpenAI (o el proveedor que prefieras, adaptando `app.py`).

## Puesta en marcha

```bash
cd api
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...        # Windows: set OPENAI_API_KEY=sk-...
uvicorn app:app --reload --port 8000
```

Comprueba que responde:

```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es NektiBot?"}'
```

## Endpoints

- `GET  /api/health` → `{ "status": "ok" }`
- `POST /api/chat` → body `{ "question": "..." }` → `{ "answer": "...", "sources": [...] }`

Puedes cambiar la estructura, añadir endpoints o reescribir lo que necesites. Lo importante es el resultado y que sepas explicar tus decisiones.
