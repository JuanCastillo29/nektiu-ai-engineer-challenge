"""
Nektiu AI Engineer Challenge — backend de partida.

Este backend HOY solo reenvía la pregunta al modelo (sin RAG).
Tu trabajo es convertirlo en un asistente FUNDAMENTADO en el documento
de `data/sample.md`. Busca los comentarios `TODO` más abajo.

Ejecutar en local:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...        # en Windows: set OPENAI_API_KEY=...
    uvicorn app:app --reload --port 8000
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Nektiu AI Engineer Challenge API")

# CORS abierto para que tu frontend (local o desplegado) pueda llamar al backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

DATA_PATH = Path(__file__).parent.parent / "data" / "sample.md"


def load_document() -> str:
    """Carga el documento sobre el que responder. Puedes cambiarlo por el tuyo."""
    return DATA_PATH.read_text(encoding="utf-8")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # --------------------------------------------------------------------- #
    # TODO (parte 1 — RAG): en vez de mandar la pregunta "a pelo", recupera
    #   los fragmentos relevantes de load_document() (chunking + búsqueda por
    #   similitud con embeddings, o la técnica que prefieras) y pásalos como
    #   contexto al modelo.
    #
    # TODO (parte 2 — honestidad): si el contexto recuperado no contiene la
    #   respuesta, la app debe responder "No lo sé" en lugar de inventar.
    #
    # TODO (parte 3 — citas): devuelve en `sources` los fragmentos que has
    #   usado para construir la respuesta.
    # --------------------------------------------------------------------- #
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Eres un asistente conciso y honesto."},
            {"role": "user", "content": req.question},
        ],
    )
    return ChatResponse(answer=completion.choices[0].message.content, sources=[])
