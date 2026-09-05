"""
Observabilidad: logs estructurados en JSON a stdout y trazas ligeras.

Una petición = una traza; cada etapa (retrieval, embeddings, LLM) = un span con su
duración y sus atributos. Sin dependencias nuevas: Render y cualquier agregador de logs
leen stdout, y el formato (`trace_id`, `span_id`, `parent_span_id`) es el de OpenTelemetry
por si un día se exporta de verdad.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

SERVICE = os.getenv("SERVICE_NAME", "nektibot")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 0 = no registrar el texto de las preguntas, solo su longitud.
QUESTION_CHARS = int(os.getenv("LOG_QUESTION_CHARS", "120"))

# Sin UTF-8 (consola de Windows), escapamos: un acento no debe tumbar el log.
ENSURE_ASCII = (sys.stdout.encoding or "").lower().replace("-", "") != "utf8"

TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")

_TRACE_ID: ContextVar[str | None] = ContextVar("trace_id", default=None)
_SPAN_ID: ContextVar[str | None] = ContextVar("span_id", default=None)

LOG = logging.getLogger("nektibot")


class JsonFormatter(logging.Formatter):
    """Una línea de JSON por registro, con la traza en curso si la hay."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "service": SERVICE,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if trace_id := _TRACE_ID.get():
            payload["trace_id"] = trace_id
            payload["span_id"] = _SPAN_ID.get()
        # `fields` va aparte para no chocar con los atributos propios de LogRecord.
        payload.update(getattr(record, "fields", None) or {})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=ENSURE_ASCII, default=str)


def configure() -> None:
    """Deja stdout como única salida de logs, en JSON. Idempotente."""
    root = logging.getLogger()
    if any(getattr(h, "_nektibot", False) for h in root.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._nektibot = True  # marca para no duplicar handlers al reimportar

    root.handlers = [handler]
    root.setLevel(LOG_LEVEL)
    # uvicorn trae sus propios handlers de texto: los vaciamos para que suban al nuestro.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
    # Su línea de acceso la sustituye el middleware, que además lleva la traza.
    logging.getLogger("uvicorn.access").disabled = True


def _hex(length: int) -> str:
    return uuid.uuid4().hex[:length]


def log(msg: str, level: int = logging.INFO, exc_info: bool = False, **fields) -> None:
    LOG.log(level, msg, exc_info=exc_info, extra={"fields": fields})


def current_trace_id() -> str | None:
    return _TRACE_ID.get()


def trace_id_from(traceparent: str | None) -> str:
    """El `trace_id` del W3C traceparent entrante, o uno nuevo si no vale."""
    match = TRACEPARENT_RE.match((traceparent or "").strip())
    return match.group(1) if match else _hex(32)


@contextmanager
def span(name: str, **attrs):
    """Mide una etapa y emite un registro al cerrarla.

    Los atributos que solo se conocen al final se añaden al dict que devuelve:
    `with span("llm.chat") as s: s["tokens"] = ...`
    """
    parent = _SPAN_ID.get()
    token = _SPAN_ID.set(_hex(16))
    started = time.perf_counter()
    fields = dict(attrs)
    try:
        yield fields
    except BaseException as exc:
        fields |= {"status": "error", "error": type(exc).__name__, "error_msg": str(exc)}
        _emit(name, started, parent, fields, logging.ERROR)
        raise
    else:
        fields.setdefault("status", "ok")
        _emit(name, started, parent, fields, logging.INFO)
    finally:
        _SPAN_ID.reset(token)


@contextmanager
def trace(name: str, trace_id: str | None = None, **attrs):
    """Span raíz: abre una traza nueva (o continúa la que llega en el traceparent)."""
    token = _TRACE_ID.set(trace_id or _hex(32))
    parent = _SPAN_ID.set(None)
    try:
        with span(name, **attrs) as fields:
            yield fields
    finally:
        _SPAN_ID.reset(parent)
        _TRACE_ID.reset(token)


def question_fields(question: str) -> dict:
    """La pregunta como atributo, recortada; con `LOG_QUESTION_CHARS=0`, solo su tamaño."""
    fields = {"question_chars": len(question)}
    if QUESTION_CHARS > 0:
        fields["question"] = question[:QUESTION_CHARS]
    return fields


def _emit(name: str, started: float, parent: str | None, fields: dict, level: int) -> None:
    LOG.log(
        level,
        name,
        extra={
            "fields": {
                "span": name,
                "parent_span_id": parent,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                **fields,
            }
        },
    )


configure()
