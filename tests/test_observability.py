from __future__ import annotations

import json
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from fastapi.testclient import TestClient

import app
import observability as obs
import rag

rag.INDEX = rag.HybridIndex(rag.CHUNKS)  # sin embedder: los tests no tocan la red

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"

PREGUNTA = "¿Cuánto cuesta el plan Business?"


class Capture(logging.Handler):
    """Recoge los registros y los devuelve ya formateados, como saldrían por stdout."""

    def __init__(self):
        super().__init__()
        self.records = []
        self.formatter = obs.JsonFormatter()

    def emit(self, record):
        # Formatear aquí y no al leer: la traza vive en contextvars y ya no estará.
        self.records.append(json.loads(self.format(record)))

    def __enter__(self):
        self.level = obs.LOG.level
        obs.LOG.addHandler(self)
        obs.LOG.setLevel(logging.DEBUG)
        return self.records

    def __exit__(self, *exc):
        obs.LOG.setLevel(self.level)
        obs.LOG.removeHandler(self)


class TestJsonFormatter(unittest.TestCase):
    def test_una_linea_de_json_con_los_campos_base(self):
        with Capture() as records:
            obs.log("hola", answer_chars=10)
        record = records[0]
        self.assertEqual(record["msg"], "hola")
        self.assertEqual(record["level"], "INFO")
        self.assertEqual(record["service"], obs.SERVICE)
        self.assertEqual(record["answer_chars"], 10)  # los campos van al primer nivel
        self.assertIn("ts", record)

    def test_sin_traza_no_hay_ids(self):
        with Capture() as records:
            obs.log("suelto")
        self.assertNotIn("trace_id", records[0])

    def test_un_campo_llamado_como_un_atributo_del_record_no_rompe(self):
        with Capture() as records:
            obs.log("choque", name="planes", module="rag")
        self.assertEqual(records[0]["name"], "planes")
        self.assertEqual(records[0]["logger"], "nektibot")

    def test_la_excepcion_va_en_el_registro(self):
        with Capture() as records:
            try:
                raise ValueError("sin cuota")
            except ValueError:
                obs.log("fallo", level=logging.ERROR, exc_info=True)
        self.assertIn("ValueError: sin cuota", records[0]["exception"])

    def test_serializa_lo_que_json_no_sabe(self):
        with Capture() as records:
            obs.log("raro", path=Path("data/sample.md"))
        self.assertIn("sample.md", records[0]["path"])


class TestTraceparent(unittest.TestCase):
    def test_continua_la_traza_que_llega(self):
        self.assertEqual(obs.trace_id_from(TRACEPARENT), TRACE_ID)

    def test_traceparent_invalido_o_ausente_abre_traza_nueva(self):
        for header in (None, "", "no-soy-un-traceparent", "00-abc-def-01"):
            with self.subTest(header=header):
                trace_id = obs.trace_id_from(header)
                self.assertEqual(len(trace_id), 32)
                self.assertNotEqual(trace_id, TRACE_ID)

    def test_cada_traza_tiene_su_id(self):
        self.assertNotEqual(obs.trace_id_from(None), obs.trace_id_from(None))


class TestSpan(unittest.TestCase):
    def test_un_registro_al_cerrar_con_duracion_y_estado(self):
        with Capture() as records:
            with obs.span("retrieval", k=2):
                pass
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["span"], "retrieval")
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(records[0]["k"], 2)
        self.assertGreaterEqual(records[0]["duration_ms"], 0)

    def test_los_atributos_tardios_se_anaden_dentro(self):
        with Capture() as records:
            with obs.span("llm.chat") as fields:
                fields["total_tokens"] = 140
        self.assertEqual(records[0]["total_tokens"], 140)

    def test_los_spans_anidan_bajo_su_padre(self):
        with Capture() as records, obs.trace("http.request", TRACE_ID):
            with obs.span("chat"):
                with obs.span("retrieval"):
                    pass
        spans = {r["span"]: r for r in records}
        self.assertEqual({r["trace_id"] for r in records}, {TRACE_ID})
        self.assertIsNone(spans["http.request"]["parent_span_id"])
        self.assertEqual(spans["chat"]["parent_span_id"], spans["http.request"]["span_id"])
        self.assertEqual(spans["retrieval"]["parent_span_id"], spans["chat"]["span_id"])

    def test_hermanos_comparten_padre_y_no_id(self):
        with Capture() as records, obs.trace("chat"):
            with obs.span("llm.chat"):
                pass
            with obs.span("llm.chat"):
                pass
        intentos = [r for r in records if r["span"] == "llm.chat"]
        self.assertEqual(len({r["parent_span_id"] for r in intentos}), 1)
        self.assertEqual(len({r["span_id"] for r in intentos}), 2)

    def test_el_error_se_registra_y_se_propaga(self):
        with Capture() as records:
            with self.assertRaises(ValueError):
                with obs.span("llm.chat"):
                    raise ValueError("sin cuota")
        self.assertEqual(records[0]["level"], "ERROR")
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[0]["error"], "ValueError")
        self.assertIn("sin cuota", records[0]["error_msg"])

    def test_la_traza_no_sobrevive_a_su_bloque(self):
        with obs.trace("http.request", TRACE_ID):
            self.assertEqual(obs.current_trace_id(), TRACE_ID)
        self.assertIsNone(obs.current_trace_id())


class TestQuestionFields(unittest.TestCase):
    def test_recorta_la_pregunta(self):
        with patch.object(obs, "QUESTION_CHARS", 5):
            self.assertEqual(
                obs.question_fields("¿Cuánto cuesta?"),
                {"question_chars": 15, "question": "¿Cuán"},
            )

    def test_a_cero_solo_queda_la_longitud(self):
        with patch.object(obs, "QUESTION_CHARS", 0):
            self.assertEqual(obs.question_fields("¿Cuánto cuesta?"), {"question_chars": 15})


class TestConfigure(unittest.TestCase):
    def test_no_duplica_handlers(self):
        before = list(logging.getLogger().handlers)
        obs.configure()
        self.assertEqual(logging.getLogger().handlers, before)


class TestRetrievalInstrumentado(unittest.TestCase):
    @staticmethod
    def span(records, name):
        return next(r for r in records if r.get("span") == name)

    def test_el_span_lleva_lo_recuperado(self):
        with Capture() as records:
            retrieval = rag.retrieve("¿Cuánto cuesta el plan Business?")
        span = self.span(records, "retrieval")
        self.assertEqual(span["k"], rag.DEFAULT_K)
        self.assertIs(span["grounded"], True)
        self.assertEqual(span["results"], len(retrieval.results))
        self.assertEqual(span["titles"], [s.chunk.title for s in retrieval.results])
        self.assertEqual(len(span["scores"]), len(retrieval.results))

    def test_pregunta_sin_candidatos(self):
        with Capture() as records:
            rag.retrieve("paella valenciana socarrat")
        span = self.span(records, "retrieval")
        self.assertIs(span["grounded"], False)
        self.assertEqual(span["titles"], [])

    def test_la_caida_del_denso_deja_aviso(self):
        def boom(texts):
            raise RuntimeError("cuota agotada")

        with Capture() as records:
            rag.HybridIndex(rag.CHUNKS, boom).search("¿Cuánto cuesta?")
        aviso = next(r for r in records if r["msg"] == "dense_retrieval_disabled")
        self.assertEqual(aviso["level"], "WARNING")
        self.assertIn("cuota agotada", aviso["exception"])
        # El span de retrieval cierra bien igual: la búsqueda se sirve con BM25.
        self.assertIs(self.span(records, "retrieval")["hybrid"], False)

    def test_el_indice_denso_en_frio_es_un_span(self):
        index = rag.HybridIndex(rag.CHUNKS, lambda texts: [[1.0, 0.0]] * len(texts))
        with Capture() as records:
            index.search("¿Cuánto cuesta?")
            index.search("¿Y el soporte?")
        indexados = [r for r in records if r.get("span") == "embeddings.index"]
        self.assertEqual(len(indexados), 1)  # los chunks se embeben una sola vez
        self.assertEqual(indexados[0]["chunks"], len(rag.CHUNKS))
        self.assertEqual(len([r for r in records if r.get("span") == "embeddings.query"]), 2)


class TestUsageFields(unittest.TestCase):
    def test_tokens_del_proveedor(self):
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=20, total_tokens=140)
        self.assertEqual(
            app.usage_fields(SimpleNamespace(usage=usage)),
            {"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140},
        )

    def test_sin_usage_no_inventa_campos(self):
        self.assertEqual(app.usage_fields(SimpleNamespace()), {})
        self.assertEqual(app.usage_fields(SimpleNamespace(usage=None)), {})


class TestPeticionTrazada(unittest.TestCase):
    def setUp(self):
        self.http = TestClient(app.app)

    @staticmethod
    def _client(content):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=20, total_tokens=140),
        )
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: completion))
        )

    @staticmethod
    def span(records, name):
        return next(r for r in records if r.get("span") == name)

    def _chat(self, content=json.dumps({"answer": "149 euros.", "used": [1]}), **kwargs):
        """Pregunta que el retrieval real sí fundamenta: la traza lleva todos sus spans."""
        with Capture() as records, patch.object(app, "client", self._client(content)):
            response = self.http.post("/api/chat", json={"question": PREGUNTA}, **kwargs)
        return response, records

    def test_la_traza_del_cliente_vuelve_en_la_cabecera(self):
        response, _ = self._chat(headers={"traceparent": TRACEPARENT})
        self.assertEqual(response.headers["x-trace-id"], TRACE_ID)

    def test_sin_traceparent_el_servidor_abre_la_traza(self):
        response, _ = self._chat()
        self.assertEqual(len(response.headers["x-trace-id"]), 32)

    def test_toda_la_peticion_cuelga_de_una_sola_traza(self):
        _, records = self._chat(headers={"traceparent": TRACEPARENT})
        self.assertEqual({r["trace_id"] for r in records}, {TRACE_ID})
        spans = {r["span"] for r in records if "span" in r}
        self.assertLessEqual({"http.request", "chat", "retrieval", "llm.chat"}, spans)

    def test_el_span_http_lleva_metodo_ruta_y_estado(self):
        _, records = self._chat()
        http = self.span(records, "http.request")
        self.assertEqual(
            (http["method"], http["path"], http["status_code"]), ("POST", "/api/chat", 200)
        )

    def test_el_span_de_chat_resume_el_resultado(self):
        _, records = self._chat()
        chat = self.span(records, "chat")
        self.assertEqual(chat["outcome"], "answered")
        # `used: [1]`: el modelo cita el primer fragmento que le pasó el retrieval.
        self.assertEqual(chat["sources"], [rag.retrieve(PREGUNTA).sources[0].title])
        self.assertEqual(chat["question"], PREGUNTA)

    def test_el_span_del_modelo_lleva_los_tokens(self):
        _, records = self._chat()
        llm = self.span(records, "llm.chat")
        self.assertEqual((llm["model"], llm["attempt"], llm["total_tokens"]), (app.MODEL, 1, 140))

    def test_la_respuesta_ilegible_deja_un_aviso_por_intento(self):
        response, records = self._chat(content="no soy json")
        self.assertEqual(response.status_code, 502)
        avisos = [r for r in records if r["msg"] == "model_response_unparsable"]
        self.assertEqual([r["attempt"] for r in avisos], [1, 2])
        self.assertEqual(self.span(records, "http.request")["status_code"], 502)

    def test_sin_candidatos_el_span_lo_dice_y_no_hay_llamada_al_modelo(self):
        with Capture() as records, patch.object(app, "client", self._client("{}")):
            self.http.post("/api/chat", json={"question": "paella valenciana socarrat"})
        self.assertEqual(self.span(records, "chat")["outcome"], "not_grounded")
        self.assertNotIn("llm.chat", {r.get("span") for r in records})


if __name__ == "__main__":
    unittest.main()
