from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from fastapi.testclient import TestClient
from openai import OpenAIError

import app
import rag


def completion(content):
    """Réplica mínima de lo que devuelve `chat.completions.create`."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def answer(text: str, used: list[int]):
    return completion(json.dumps({"answer": text, "used": used}))


PLANES = rag.ScoredChunk(rag.Chunk("Planes y precios", "El plan Business cuesta 149 euros."), 2.0)
SOPORTE = rag.ScoredChunk(rag.Chunk("Soporte", "Responde en 24 horas."), 1.0)


class FakeCompletions:
    """Devuelve las respuestas en orden y registra los kwargs de cada llamada."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


def fake_client(*responses):
    completions = FakeCompletions(*responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


class TestBuildContext(unittest.TestCase):
    def test_numera_desde_uno_con_titulo_y_cuerpo(self):
        self.assertEqual(
            app.build_context([PLANES, SOPORTE]),
            "[1] Planes y precios\nEl plan Business cuesta 149 euros.\n\n"
            "[2] Soporte\nResponde en 24 horas.",
        )

    def test_sin_resultados_contexto_vacio(self):
        self.assertEqual(app.build_context([]), "")


class TestCite(unittest.TestCase):
    def test_traduce_indices_a_titulos(self):
        self.assertEqual(app.cite([2, 1], [PLANES, SOPORTE]), ["Soporte", "Planes y precios"])

    def test_descarta_indices_fuera_de_rango(self):
        self.assertEqual(app.cite([0, 3, -1, 1], [PLANES, SOPORTE]), ["Planes y precios"])

    def test_elimina_titulos_duplicados(self):
        self.assertEqual(app.cite([1, 1], [PLANES, SOPORTE]), ["Planes y precios"])

    def test_sin_citas_lista_vacia(self):
        self.assertEqual(app.cite([], [PLANES]), [])


class TestGenerate(unittest.TestCase):
    def _generate(self, *responses, question="¿Cuánto cuesta?", results=(PLANES, SOPORTE)):
        client, completions = fake_client(*responses)
        with patch.object(app, "client", client):
            return app.generate(question, list(results)), completions

    def test_respuesta_valida_con_sus_fuentes(self):
        response, completions = self._generate(answer("149 euros.", [1]))
        self.assertEqual(response.answer, "149 euros.")
        self.assertEqual(response.sources, ["Planes y precios"])
        self.assertEqual(len(completions.calls), 1)

    def test_reintenta_una_vez_si_el_json_es_ilegible(self):
        response, completions = self._generate(completion("no soy json"), answer("ok", [2]))
        self.assertEqual(response.answer, "ok")
        self.assertEqual(response.sources, ["Soporte"])
        self.assertEqual(len(completions.calls), 2)

    def test_agota_los_intentos_y_devuelve_502(self):
        with self.assertRaises(app.HTTPException) as ctx:
            self._generate(completion("{"))
        self.assertEqual(ctx.exception.status_code, 502)

    def test_se_rinde_al_segundo_intento_sin_probar_un_tercero(self):
        # La tercera respuesta es válida: si se llegase a pedir, el test no vería el 502.
        client, completions = fake_client(completion("{"), completion("{"), answer("tarde", []))
        with patch.object(app, "client", client), self.assertRaises(app.HTTPException):
            app.generate("¿Y?", [PLANES])
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(app.PARSE_ATTEMPTS, 2)

    def test_json_sin_las_claves_esperadas_es_ilegible(self):
        with self.assertRaises(app.HTTPException):
            self._generate(completion('{"answer": "hola"}'))

    def test_content_nulo_es_ilegible(self):
        with self.assertRaises(app.HTTPException):
            self._generate(completion(None))

    def test_prompt_lleva_sistema_contexto_y_pregunta(self):
        _, completions = self._generate(answer("149 euros.", [1]))
        messages = completions.calls[0]["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": app.SYSTEM_PROMPT})
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("[1] Planes y precios", messages[1]["content"])
        self.assertIn("Pregunta: ¿Cuánto cuesta?", messages[1]["content"])

    def test_llamada_con_temperatura_y_schema(self):
        _, completions = self._generate(answer("149 euros.", [1]))
        call = completions.calls[0]
        self.assertEqual(call["model"], app.MODEL)
        self.assertEqual(call["temperature"], app.TEMPERATURE)
        self.assertEqual(call["response_format"]["type"], "json_schema")
        self.assertEqual(call["response_format"]["json_schema"], app.ANSWER_SCHEMA)

    def test_los_reintentos_repiten_el_mismo_prompt(self):
        _, completions = self._generate(completion("nope"), answer("ok", []))
        self.assertEqual(completions.calls[0]["messages"], completions.calls[1]["messages"])

    def test_propaga_el_error_del_proveedor(self):
        with self.assertRaises(OpenAIError):
            self._generate(OpenAIError("sin cuota"))


class TestConstantes(unittest.TestCase):
    def test_el_fallback_es_el_texto_que_impone_el_system_prompt(self):
        self.assertEqual(app.NO_SE, "No lo sé.")
        self.assertIn(f'"{app.NO_SE}"', app.SYSTEM_PROMPT)


class TestAnswerSchema(unittest.TestCase):
    def test_schema_estricto_y_cerrado(self):
        schema = app.ANSWER_SCHEMA
        # `is` y no `assertTrue/False`: con el flag a None el proveedor no valida el schema.
        self.assertIs(schema["strict"], True)
        self.assertIs(schema["schema"]["additionalProperties"], False)
        self.assertEqual(sorted(schema["schema"]["required"]), ["answer", "used"])
        self.assertEqual(schema["schema"]["properties"]["used"]["items"]["type"], "integer")


class TestHealth(unittest.TestCase):
    def test_health_ok(self):
        response = TestClient(app.app).get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class TestChat(unittest.TestCase):
    def setUp(self):
        self.http = TestClient(app.app)

    def _post(self, payload):
        return self.http.post("/api/chat", json=payload)

    def test_respuesta_fundamentada_con_fuentes(self):
        client, completions = fake_client(answer("149 euros.", [1]))
        retrieval = rag.Retrieval([PLANES, SOPORTE])
        with patch.object(app, "client", client), patch.object(rag, "retrieve", return_value=retrieval):
            response = self._post({"question": "¿Cuánto cuesta el plan Business?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": "149 euros.", "sources": ["Planes y precios"]})
        self.assertEqual(len(completions.calls), 1)

    def test_sin_candidatos_responde_no_lo_se_sin_llamar_al_modelo(self):
        client, completions = fake_client(answer("no debería llamarse", [1]))
        with patch.object(app, "client", client), patch.object(rag, "retrieve", return_value=rag.Retrieval([])):
            response = self._post({"question": "¿Cómo se cocina una paella?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": app.NO_SE, "sources": []})
        self.assertEqual(completions.calls, [])

    def test_la_pregunta_llega_recortada_al_retrieval(self):
        client, _ = fake_client(answer("149 euros.", [1]))
        with patch.object(app, "client", client), patch.object(rag, "retrieve") as retrieve:
            retrieve.return_value = rag.Retrieval([PLANES])
            self._post({"question": "  ¿Cuánto cuesta?\n"})
        self.assertEqual(retrieve.call_args[0][0], "¿Cuánto cuesta?")

    def test_pregunta_solo_espacios_es_422(self):
        self.assertEqual(self._post({"question": "   \n"}).status_code, 422)

    def test_pregunta_ausente_es_422(self):
        self.assertEqual(self._post({}).status_code, 422)

    def test_error_del_proveedor_es_502(self):
        client, _ = fake_client(OpenAIError("sin cuota"))
        with patch.object(app, "client", client), patch.object(rag, "retrieve", return_value=rag.Retrieval([PLANES])):
            response = self._post({"question": "¿Cuánto cuesta?"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("sin cuota", response.json()["detail"])

    def test_respuesta_ilegible_es_502(self):
        client, _ = fake_client(completion("no soy json"))
        with patch.object(app, "client", client), patch.object(rag, "retrieve", return_value=rag.Retrieval([PLANES])):
            response = self._post({"question": "¿Cuánto cuesta?"})
        self.assertEqual(response.status_code, 502)

    def test_cors_abierto(self):
        response = self.http.options(
            "/api/chat",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
        self.assertEqual(response.headers["access-control-allow-origin"], "*")


class TestChatIntegracion(unittest.TestCase):
    """Endpoint contra el retrieval real: solo se mockea el proveedor."""

    def test_pregunta_del_documento_cita_su_seccion(self):
        question = "¿Cuánto cuesta el plan Business?"
        expected = rag.retrieve(question).sources
        # El modelo cita todos los fragmentos recuperados; el orden lo fija el retrieval.
        client, completions = fake_client(answer("149 euros al mes.", list(range(1, len(expected) + 1))))
        with patch.object(app, "client", client):
            response = TestClient(app.app).post("/api/chat", json={"question": question})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"], expected)
        self.assertIn("Planes y precios", expected)
        self.assertIn("Planes y precios", completions.calls[0]["messages"][1]["content"])

    def test_pregunta_ajena_al_documento_no_llama_al_modelo(self):
        client, completions = fake_client(answer("no debería llamarse", [1]))
        with patch.object(app, "client", client):
            response = TestClient(app.app).post(
                "/api/chat", json={"question": "paella valenciana socarrat"}
            )
        self.assertEqual(response.json(), {"answer": app.NO_SE, "sources": []})
        self.assertEqual(completions.calls, [])


if __name__ == "__main__":
    unittest.main()
