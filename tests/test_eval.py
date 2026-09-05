"""Tests del runner de evaluación: la aritmética de las métricas, sin red."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

import rag
import run


def case(id_: int, expected: str | None, category: str = "directa") -> run.Case:
    return run.Case(id=id_, category=category, question=f"pregunta {id_}", expected_source=expected)


def retrieved(id_: int, expected: str | None, titles: list[str], **kwargs) -> run.RetrievalOutcome:
    results = [rag.ScoredChunk(rag.Chunk(t, ""), 1.0) for t in titles]
    return run.RetrievalOutcome(case(id_, expected, **kwargs), results)


def generated(
    id_: int,
    expected: str | None,
    cited: list[str] = [],
    *,
    abstiene: bool = False,
    fundamentada: bool = True,
    correcta: bool = True,
) -> run.GenerationOutcome:
    judge = {
        "abstiene": abstiene,
        "fundamentada": fundamentada,
        "correcta": correcta,
        "motivo": "",
    }
    return run.GenerationOutcome(case(id_, expected), "respuesta", cited, judge)


class TestLoadCases(unittest.TestCase):
    def test_lee_el_dataset_del_repo(self):
        cases = run.load_cases()
        self.assertEqual(len(cases), 50)
        self.assertEqual(cases[0].id, 1)

    def test_probes_es_opcional(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.json"
            path.write_text(
                json.dumps(
                    {"cases": [{"id": 1, "category": "directa", "question": "q",
                                "expected_source": None}]}
                ),
                encoding="utf-8",
            )
            self.assertIsNone(run.load_cases(path)[0].probes)


class TestRetrievalOutcome(unittest.TestCase):
    def test_rank_es_la_posicion_uno_based(self):
        self.assertEqual(retrieved(1, "B", ["A", "B"]).rank, 2)

    def test_rank_cero_si_no_esta(self):
        outcome = retrieved(1, "C", ["A", "B"])
        self.assertEqual(outcome.rank, 0)
        self.assertFalse(outcome.hit)

    def test_una_no_respondible_nunca_es_acierto(self):
        # `expected_source` None no puede estar en `titles`, pero el flag manda.
        self.assertFalse(retrieved(1, None, ["A"]).hit)

    def test_empty_solo_sin_candidatos(self):
        self.assertTrue(retrieved(1, "A", []).empty)
        self.assertFalse(retrieved(1, "A", ["A"]).empty)


class TestRetrievalMetrics(unittest.TestCase):
    def test_recall_y_mrr_sobre_las_respondibles(self):
        m = run.retrieval_metrics([
            retrieved(1, "A", ["A", "B"]),      # rank 1
            retrieved(2, "B", ["A", "B"]),      # rank 2
            retrieved(3, "C", ["A", "B"]),      # fuera de k
            retrieved(4, "D", []),              # sin candidatos
            retrieved(5, None, []),             # no respondible
        ])
        self.assertEqual(m["respondibles"], 4)
        self.assertEqual(m["recall_at_k"], 0.5)
        self.assertEqual(m["mrr_at_k"], (1 + 0.5) / 4)
        self.assertEqual(m["sin_candidatos_respondibles"], 0.25)
        self.assertEqual(m["recuperado_pero_fuera_de_k"], 0.25)

    def test_silencio_se_mide_solo_en_las_no_respondibles(self):
        m = run.retrieval_metrics([
            retrieved(1, None, []),
            retrieved(2, None, ["A"]),
            retrieved(3, "A", []),
        ])
        self.assertEqual(m["no_respondibles"], 2)
        self.assertEqual(m["silencio_no_respondibles"], 0.5)

    def test_sin_respondibles_las_ratios_son_none(self):
        m = run.retrieval_metrics([retrieved(1, None, [])])
        self.assertIsNone(m["recall_at_k"])
        self.assertIsNone(m["mrr_at_k"])

    def test_por_categoria_ignora_las_no_respondibles(self):
        tally = run.by_category([
            retrieved(1, "A", ["A"], category="directa"),
            retrieved(2, "B", ["A"], category="directa"),
            retrieved(3, None, [], category="fuera_alcance"),
        ])
        self.assertEqual(tally, {"directa": (1, 2)})


class TestGenerationMetrics(unittest.TestCase):
    def test_abstencion_por_mitades_del_dataset(self):
        m = run.generation_metrics([
            generated(1, "A", ["A"]),                    # responde y toca
            generated(2, "B", abstiene=True),            # abstención falsa
            generated(3, None, abstiene=True),           # abstención correcta
            generated(4, None, ["A"]),                   # responde sin cobertura
        ])
        self.assertEqual(m["abstencion_falsa"], 0.5)
        self.assertEqual(m["abstencion_correcta"], 0.5)
        self.assertEqual(m["respuesta_sin_cobertura"], 0.5)

    def test_la_cita_se_mide_solo_donde_hubo_respuesta(self):
        m = run.generation_metrics([
            generated(1, "A", ["A"]),
            generated(2, "B", ["A"]),
            generated(3, "C", abstiene=True),            # no cuenta: se abstuvo
            generated(4, None, ["A"]),                   # no cuenta: no respondible
        ])
        self.assertEqual(m["cita_la_seccion_esperada"], 0.5)

    def test_citar_al_abstenerse_es_un_fallo(self):
        m = run.generation_metrics([
            generated(1, None, ["A"], abstiene=True),
            generated(2, None, [], abstiene=True),
        ])
        self.assertEqual(m["cita_al_abstenerse"], 0.5)

    def test_juez_sobre_todos_los_casos(self):
        m = run.generation_metrics([
            generated(1, "A", ["A"]),
            generated(2, "B", ["B"], fundamentada=False, correcta=False),
        ])
        self.assertEqual(m["fundamentada"], 0.5)
        self.assertEqual(m["correcta"], 0.5)

    def test_sin_abstenciones_la_ratio_es_none(self):
        self.assertIsNone(run.generation_metrics([generated(1, "A", ["A"])])["cita_al_abstenerse"])


class TestJudgeAnswer(unittest.TestCase):
    def setUp(self):
        self.verdict = {"abstiene": False, "fundamentada": True, "correcta": True, "motivo": "ok"}
        self.calls = []

        def create(**kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.verdict)))]
            )

        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def test_devuelve_el_veredicto_parseado(self):
        verdict = run.judge_answer(self.client, "m", "doc", case(1, "A"), "respuesta", ["A"])
        self.assertEqual(verdict, self.verdict)
        self.assertEqual(self.calls[0]["response_format"]["json_schema"], run.JUDGE_SCHEMA)

    def test_el_juez_no_ve_la_seccion_esperada(self):
        run.judge_answer(self.client, "m", "doc", case(1, "Planes y precios"), "respuesta", [])
        prompt = self.calls[0]["messages"][1]["content"]
        self.assertNotIn("Planes y precios", prompt)
        self.assertIn("ninguno", prompt)


if __name__ == "__main__":
    unittest.main()
