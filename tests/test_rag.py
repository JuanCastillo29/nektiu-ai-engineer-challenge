"""Tests unitarios del retrieval BM25 (`python -m unittest discover tests`)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

import rag


class TestTokenize(unittest.TestCase):
    def test_normaliza_minusculas_y_tildes(self):
        self.assertEqual(rag.tokenize("¿Cuánta INSTALACIÓN?"), ["instalacion"])

    def test_elimina_stopwords(self):
        self.assertEqual(rag.tokenize("el plan de la empresa"), ["plan", "empresa"])

    def test_une_separador_de_miles(self):
        self.assertIn("2000", rag.tokenize("2.000 consultas"))

    def test_stemming_de_plurales(self):
        self.assertEqual(rag.tokenize("planes asistentes"), rag.tokenize("plan asistente"))

    def test_limites_del_stemmer(self):
        self.assertEqual(rag.tokenize("mes gris meses ares"), ["mes", "gri", "mes", "are"])

    def test_texto_sin_terminos_utiles(self):
        self.assertEqual(rag.tokenize("¿y de lo que...?"), [])


class TestLoadChunks(unittest.TestCase):
    def _write(self, text: str) -> Path:
        path = Path(self.tmp.name) / "doc.md"
        path.write_text(text, encoding="utf-8")
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_una_seccion_por_encabezado_y_descarta_preambulo(self):
        path = self._write("# Título\n\nPreámbulo\n\n## Uno\n\ncuerpo uno\n\n## Dos\n\ncuerpo dos\n")
        chunks = rag.load_chunks(path)
        self.assertEqual([c.title for c in chunks], ["Uno", "Dos"])
        self.assertEqual(chunks[0].text, "cuerpo uno")

    def test_ignora_secciones_vacias(self):
        chunks = rag.load_chunks(self._write("## Vacía\n\n## Llena\n\ntexto\n"))
        self.assertEqual([c.title for c in chunks], ["Llena"])

    def test_indexable_incluye_el_titulo(self):
        chunk = rag.Chunk("Planes y precios", "cuerpo")
        self.assertEqual(chunk.indexable, "Planes y precios\ncuerpo")


class TestBM25Index(unittest.TestCase):
    def setUp(self):
        self.index = rag.BM25Index([
            rag.Chunk("Planes y precios", "El plan Business cuesta 149 euros al mes."),
            rag.Chunk("Integraciones", "Se integra con Slack y el correo electrónico."),
            rag.Chunk("Soporte", "El soporte dedicado responde en 24 horas."),
        ])

    def test_devuelve_el_chunk_relevante_primero(self):
        results = self.index.search("¿Cuánto cuesta el plan Business?").results
        self.assertEqual(results[0].chunk.title, "Planes y precios")
        self.assertGreater(results[0].score, 0)

    def test_respeta_el_limite_k(self):
        self.assertLessEqual(len(self.index.search("el plan y el soporte", k=1).results), 1)

    def test_ordena_por_score_descendente(self):
        scores = [s.score for s in self.index.search("plan soporte Slack", k=3).results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_pregunta_sin_coincidencias_no_esta_fundamentada(self):
        retrieval = self.index.search("¿Cómo se cocina una paella?")
        self.assertEqual(retrieval.results, [])
        self.assertFalse(retrieval.grounded)

    def test_pregunta_solo_de_stopwords(self):
        self.assertFalse(self.index.search("¿y de lo que?").grounded)

    def test_score_cero_para_terminos_ausentes(self):
        self.assertEqual(self.index.score(["paella"], 0), 0.0)


class TestBM25Formula(unittest.TestCase):
    """Valores golden: BM25 Okapi calculado a mano sobre un índice de dos chunks.

    Los títulos son stopwords, así que solo cuenta el cuerpo:
      doc0 = [gato, gato, perro] (len 3), doc1 = [perro] (len 1), avg = 2.
    """

    def setUp(self):
        self.index = rag.BM25Index([rag.Chunk("Uno", "gato gato perro"), rag.Chunk("Dos", "perro")])

    def test_normalizacion_por_longitud(self):
        self.assertEqual(self.index.doc_lengths, [3, 1])
        self.assertAlmostEqual(self.index.length_norms[0], 1.375)
        self.assertAlmostEqual(self.index.length_norms[1], 0.625)

    def test_idf_penaliza_los_terminos_comunes(self):
        self.assertAlmostEqual(self.index.idf["gato"], 0.6931472, places=6)   # df=1
        self.assertAlmostEqual(self.index.idf["perro"], 0.1823216, places=6)  # df=2

    def test_scores_golden(self):
        self.assertAlmostEqual(self.index.score(["gato"], 0), 0.8531042, places=6)
        self.assertAlmostEqual(self.index.score(["perro"], 0), 0.1488339, places=6)
        self.assertAlmostEqual(self.index.score(["perro"], 1), 0.2352536, places=6)

    def test_terminos_de_score_bajo_siguen_contando(self):
        # "perro" puntúa 0.24: por debajo de 1 pero por encima de 0, y es fundamentable.
        self.assertTrue(self.index.search("perro").grounded)


class TestRetrieval(unittest.TestCase):
    def test_sources_sin_duplicados_y_en_orden(self):
        a, b = rag.Chunk("A", "x"), rag.Chunk("B", "y")
        retrieval = rag.Retrieval([
            rag.ScoredChunk(a, 3.0), rag.ScoredChunk(b, 2.0), rag.ScoredChunk(a, 1.0),
        ])
        self.assertEqual(retrieval.sources, ["A", "B"])
        self.assertTrue(retrieval.grounded)

    def test_sin_resultados_no_hay_fuentes(self):
        self.assertEqual(rag.Retrieval([]).sources, [])


class TestRetrieveIntegracion(unittest.TestCase):
    """Contrato de extremo a extremo sobre el documento real."""

    def test_documento_por_defecto_indexado(self):
        self.assertGreater(len(rag.CHUNKS), 0)

    def test_precio_cita_la_seccion_de_planes(self):
        self.assertIn("Planes y precios", rag.retrieve("¿Cuánto cuesta el plan Business?").sources)

    def test_k_por_defecto(self):
        self.assertEqual(rag.DEFAULT_K, 2)
        self.assertLessEqual(len(rag.retrieve("plan precio soporte datos").results), 2)


if __name__ == "__main__":
    unittest.main()
