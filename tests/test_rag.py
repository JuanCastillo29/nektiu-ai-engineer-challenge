"""Tests unitarios del retrieval híbrido (`python -m unittest discover tests`)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

import rag

rag.INDEX = rag.HybridIndex(rag.CHUNKS)  # sin embedder: los tests no tocan la red


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


class FakeEmbedder:
    """Embeddings de mentira: cada texto lleva su vector escrito en el propio test."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self.vectors[t] for t in texts]


def boom(texts):
    raise RuntimeError("502")


PLANES = rag.Chunk("Planes", "El plan Business cuesta 149 euros al mes.")
SOPORTE = rag.Chunk("Soporte", "Atendemos de 9:00 a 18:00, de lunes a viernes.")

# Ejes ortogonales: "precio" y "horario". La pregunta por el horario no comparte ni un
# término con el documento —ese es justo el fallo del léxico— pero apunta al mismo eje.
VECTORS = {
    PLANES.indexable: [1.0, 0.0, 0.0],
    SOPORTE.indexable: [0.0, 1.0, 0.0],
    "¿En qué horario atienden?": [0.0, 1.0, 0.0],
    "¿Cuánto cuesta el plan Business?": [1.0, 0.0, 0.0],
    "¿Cómo se cocina una paella?": [0.6, 0.6, 0.0],
    "tibia": [0.28, 0.28, 0.92],  # coseno 0.28 con ambos: por debajo de MIN_COSINE
}


def hybrid(embed=None) -> rag.HybridIndex:
    return rag.HybridIndex([PLANES, SOPORTE], embed=embed or FakeEmbedder(VECTORS))


class TestEmbeddingIndex(unittest.TestCase):
    def test_normaliza_los_vectores(self):
        index = rag.EmbeddingIndex([PLANES], embed=lambda texts: [[3.0, 4.0]])
        self.assertEqual(index.vectors().tolist(), [[0.6, 0.8]])

    def test_los_vectores_de_los_chunks_se_calculan_una_sola_vez(self):
        embedder = FakeEmbedder(VECTORS)
        index = rag.EmbeddingIndex([PLANES, SOPORTE], embed=embedder)
        index.similarities("¿Cuánto cuesta el plan Business?")
        index.similarities("¿Cuánto cuesta el plan Business?")
        self.assertEqual(embedder.calls, 3)  # 1 de los chunks + 1 por pregunta

    def test_coseno_entre_pregunta_y_chunks(self):
        sims = rag.EmbeddingIndex([PLANES, SOPORTE], embed=FakeEmbedder(VECTORS)).similarities(
            "¿En qué horario atienden?"
        )
        self.assertAlmostEqual(sims[0], 0.0)
        self.assertAlmostEqual(sims[1], 1.0)

    def test_un_fallo_del_proveedor_apaga_el_denso_sin_propagarse(self):
        index = rag.EmbeddingIndex([PLANES], embed=boom)
        self.assertIsNone(index.similarities("lo que sea"))
        self.assertFalse(index.available)


class TestHybridIndex(unittest.TestCase):
    def test_recupera_sin_vocabulario_comun(self):
        """El fallo que arregla el denso: "horario" no aparece en el documento."""
        titles = [s.chunk.title for s in hybrid().search("¿En qué horario atienden?").results]
        self.assertEqual(titles[0], "Soporte")

    def test_el_lexico_sigue_contando(self):
        results = hybrid().search("¿Cuánto cuesta el plan Business?").results
        self.assertEqual(results[0].chunk.title, "Planes")
        self.assertGreater(results[0].lexical, 0)

    def test_score_rrf_golden(self):
        # "Planes" es 1º por las dos vías: 1/(60+1) + 1/(60+1).
        results = hybrid().search("¿Cuánto cuesta el plan Business?").results
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].score, 2 / 61)

    def test_un_candidato_de_una_sola_via_puntua_una_vez(self):
        # Nada de "horario" ni "atienden" está en el documento: solo entra por el denso.
        results = hybrid().search("¿En qué horario atienden?").results
        self.assertAlmostEqual(results[0].score, 1 / 61)
        self.assertEqual(results[0].lexical, 0.0)

    def test_el_suelo_de_coseno_descarta_al_candidato_tibio(self):
        retrieval = hybrid().search("tibia")
        self.assertEqual(retrieval.results, [])
        self.assertFalse(retrieval.grounded)

    def test_sin_denso_es_bm25(self):
        index = rag.HybridIndex([PLANES, SOPORTE])
        self.assertFalse(index.hybrid)
        results = index.search("¿Cómo se cocina una paella?").results
        self.assertEqual(results, [])

    def test_si_el_denso_se_cae_la_busqueda_sigue(self):
        index = hybrid(embed=boom)
        self.assertEqual(index.search("¿En qué horario atienden?").results, [])  # solo BM25
        self.assertGreater(len(index.search("plan Business").results), 0)


class TestRetrieval(unittest.TestCase):
    def test_sources_sin_duplicados_y_en_orden(self):
        a, b = rag.Chunk("A", "x"), rag.Chunk("B", "y")
        retrieval = rag.Retrieval([
            rag.ScoredChunk(a, 3.0), rag.ScoredChunk(b, 2.0), rag.ScoredChunk(a, 1.0),
        ])
        self.assertEqual(retrieval.sources, [a, b])
        self.assertTrue(retrieval.grounded)

    def test_sin_resultados_no_hay_fuentes(self):
        self.assertEqual(rag.Retrieval([]).sources, [])


class TestRetrieveIntegracion(unittest.TestCase):
    """Contrato de extremo a extremo sobre el documento real."""

    def test_documento_por_defecto_indexado(self):
        self.assertGreater(len(rag.CHUNKS), 0)

    def test_precio_cita_la_seccion_de_planes(self):
        sources = rag.retrieve("¿Cuánto cuesta el plan Business?").sources
        self.assertIn("Planes y precios", [c.title for c in sources])

    def test_el_indice_por_defecto_es_lexico_sin_api_key(self):
        # INDEX se reconstruye sin embedder arriba: ni red ni key en los tests.
        self.assertFalse(rag.INDEX.hybrid)
        self.assertEqual(rag.retriever_name(), "léxico (BM25)")

    def test_k_por_defecto(self):
        self.assertEqual(rag.DEFAULT_K, 2)
        self.assertLessEqual(len(rag.retrieve("plan precio soporte datos").results), 2)


if __name__ == "__main__":
    unittest.main()
