"""
Retrieval del asistente: trocea el documento y busca los fragmentos relevantes.

Híbrido: BM25 léxico + embeddings densos, fusionados por RRF. El léxico acierta el
término exacto; el denso cubre la pregunta que no comparte vocabulario con el documento.
Sin API key el denso se apaga y queda solo BM25.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import observability as obs

load_dotenv(Path(__file__).parent / ".env")  # la key decide si hay retrieval denso

DATA_PATH = Path(__file__).parent.parent / "data" / "sample.md"

K1 = 1.5
B = 0.75

DEFAULT_K = 2

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

MIN_COSINE = 0.30
RRF_K = 60

_ES_PLURAL_ENDINGS = "lnrdzjsxy"

_STOPWORDS = {
    "a", "al", "algo", "alguna", "algunas", "alguno", "algunos", "ante", "aqui", "asi",
    "aunque", "cada", "como", "con", "contra", "cual", "cuales", "cuando", "cuanta",
    "cuantas", "cuanto", "cuantos", "de", "del", "desde", "donde", "dos", "e", "el",
    "ella", "ellas", "ello", "ellos", "en", "entre", "era", "eran", "eres", "es", "esa",
    "esas", "ese", "eso", "esos", "esta", "estan", "estar", "estas", "este", "esto",
    "estos", "ha", "hace", "hacen", "hacer", "hasta", "hay", "la", "las", "le", "les",
    "lo", "los", "mas", "me", "mi", "mis", "mucho", "muy", "ni", "no", "nos", "o", "otra",
    "otras", "otro", "otros", "para", "pero", "poco", "por", "porque", "puede", "pueden",
    "que", "quien", "quienes", "se", "segun", "ser", "si", "sin", "sobre", "solo", "son",
    "su", "sus", "tambien", "tan", "te", "tener", "tiene", "tienen", "todo", "todos",
    "tu", "un", "una", "unas", "uno", "unos", "y", "ya",
}


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("es") and token[-3] in _ES_PLURAL_ENDINGS:
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Texto -> términos indexables (minúsculas, sin tildes, sin stopwords, sin plural)."""
    text = _strip_accents(text.lower())
    text = re.sub(r"(?<=\d)[.,](?=\d)", "", text)  # "2.000" -> "2000"
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    """Un fragmento citable del documento: una sección `##` del Markdown."""

    title: str
    text: str

    @property
    def indexable(self) -> str:
        """Se indexa con el encabezado; se cita solo con el título."""
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    lexical: float = 0.0
    dense: float | None = None


@dataclass(frozen=True)
class Retrieval:
    results: list[ScoredChunk]

    @property
    def grounded(self) -> bool:
        """Sin candidatos no hay nada sobre lo que fundamentar una respuesta."""
        return bool(self.results)

    @property
    def sources(self) -> list[Chunk]:
        """Fragmentos citables, sin duplicados y en orden de relevancia."""
        return list(dict.fromkeys(s.chunk for s in self.results))


def load_chunks(path: Path = DATA_PATH) -> list[Chunk]:
    """
    Una sección `##` = un chunk; el preámbulo anterior al primer `##` es metadato y se cae.
    """
    chunks = []
    for section in re.split(r"(?m)^## ", path.read_text(encoding="utf-8"))[1:]:
        title, _, body = section.partition("\n")
        if body.strip():
            chunks.append(Chunk(title.strip(), body.strip()))
    return chunks


class BM25Index:
    """Índice léxico BM25 (Okapi) en memoria, dueño de la búsqueda sobre sus chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.doc_tokens = [Counter(tokenize(c.indexable)) for c in chunks]
        self.doc_lengths = [sum(tf.values()) for tf in self.doc_tokens]

        avg_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.length_norms = [1 - B + B * (n / avg_length) for n in self.doc_lengths]

        doc_freq: Counter = Counter()
        for tf in self.doc_tokens:
            doc_freq.update(tf.keys())
        self.idf = {
            term: math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score(self, query_tokens: list[str], doc_id: int) -> float:
        tf_doc = self.doc_tokens[doc_id]
        length_norm = self.length_norms[doc_id]

        total = 0.0
        for term in query_tokens:
            tf = tf_doc.get(term, 0)
            if tf == 0:
                continue
            total += self.idf[term] * (tf * (K1 + 1)) / (tf + K1 * length_norm)
        return total

    def scores(self, question: str) -> list[float]:
        query_tokens = tokenize(question)
        if not query_tokens:
            return [0.0] * len(self.chunks)
        return [self.score(query_tokens, i) for i in range(len(self.chunks))]

    def search(self, question: str, k: int = DEFAULT_K) -> Retrieval:
        scored = [
            ScoredChunk(chunk, score, lexical=score)
            for chunk, score in zip(self.chunks, self.scores(question))
            if score > 0
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return Retrieval(scored[:k])


_CLIENT = None


def _client():
    """Cliente único y perezoso: sin denso no se importa el SDK ni se abre conexión."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        _CLIENT = OpenAI(
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "30")),
        )
    return _CLIENT


def _openai_embed(texts: list[str]) -> list[list[float]]:
    response = _client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _unit_rows(vectors: list[list[float]]) -> np.ndarray:
    """Vectores normalizados por filas: con norma 1, el coseno es el producto escalar."""
    matrix = np.asarray(vectors, dtype=float)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


class EmbeddingIndex:
    """Índice denso: coseno entre la pregunta y cada chunk, con los vectores normalizados."""

    def __init__(self, chunks: list[Chunk], embed=_openai_embed) -> None:
        self.chunks = chunks
        self.available = True
        self._embed = embed
        self._vectors: np.ndarray | None = None
        self._lock = threading.Lock()

    def vectors(self) -> np.ndarray:
        """Los vectores de los chunks, calculados una sola vez (una llamada)."""
        with self._lock:  # sin él, N peticiones en frío embeben los chunks N veces
            if self._vectors is None:
                # Span propio: es el arranque en frío que paga la primera pregunta.
                with obs.span("embeddings.index", chunks=len(self.chunks), model=EMBEDDING_MODEL):
                    self._vectors = _unit_rows(self._embed([c.indexable for c in self.chunks]))
        return self._vectors

    def similarities(self, question: str) -> list[float] | None:
        """Coseno con cada chunk, o None si el denso no está disponible."""
        if not self.available:
            return None
        try:
            with obs.span("embeddings.query", model=EMBEDDING_MODEL):
                return list(self.vectors() @ _unit_rows(self._embed([question]))[0])
        except Exception:  # red, cuota, key inválida: el híbrido sigue con BM25
            self.available = False
            obs.log(
                "dense_retrieval_disabled",
                level=logging.WARNING,
                model=EMBEDDING_MODEL,
                exc_info=True,
            )
            return None


def _ranking(scores: list[float], floor: float) -> dict[int, int]:
    """Índices por encima del suelo, ordenados por score -> puesto (1-based)."""
    candidates = sorted(
        (i for i, s in enumerate(scores) if s > floor), key=lambda i: -scores[i]
    )
    return {i: rank for rank, i in enumerate(candidates, start=1)}


class HybridIndex:
    """BM25 + embeddings fusionados por RRF; sin `embed` se queda en solo léxico.

    Candidato es lo que pasa el filtro de *alguno* de los dos: score léxico > 0 o coseno
    por encima de `MIN_COSINE`.
    """

    def __init__(self, chunks: list[Chunk], embed=None) -> None:
        self.lexical = BM25Index(chunks)
        self.dense = EmbeddingIndex(chunks, embed) if embed else None

    @property
    def hybrid(self) -> bool:
        return self.dense is not None and self.dense.available

    def search(self, question: str, k: int = DEFAULT_K) -> Retrieval:
        with obs.span("retrieval", k=k, hybrid=self.hybrid) as fields:
            retrieval = self._search(question, k)
            fields |= {
                "hybrid": self.hybrid,  # el denso puede haberse caído durante la búsqueda
                "results": len(retrieval.results),
                "grounded": retrieval.grounded,
                "titles": [s.chunk.title for s in retrieval.results],
                "scores": [
                    {"rrf": round(s.score, 4), "bm25": round(s.lexical, 3),
                     "cos": None if s.dense is None else round(s.dense, 3)}
                    for s in retrieval.results
                ],
            }
            return retrieval

    def _search(self, question: str, k: int) -> Retrieval:
        similarities = self.dense.similarities(question) if self.dense else None
        lexical = self.lexical.scores(question)

        rankings = [_ranking(lexical, 0.0)]
        if similarities is not None:
            rankings.append(_ranking(similarities, MIN_COSINE))

        scored = [
            ScoredChunk(
                self.lexical.chunks[i],
                sum(1 / (RRF_K + r[i]) for r in rankings if i in r),
                lexical=lexical[i],
                dense=None if similarities is None else similarities[i],
            )
            for i in set().union(*rankings)
        ]
        # Empate en RRF (mismo puesto por cada vía): decide el coseno, que es el único de
        # los dos scores comparable entre preguntas.
        scored.sort(key=lambda s: (s.score, s.dense or 0.0), reverse=True)
        return Retrieval(scored[:k])


CHUNKS = load_chunks()
# Sin API key no hay denso: los tests y `--lexical` corren sin tocar la red.
INDEX = HybridIndex(CHUNKS, _openai_embed if os.getenv("OPENAI_API_KEY") else None)

obs.log("index_ready", chunks=len(CHUNKS), hybrid=INDEX.hybrid, embedding_model=EMBEDDING_MODEL)


def retriever_name() -> str:
    return f"híbrido (BM25 + {EMBEDDING_MODEL}, RRF)" if INDEX.hybrid else "léxico (BM25)"


def retrieve(question: str, k: int = DEFAULT_K) -> Retrieval:
    """Búsqueda sobre el índice por defecto."""
    return INDEX.search(question, k)


if __name__ == "__main__":
    import sys

    if "LOG_LEVEL" not in os.environ:
        logging.getLogger().setLevel(logging.WARNING)

    question = " ".join(sys.argv[1:]) or "¿Cuánto cuesta el plan Business?"
    r = retrieve(question)
    print(f"Pregunta: {question}   [{retriever_name()}]")
    print(f"Tokens:   {tokenize(question)}")
    for s in r.results:
        dense = "    n/a" if s.dense is None else f"{s.dense:6.3f}"
        print(f"  rrf {s.score:.4f}  bm25 {s.lexical:6.3f}  cos {dense}  {s.chunk.title}")
