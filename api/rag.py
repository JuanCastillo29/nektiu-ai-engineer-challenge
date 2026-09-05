"""
Retrieval del asistente: trocea el documento y busca los fragmentos relevantes.

Etapa actual: BM25 léxico (monolingüe español) sobre chunks definidos por los encabezados `##`.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "sample.md"

K1 = 1.5
B = 0.75

DEFAULT_K = 2

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

    def search(self, question: str, k: int = DEFAULT_K) -> Retrieval:
        """Los k mejores. Score 0 significa no compartir ningún término con la pregunta."""
        query_tokens = tokenize(question)
        if not query_tokens:
            return Retrieval([])

        scored = [
            ScoredChunk(chunk, score)
            for i, chunk in enumerate(self.chunks)
            if (score := self.score(query_tokens, i)) > 0
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return Retrieval(scored[:k])


CHUNKS = load_chunks()
INDEX = BM25Index(CHUNKS)


def retrieve(question: str, k: int = DEFAULT_K) -> Retrieval:
    """
    Búsqueda sobre el índice por defecto.

    No hay umbral de score porque los valores de BM25 no son comparables entre preguntas;
    ese gate llegará con la similitud coseno al añadir los embeddings.
    """
    return INDEX.search(question, k)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "¿Cuánto cuesta el plan Business?"
    r = retrieve(question)
    print(f"Pregunta: {question}")
    print(f"Tokens:   {tokenize(question)}")
    for s in r.results:
        print(f"  {s.score:6.3f}  {s.chunk.title}")
