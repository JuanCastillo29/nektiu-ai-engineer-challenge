"""
Runner de evaluación sobre `eval/dataset.json`.

    python eval/run.py --retrieval          # solo retrieval (usa embeddings si hay API key)
    python eval/run.py --retrieval --lexical  # solo BM25: ni API key ni llamadas (CI)
    python eval/run.py                      # retrieval + generación (necesita OPENAI_API_KEY)
    python eval/run.py --k 3 --out r.json   # k del retriever y volcado por caso

El juez es `gpt-5.6-sol`, distinto y más capaz que el asistente a propósito;
`EVAL_JUDGE_MODEL` lo cambia.

Mide dos cosas distintas y las informa por separado: si el retriever pone delante la
sección correcta, y si la respuesta generada es honesta con lo que dice el documento.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "api"))

import rag  # noqa: E402

DATASET_PATH = Path(__file__).parent / "dataset.json"

# Juez distinto del asistente: un modelo tiende a dar por buenas sus propias respuestas.
DEFAULT_JUDGE_MODEL = "gpt-5.6-sol"

JUDGE_SYSTEM = """\
Eres un evaluador de un asistente RAG. Recibes el documento completo del que puede \
hablar el asistente, una pregunta de usuario, la respuesta que dio y los fragmentos \
que citó. Juzgas la respuesta, no el documento.

- abstiene: true si la respuesta declina responder —dice que no lo sabe o que el \
documento no lo cubre—, aunque lo diga con otras palabras.
- fundamentada: false solo si la respuesta afirma algo que el documento no dice. Una \
abstención siempre está fundamentada.
- correcta: true si es la respuesta que daría un asistente honesto que solo puede usar \
este documento: contesta lo que se pregunta, conserva los matices y condiciones del \
documento y no añade certeza; o se abstiene porque el documento no lo cubre.
- motivo: una frase explicando el juicio.\
"""

JUDGE_SCHEMA = {
    "name": "judgement",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "abstiene": {"type": "boolean"},
            "fundamentada": {"type": "boolean"},
            "correcta": {"type": "boolean"},
            "motivo": {"type": "string"},
        },
        "required": ["abstiene", "fundamentada", "correcta", "motivo"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Case:
    id: int
    category: str
    question: str
    expected_source: str | None
    probes: str | None = None

    @property
    def answerable(self) -> bool:
        """El documento cubre la pregunta; abstenerse aquí es un fallo."""
        return self.expected_source is not None


@dataclass(frozen=True)
class RetrievalOutcome:
    case: Case
    results: list[rag.ScoredChunk]

    @property
    def titles(self) -> list[str]:
        return [s.chunk.title for s in self.results]

    @property
    def rank(self) -> int:
        """Posición (1-based) de la sección esperada; 0 si no está entre las k."""
        if self.case.expected_source in self.titles:
            return self.titles.index(self.case.expected_source) + 1
        return 0

    @property
    def hit(self) -> bool:
        return self.case.answerable and self.rank > 0

    @property
    def empty(self) -> bool:
        """Sin candidatos: la app responde "No lo sé" sin llegar a llamar al modelo."""
        return not self.titles


@dataclass(frozen=True)
class GenerationOutcome:
    case: Case
    answer: str
    cited: list[str]
    judge: dict

    @property
    def abstains(self) -> bool:
        return bool(self.judge["abstiene"])

    @property
    def cites_expected(self) -> bool:
        return self.case.expected_source in self.cited


def load_cases(path: Path = DATASET_PATH) -> list[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Case(
            id=c["id"],
            category=c["category"],
            question=c["question"],
            expected_source=c["expected_source"],
            probes=c.get("probes"),
        )
        for c in data["cases"]
    ]


# ---------------------------------------------------------------- retrieval


def run_retrieval(cases: list[Case], k: int, workers: int = 1) -> list[RetrievalOutcome]:
    """En híbrido cada caso es una llamada de embedding, así que van en paralelo."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda c: RetrievalOutcome(c, rag.retrieve(c.question, k).results), cases)
        )


def retrieval_metrics(outcomes: list[RetrievalOutcome]) -> dict:
    """
    recall@k y MRR sobre las respondibles; el resto son las dos formas de fallar.

    `sin_candidatos` en respondibles es la abstención falsa: el documento contesta y la
    app dice "No lo sé". En las no respondibles es lo contrario, un acierto gratis.
    """
    answerable = [o for o in outcomes if o.case.answerable]
    unanswerable = [o for o in outcomes if not o.case.answerable]
    hits = [o for o in answerable if o.hit]

    return {
        "n": len(outcomes),
        "respondibles": len(answerable),
        "aciertos": len(hits),
        "recall_at_k": _ratio(len(hits), len(answerable)),
        "mrr_at_k": _mean([1 / o.rank for o in hits], len(answerable)),
        "sin_candidatos_respondibles": _ratio(
            sum(o.empty for o in answerable), len(answerable)
        ),
        "recuperado_pero_fuera_de_k": _ratio(
            sum(not o.hit and not o.empty for o in answerable), len(answerable)
        ),
        "no_respondibles": len(unanswerable),
        "silencio_no_respondibles": _ratio(
            sum(o.empty for o in unanswerable), len(unanswerable)
        ),
    }


def by_category(outcomes: list[RetrievalOutcome]) -> dict[str, tuple[int, int]]:
    """Categoría -> (aciertos, respondibles). Las no respondibles no puntúan aquí."""
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for o in outcomes:
        if o.case.answerable:
            tally[o.case.category][0] += o.hit
            tally[o.case.category][1] += 1
    return {c: (v[0], v[1]) for c, v in sorted(tally.items())}


# --------------------------------------------------------------- generación


def run_generation(
    outcomes: list[RetrievalOutcome], k: int, workers: int
) -> tuple[list[GenerationOutcome], dict[str, str]]:
    """Devuelve los resultados y qué modelos los produjeron."""
    import app  # importarlo antes exigiría la API key también en --retrieval

    judge_model = os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    document = rag.DATA_PATH.read_text(encoding="utf-8")

    def one(outcome: RetrievalOutcome) -> GenerationOutcome:
        case = outcome.case
        if outcome.results:  # se reusa lo ya recuperado: no se paga dos veces el embedding
            response = app.generate(case.question, outcome.results)
            answer, cited = response.answer, [s.title for s in response.sources]
        else:
            answer, cited = app.NO_SE, []  # misma puerta que en /api/chat
        judge = judge_answer(app.client, judge_model, document, case, answer, cited)
        return GenerationOutcome(case, answer, cited, judge)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, outcomes)), {"asistente": app.MODEL, "juez": judge_model}


def judge_answer(
    client, model: str, document: str, case: Case, answer: str, cited: list[str]
) -> dict:
    """Juez LLM. No ve `expected_source`: la verdad de referencia es el documento."""
    user = (
        f"Documento:\n\n{document}\n\n"
        f"Pregunta del usuario: {case.question}\n\n"
        f"Respuesta del asistente: {answer}\n\n"
        f"Fragmentos citados: {', '.join(cited) or 'ninguno'}"
    )
    completion = client.chat.completions.create(
        model=model,
        response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(completion.choices[0].message.content)


def generation_metrics(outcomes: list[GenerationOutcome]) -> dict:
    """
    Tres ejes: cuándo se abstiene, si cita bien y qué dice el juez de la respuesta.

    La abstención se mide por separado en cada mitad del dataset porque los dos errores
    no cuestan lo mismo: responder a lo que el documento no cubre es inventar.
    """
    answerable = [o for o in outcomes if o.case.answerable]
    unanswerable = [o for o in outcomes if not o.case.answerable]
    answered = [o for o in answerable if not o.abstains]
    abstained = [o for o in outcomes if o.abstains]

    return {
        "n": len(outcomes),
        "abstencion_falsa": _ratio(sum(o.abstains for o in answerable), len(answerable)),
        "abstencion_correcta": _ratio(
            sum(o.abstains for o in unanswerable), len(unanswerable)
        ),
        "respuesta_sin_cobertura": _ratio(
            sum(not o.abstains for o in unanswerable), len(unanswerable)
        ),
        "cita_la_seccion_esperada": _ratio(
            sum(o.cites_expected for o in answered), len(answered)
        ),
        "cita_al_abstenerse": _ratio(sum(bool(o.cited) for o in abstained), len(abstained)),
        "fundamentada": _ratio(
            sum(o.judge["fundamentada"] for o in outcomes), len(outcomes)
        ),
        "correcta": _ratio(sum(o.judge["correcta"] for o in outcomes), len(outcomes)),
    }


# ------------------------------------------------------------------ informe


def _ratio(n: int, total: int) -> float | None:
    return n / total if total else None


def _mean(values: list[float], total: int) -> float | None:
    return sum(values) / total if total else None


def _pct(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.1%}"


def report(
    retrieval: list[RetrievalOutcome],
    generation: list[GenerationOutcome] | None,
    k: int,
    models: dict[str, str] | None = None,
) -> None:
    m = retrieval_metrics(retrieval)
    print(f"\nRETRIEVAL (k = {k})  ·  {m['n']} casos, {m['respondibles']} respondibles"
          f"  ·  retriever {rag.retriever_name()}\n")
    print(f"  recall@{k}                     {_pct(m['recall_at_k'])}"
          f"   ({m['aciertos']}/{m['respondibles']})")
    print(f"  MRR@{k}                        {_pct(m['mrr_at_k'])}")
    print(f"  sin candidatos (fallo)       {_pct(m['sin_candidatos_respondibles'])}")
    print(f"  recuperado fuera de k        {_pct(m['recuperado_pero_fuera_de_k'])}")
    print(f"  silencio en fuera de alcance {_pct(m['silencio_no_respondibles'])}"
          f"   (de {m['no_respondibles']})")

    print("\n  por categoría")
    for category, (hits, total) in by_category(retrieval).items():
        print(f"    {category:<23}{_pct(hits / total)}   ({hits}/{total})")

    fails = [o for o in retrieval if o.case.answerable and not o.hit]
    if fails:
        print(f"\n  fallos ({len(fails)})")
        for o in fails:
            where = "sin candidatos" if o.empty else f"recuperó {o.titles}"
            print(f"    #{o.case.id:<3} {o.case.question[:46]:<48} {where}")

    if generation is None:
        return

    g = generation_metrics(generation)
    who = f"  ·  asistente {models['asistente']}, juez {models['juez']}" if models else ""
    print(f"\nGENERACIÓN  ·  {g['n']} casos{who}")
    if models and models["asistente"] == models["juez"]:
        print("  (juez y asistente son el mismo modelo: tiende a absolverse)")
    print()
    print(f"  abstención falsa (fallo)     {_pct(g['abstencion_falsa'])}")
    print(f"  abstención correcta          {_pct(g['abstencion_correcta'])}")
    print(f"  responde sin cobertura       {_pct(g['respuesta_sin_cobertura'])}")
    print(f"  cita la sección esperada     {_pct(g['cita_la_seccion_esperada'])}")
    print(f"  cita al abstenerse (fallo)   {_pct(g['cita_al_abstenerse'])}")
    print(f"  fundamentada (juez)          {_pct(g['fundamentada'])}")
    print(f"  correcta (juez)              {_pct(g['correcta'])}")

    bad = [o for o in generation if not o.judge["correcta"] or not o.judge["fundamentada"]]
    if bad:
        print(f"\n  respuestas señaladas por el juez ({len(bad)})")
        for o in bad:
            flags = "".join(
                f" -{name}" for name in ("fundamentada", "correcta") if not o.judge[name]
            )
            print(f"    #{o.case.id:<3} {o.case.question[:46]:<48}{flags}")
            print(f"         {o.judge['motivo']}")


def dump(
    path: Path,
    k: int,
    retrieval: list[RetrievalOutcome],
    generation: list[GenerationOutcome] | None,
    models: dict[str, str] | None = None,
) -> None:
    generated = {o.case.id: o for o in generation or []}
    payload = {
        "k": k,
        "retriever": {"hibrido": rag.INDEX.hybrid, "embeddings": rag.EMBEDDING_MODEL},
        "modelos": models,  # cambiar de juez cambia los números: queda registrado
        "retrieval": retrieval_metrics(retrieval),
        "por_categoria": {
            c: {"aciertos": h, "total": t} for c, (h, t) in by_category(retrieval).items()
        },
        "generacion": generation_metrics(generation) if generation else None,
        "casos": [
            {
                "id": o.case.id,
                "category": o.case.category,
                "question": o.case.question,
                "expected_source": o.case.expected_source,
                "retrieved": o.titles,
                "rank": o.rank,
                **(
                    {"answer": g.answer, "cited": g.cited, "judge": g.judge}
                    if (g := generated.get(o.case.id))
                    else {}
                ),
            }
            for o in retrieval
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVolcado en {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de NektiBot.")
    parser.add_argument("--k", type=int, default=rag.DEFAULT_K, help="fragmentos a recuperar")
    parser.add_argument("--retrieval", action="store_true",
                        help="solo retrieval: no llama al modelo de respuesta ni al juez")
    parser.add_argument("--lexical", action="store_true",
                        help="apaga los embeddings: solo BM25, sin red (para CI)")
    parser.add_argument("--limit", type=int, help="evaluar solo los N primeros casos")
    parser.add_argument("--workers", type=int, default=4, help="llamadas en paralelo")
    parser.add_argument("--out", type=Path, help="volcar el resultado por caso en JSON")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.lexical:
        rag.INDEX = rag.HybridIndex(rag.CHUNKS)  # sin embedder: solo BM25

    cases = load_cases()[: args.limit]
    retrieval = run_retrieval(cases, args.k, args.workers)
    generation, models = (
        (None, None) if args.retrieval else run_generation(retrieval, args.k, args.workers)
    )

    report(retrieval, generation, args.k, models)
    if args.out:
        dump(args.out, args.k, retrieval, generation, models)


if __name__ == "__main__":
    main()
