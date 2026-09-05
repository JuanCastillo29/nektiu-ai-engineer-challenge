# NektiBot — asistente RAG sobre un documento

Reto técnico de **Nektiu** (AI Engineer / Data). Chat que responde preguntas sobre
[`data/sample.md`](data/sample.md) usando **solo** lo que dice el documento: cita los
fragmentos en los que se apoya y responde "No lo sé" cuando el documento no lo cubre.

**App desplegada:** https://nektibot.onrender.com

> ⏱️ Está en el plan gratuito de Render, que apaga el servicio tras ~15 min sin tráfico.
> Si lleva rato sin uso, **la primera carga tarda ~40 s** en levantarlo. Como el mismo
> servicio sirve la página y la API, ese coste se lo come la carga inicial del HTML: una
> vez abierta, el chat responde a velocidad normal.

---

## Cómo ejecutarlo en local

Requisitos: Python 3.10+ y una API key de OpenAI.

```bash
cd api
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # y rellena OPENAI_API_KEY
uvicorn app:app --reload --port 8000
```

Abre <http://localhost:8000/> — el backend sirve también el frontend.

La clave se lee de `api/.env` (ignorado por git) o de la variable de entorno, que tiene
prioridad. `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL` y `OPENAI_TEMPERATURE` son opcionales.
Sin API key el retrieval funciona igual, solo que sin la mitad densa.

### Tests

```bash
python -m pytest tests -q      # 113 tests, sin red: ni modelo ni embeddings
```

### Endpoints

- `GET /api/health` → `{ "status": "ok", "model": "...", "hybrid_retrieval": true,
  "embedding_model": "..." }`

  Los valores efectivos, para comprobar contra el despliegue qué está corriendo de verdad:
  qué modelos, y si el retrieval va híbrido o ha degradado a solo BM25.

- `POST /api/chat` → `{ "question": "..." }` →
  `{ "answer": "...", "sources": [{ "title": "...", "text": "..." }] }`

  Cada fuente lleva el título de la sección y su texto completo, para que el frontend
  pueda mostrar el fragmento y el usuario contraste la respuesta.

---

## Cómo está construido

```
data/sample.md ──► chunks por encabezado `##` ──┬─► índice BM25 (léxico)
                                                └─► vectores (embeddings)
                                                        │
pregunta ───────────────────────────────────────────────┤ retrieve(k=2)
                                                        ▼
                                          RRF sobre los dos rankings
                                     (candidato: bm25 > 0  ó  coseno ≥ 0,30)
                                                        │
                          ¿sin candidatos? ──sí──►  "No lo sé" (sin llamar al modelo)
                                    │ no
                                    ▼
                fragmentos numerados + system prompt ──► LLM (JSON estricto)
                                    │
                                    ▼
                    { answer, used: [1,2] } ──► fuentes citadas
```

- [`api/rag.py`](api/rag.py) — troceado y retrieval.
- [`api/app.py`](api/app.py) — API, prompt, generación y citas.
- [`frontend/index.html`](frontend/index.html) — la interfaz entera, en un fichero.
- [`api/observability.py`](api/observability.py) — logs JSON y trazas (ver abajo).
- [`eval/run.py`](eval/run.py) — el runner de la evaluación (ver abajo).

## Decisiones

**Chunking por secciones `##`, no por ventana de N tokens.** El documento ya viene
estructurado en secciones semánticas de tamaño parecido. Partir por encabezado da
fragmentos que coinciden con una unidad de significado y que además tienen **un título
natural para citar** — el usuario ve "Planes y precios", no "chunk 3". El preámbulo
anterior al primer `##` es metadato del documento y se descarta.

**Retrieval híbrido: BM25 + embeddings, fusionados por RRF.** La primera versión era
solo BM25 —determinista, sin dependencias, sin coste y sin API key—, y la evaluación
midió exactamente dónde se rompía: **6 de sus 9 fallos eran desajuste de vocabulario**,
preguntas que el documento sí responde pero con otras palabras (*horario* → "de 9:00 a
18:00", *cuesta* → "49 €/mes"). Ningún ajuste del léxico arregla eso, porque no hay
término que compartir. Los embeddings sí: `text-embedding-3-small` sobre los 7 chunks,
una sola llamada al arrancar, y una por pregunta.

Los dos retrievers se **fusionan por RRF** (Reciprocal Rank Fusion: cada uno aporta
`1/(60 + puesto)`), no por suma ponderada de scores. Un BM25 de 3,1 y un coseno de 0,42
no viven en la misma escala, así que sumarlos exige inventar pesos y renormalizar por
pregunta; RRF solo mira el **orden**, que es lo único que ambos producen de forma
comparable, y no tiene nada que calibrar.

Un chunk es candidato si **alguna** de las dos vías lo propone: `bm25 > 0` o
`coseno ≥ 0,30`. La unión es lo que recupera la pregunta sin vocabulario común; el suelo
de coseno es lo que impide que, al haber vector para todo, los 7 chunks sean siempre
candidatos y desaparezca la puerta de "sin candidatos".

**Si el proveedor de embeddings falla, el retrieval no falla**: el índice denso se marca
caído y la búsqueda sigue en BM25. Degradar a un retrieval peor es mejor que devolver un
502 en cada pregunta, y por eso el léxico se mantiene aunque el denso lo supere.

El BM25 está implementado a mano —IDF, saturación por `k1`, normalización por longitud
del documento— con tests *golden* de la fórmula. Al tokenizar se normalizan tildes y mayúsculas, se quitan stopwords españolas, se unen
los separadores de millar (`2.000` → `2000`) y se aplica un stemmer mínimo de plurales.
Se indexa el título junto al cuerpo, porque el encabezado suele contener el término por
el que se pregunta.

**`k = 2`, y el único umbral es el del coseno.** Los valores de BM25 **no son comparables
entre preguntas** (dependen del IDF de los términos concretos), así que ahí el filtro
sigue siendo *score > 0*. El coseno sí es comparable, así que su suelo es calibrable — y
está calibrado, barriendo el dataset entero:

| suelo de coseno | recall@2 | sin candidatos (respondibles) | silencio en fuera de alcance |
|---|---|---|---|
| sin denso (solo BM25) | 30/39 | 6 | 6/11 |
| 0,20 | 37/39 | 0 | 0/11 |
| **0,30** | **37/39** | **1** | **1/11** |
| 0,35 | 36/39 | 1 | 4/11 |
| 0,40 | 35/39 | 2 | 6/11 |

Entre 0,20 y 0,30 el recall no se mueve, así que se elige el más alto de los dos: conserva
alguna abstención gratuita sin pagar recall. Por encima de 0,35 empieza a costar preguntas
respondibles. Aviso honesto: el suelo está calibrado sobre las mismas 50 preguntas con las
que se mide, así que es calibración, no validación; con un corpus real esto pide partir el
dataset en dos.

Subir a `k = 3` no aporta nada con este suelo (37/39 también), así que `k` se queda en 2:
menos contexto, menos ruido y menos que pagar por pregunta.

**Cuándo decir "No lo sé": dos puertas independientes.**

1. **Sin candidatos** (en `app.py`): se responde `"No lo sé."` y se ahorra la llamada al
   modelo. Barato, determinista y no puede alucinar. Con el híbrido esta puerta se cierra
   casi siempre —el denso encuentra algo parecido para casi cualquier pregunta— y pasa de
   resolver 6 de las 11 fuera de alcance a resolver 1. Es el precio del recall: el peso de
   abstenerse se traslada a la puerta 2, y la evaluación confirma que aguanta (la
   abstención correcta se queda en el 91 %).
2. **Con candidatos pero sin respuesta**: lo decide el modelo, con una regla explícita en
   el system prompt —*"que un fragmento trate el tema no significa que contenga la
   respuesta"*. Es el caso difícil y el más frecuente: `¿Cumple con el RGPD?` recupera
   "Privacidad y datos" con buen score, y esa sección no dice nada del RGPD.

**Las citas las declara el modelo, no el retriever.** La respuesta viene en JSON con un
campo `used` con los índices de los fragmentos que sustentan lo dicho. Mostrar sin más lo
recuperado sería mentir a medias: con `k = 2` casi siempre sobra uno. Si el modelo
responde "No lo sé", `used` va vacío y no se cita nada.

**Structured output estricto, con un reintento.** `json_schema` con `strict: true` y
`additionalProperties: false`, más un segundo intento si el JSON sale ilegible. A la
segunda se devuelve un 502 en vez de enseñarle basura al usuario.

**Frontend en HTML plano.** Son ~100 líneas de lógica; React añadiría toolchain y build
sin aportar nada. Se pinta con `textContent`, nunca `innerHTML`, porque el texto viene del
modelo. Las fuentes van bajo cada respuesta en un `<details>` con el fragmento entero,
para poder contrastar lo que dice el bot con lo que dice el documento.

**Un solo servicio.** FastAPI sirve la API y monta el frontend estático en `/`. Un único
despliegue, un único enlace público, y `fetch("/api/chat")` con URL relativa: sin CORS y
sin variable de entorno con la URL del backend. La API key vive solo en el servidor.

**Backend sin estado.** `POST /api/chat` solo acepta `question`; el hilo de conversación
es únicamente visual. Con preguntas independientes sobre un manual no aporta, y evita
tener que decidir qué parte del historial entra en el contexto.

---

## Observabilidad

Cada petición escribe **una línea de JSON por etapa** en stdout, que es de donde leen
Render y cualquier agregador. Las etapas son *spans*: `http.request` cuelga de la traza,
`chat` cuelga de él, y de `chat` cuelgan `retrieval`, `embeddings.*` y `llm.chat`.

```
$ curl -s localhost:8000/api/chat -H 'content-type: application/json'        -d '{"question": "¿Cuánto cuesta el plan Business?"}'

{"ts":"…","level":"INFO","msg":"retrieval","trace_id":"4bf92f…","span_id":"714cc4…",
 "parent_span_id":"1a8009…","duration_ms":142.6,"k":2,"hybrid":true,"grounded":true,
 "titles":["Planes y precios","Fuentes de datos soportadas"],
 "scores":[{"rrf":0.0325,"bm25":1.94,"cos":0.61},{"rrf":0.0161,"bm25":1.75,"cos":0.28}]}
{"ts":"…","msg":"llm.chat","duration_ms":1893.4,"model":"gpt-5.6-luna","attempt":1,
 "prompt_tokens":812,"completion_tokens":41,"total_tokens":853,"status":"ok"}
{"ts":"…","msg":"chat","duration_ms":2041.7,"question":"¿Cuánto cuesta el plan Business?",
 "outcome":"answered","sources":["Planes y precios"]}
{"ts":"…","msg":"http.request","duration_ms":2044.9,"method":"POST","path":"/api/chat",
 "status_code":200}
```

**Por qué trazas y no solo líneas de log.** Un RAG falla por etapas y todas terminan en la
misma respuesta mediocre: la lenta puede ser el retrieval denso o el modelo, y la mala
respuesta puede venir de un fragmento que no llegó o de un prompt que no lo usó. Con
`duration_ms` por span y los scores de lo recuperado en el mismo `trace_id`, esa pregunta
se contesta mirando una traza en vez de reproduciendo el caso.

Las preguntas que responde el log tal como está:

- **Dónde se va el tiempo.** El arranque en frío tiene span propio (`embeddings.index`,
  los vectores de los 7 chunks) para que no se confunda con la latencia normal.
- **Si el retrieval ha degradado a solo BM25.** `hybrid` va en cada span de `retrieval`, y
  la caída deja un `dense_retrieval_disabled` con la excepción. Es el fallo silencioso del
  sistema —responde 200 y solo baja el recall—, así que es lo primero que hay que ver.
- **Cuánto se abstiene y por qué.** `outcome: not_grounded` es abstención sin llamar al
  modelo, con los `scores` al lado para juzgar si el suelo de coseno se pasó de estricto.
- **Qué cuesta.** Tokens por llamada, y `attempt`/`model_response_unparsable` para saber si
  se está pagando el mismo prompt dos veces.

**Sin dependencias nuevas.** Los campos son los de OpenTelemetry (`trace_id`, `span_id`,
`parent_span_id`), y el servicio continúa el `traceparent` que le llegue y devuelve
`X-Trace-Id` en la respuesta —el usuario puede reportar un fallo con su identificador—,
pero no exporta a un colector: el SDK son varios megas y un proceso más en un plan free
donde el cuello de botella es el arranque en frío. Migrar cuando haga falta es cambiar
`observability.py`, porque el resto del código solo ve `obs.span(...)`.

Se ajusta con `LOG_LEVEL` (por defecto `INFO`), `SERVICE_NAME` y `LOG_QUESTION_CHARS`,
que recorta el texto de las preguntas en el log; a `0` deja solo su longitud, para un
despliegue donde las preguntas de los usuarios sean dato sensible.

---

## Evaluación

[`eval/dataset.json`](eval/dataset.json) tiene **50 preguntas** escritas como las haría un
usuario real, clasificadas en 10 categorías según el tipo de reto que plantean (directa,
fuera de alcance, matiz, inferencia negativa, síntesis, ambigüedad…). 11 no las cubre el
documento: ahí lo correcto es abstenerse.

Los casos marcados con `probes` son los interesantes: no están para subir el porcentaje,
sino para señalar dónde se rompe el sistema.

[`eval/run.py`](eval/run.py) es el runner. Mide las dos etapas por separado, porque
arreglan cosas distintas:

```bash
python eval/run.py --retrieval --lexical   # solo BM25: ni API key ni red (esto va en CI)
python eval/run.py --retrieval             # retrieval híbrido: 50 llamadas de embedding
python eval/run.py --workers 8             # + generación (~88 llamadas: respuesta y juez)
python eval/run.py --k 3 --out r.json      # barrer k, volcar el detalle por caso
```

`--lexical` apaga los embeddings, que es también la forma de medir cuánto aportan: los dos
retrievers se comparan corriendo el mismo runner dos veces.

**Métricas de retrieval** (con `--lexical`, deterministas y sin coste): `recall@k` y `MRR@k`
sobre las 39 respondibles, más las dos formas de fallar separadas —*sin candidatos*, que
es la abstención falsa, y *recuperado pero fuera de k*, que se arregla subiendo `k`— y el
desglose por categoría. En las 11 no respondibles no se mide recall sino cuántas se
resuelven ya en el retriever, sin gastar una llamada.

**Métricas de generación**: la abstención se mide en las dos mitades del dataset por
separado, porque los errores no cuestan lo mismo (responder a lo que el documento no cubre
es inventar); la cita se compara con `expected_source`; y un **juez LLM** puntúa
`fundamentada` (¿afirma algo que el documento no dice?) y `correcta` (¿responde lo que se
pregunta, con los matices del documento, o se abstiene porque de verdad no está?). El juez
recibe el documento entero y **no ve `expected_source`**: la verdad de referencia es el
documento, no la etiqueta del dataset, así que no se limita a premiar lo que ya esperamos.

**El juez es `gpt-5.6-sol`, más capaz que el modelo que responde** (`EVAL_JUDGE_MODEL` lo
cambia). Juzgar es más difícil que contestar: el asistente ve dos fragmentos y extrae; el
juez lee el documento entero y tiene que decidir si una abstención estaba justificada o si
la respuesta perdió un matiz — razonamiento, no extracción. Un juez menos capaz que el
sistema evaluado no detecta los fallos finos, que son justo los que importan aquí.

El coste extra no es un problema porque **la evaluación es puntual y offline**: no está en
el camino de ninguna petición de usuario. La pasada entera son 88 llamadas —38 de respuesta
y 50 de juicio— y sale por **~0,22 $**, de los que 0,21 son del juez, que se lleva el
documento completo en cada llamada. Pagar diez veces más en la medida que en el servicio
es buen negocio cuando la medida se ejecuta una vez por cambio y el servicio, siempre.
`--retrieval` es gratis, y es el que va en CI.

### Resultado actual

El sistema arrancó con retrieval solo léxico. La evaluación señaló el cuello de botella
—6 de sus 9 fallos de retrieval no recuperaban **nada** ante una pregunta que el documento
sí responde— y el híbrido es la respuesta a esa medida. Los mismos 50 casos, antes y
después:

| | BM25 | Híbrido (BM25 + embeddings, RRF) |
|---|---|---|
| recall@2 | 76,9 % (30/39) | **94,9 % (37/39)** |
| MRR@2 | 0,70 | **0,92** |
| Sin candidatos en respondibles (fallo) | 15,4 % — 6/39 | **2,6 % — 1/39** |
| Recuperado pero fuera de k | 7,7 % | 2,6 % |
| Silencio en fuera de alcance (ahorro) | 54,5 % | 9,1 % |
| Correcta (juez) | 84 % | **96 %** |
| Abstención falsa (respondibles) | 26 % — 10/39 | **10,3 % — 4/39** |
| Abstención correcta (fuera de alcance) | 91 % — 10/11 | 91 % — 10/11 |
| Responde sin cobertura | 9 % — 1/11 | 9 % — 1/11 |
| Fundamentada (juez) | 100 % | **100 %** |
| Cita la sección esperada | 100 % | 100 % |
| Cita al abstenerse | 5 % | 0 % |

Las seis preguntas que el léxico dejaba sin ningún candidato —*horario*, *limitaciones*,
*cuesta*, *dirigido*, *infraestructura propia*, *actualizar un ticket*— las recupera ahora
el denso, todas menos *¿a quién está dirigido?*. **Lo que no se movió es tan importante como lo que subió**: fundamentada sigue
en 100 % y la abstención correcta en 91 %, así que el recall no se compró inventando.

**Lo que queda** (asistente `gpt-5.6-luna`, juez `gpt-5.6-sol`):

| Caso | Qué pasa |
|---|---|
| #4 *¿A quién está dirigido?* | Único "sin candidatos" que queda: el coseno con "Qué es NektiBot" es 0,216, por debajo del suelo de 0,30. Bajarlo a 0,20 lo arregla y cuesta la última abstención gratuita. |
| #23 *¿Admite hojas de cálculo de Excel?* | Recupera "Límites conocidos" en vez de "Fuentes de datos soportadas". Fallo de ranking, no de candidatos: con `k = 3` entra. |
| #32 *¿Puede responder en japonés?* | **Ya no es un fallo de retrieval**: trae "Idiomas", que enumera los seis idiomas soportados, y aun así el modelo se abstiene en vez de concluir que el japonés no está. Es la regla 4 del prompt (inferencia negativa sobre una enumeración exhaustiva) sin aplicar. |

El diagnóstico se ha desplazado: **antes el techo era el retriever; ahora lo es el
prompt**. De los 2 casos que el juez marca incorrectos, uno es retrieval (#4) y el otro es
generación pura (#32), y ese ya no se arregla tocando el índice.

El coste añadido es despreciable: `text-embedding-3-small` a 0,02 $/1M tokens son ~700
tokens para indexar el documento entero y ~15 por pregunta —del orden de 0,00002 $ cada
una, cuatro órdenes de magnitud por debajo de la llamada de respuesta— más una llamada de
red de ~150 ms en el camino de la petición. La pasada completa de evaluación sigue
costando **~0,22 $**, de los que 0,21 son del juez.

**El juez es parte del instrumento, así que el volcado guarda cuál se usó.** Cambiar de
juez cambia el número: un porcentaje sin decir quién lo puntuó no es comparable con nada,
ni siquiera consigo mismo en otra fecha. Por eso `modelos` y `retriever` van en el JSON
junto a las métricas, y el informe los imprime en la cabecera.

Un aviso al leer estos números: con `temperature = 1`, varias respuestas cambian entre
pasadas. Las diferencias de uno o dos casos son ruido, no señal.

## Qué mejoraría con más tiempo

Por orden de impacto, después de que el híbrido moviera el cuello de botella al prompt:

1. **La inferencia negativa sobre enumeraciones** (#32, *¿japonés?*): el fragmento correcto
   llega y el modelo se abstiene igual. La regla 4 del prompt ya lo cubre en texto, pero no
   se aplica; toca reescribirla con un ejemplo y volver a medir. Es el único fallo que
   queda que no es de retrieval.
2. **CI/CD de verdad.** Hoy Render despliega solo en cada push a `main`, con
   `/api/health` como única puerta. Le faltan tres piezas:

   - **`eval/run.py --retrieval --lexical` en CI**, que no cuesta nada y no necesita API
     key, para que los números de arriba se recalculen solos en cada cambio en vez de
     copiarse a mano a este README, y con un umbral que rompa el build si el recall baja.
     La parte con juez y con denso, a mano antes de cada release: cuesta dinero y con
     `temperature = 1` no es reproducible.
   - **Que el deploy lo dispare el CI, no el push**: `autoDeploy: false` y el *deploy
     hook* de Render desde GitHub Actions solo si pasan tests y evaluación. Después, un
     **smoke test que mire el cuerpo de `/api/health`, no el status**: el retrieval está
     hecho para degradar a BM25 si los embeddings fallan, así que con una API key mal
     copiada el servicio responde 200, Render da el deploy por bueno y lo único que pasa
     es que el recall cae al 76,9 % sin que nadie lo vea. Para eso `/api/health` publica
     `hybrid_retrieval` y `model`. El orden importa —los vectores son perezosos y el denso
     solo se marca caído tras el primer fallo real—: antes del health, un `POST /api/chat`
     con una pregunta que solo recupera el denso (*¿cuánto cuesta?*).
   - **Rollback en dos tiempos**: el botón de Render vuelve a un deploy anterior en
     minutos, pero no toca el repo y el siguiente push reintroduce el fallo; el arreglo es
     el `revert`. Funciona porque los modelos viven en `render.yaml` y no en un formulario
     web: el estado del despliegue se deduce del `git log` y el `revert` devuelve también
     el modelo. Mismo argumento que con el juez en la evaluación, aplicado al despliegue.
3. **Calibrar el suelo de coseno contra un dataset separado.** Hoy está elegido sobre las
   mismas 50 preguntas que lo miden. Con más casos, partir en calibración y validación —y
   entonces sí bajarlo a 0,20, que arregla #4, midiendo qué cuesta en falsas respuestas.
4. **Partir `abstiene` en total y parcial** en la rúbrica del juez, para que "no consta,
   pero esto es lo que sí dice el documento" deje de contarse como abstención y de
   penalizar por citar.
5. **Streaming** de la respuesta, para que la espera no sea una pantalla quieta.
6. Con un corpus grande: chunking con solape, **vectores persistidos** (hoy se recalculan
   al arrancar el proceso; con 7 chunks es una llamada, con 7.000 no), caché de embeddings
   de pregunta y rate limiting.

---

## Despliegue

Un único servicio web en Render, definido en [`render.yaml`](render.yaml). Los modelos
(`OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`) van versionados en el blueprint; `OPENAI_API_KEY` va con `sync: false` y se
rellena a mano en el dashboard, para que el secreto no toque el repositorio.

Para comprobar qué está corriendo en el despliegue:

```bash
curl https://nektibot.onrender.com/api/health
```
