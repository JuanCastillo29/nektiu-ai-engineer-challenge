# Nektiu — AI Engineer Challenge

Bienvenido/a al reto técnico de **Nektiu** para el puesto de **AI Engineer / Data Junior**.

No buscamos que resuelvas un examen teórico: queremos verte **construir algo que funcione de punta a punta** y, sobre todo, que **entiendas y sepas defender** lo que has hecho. Puedes (y te animamos a) usar herramientas de desarrollo asistido por IA como Cursor, Copilot o similares — forman parte de nuestro día a día.

---

## El reto

Construye y **despliega** una pequeña aplicación de chat que responda **preguntas sobre un documento** (un asistente RAG sencillo).

Te damos un backend mínimo en FastAPI (`/api/app.py`) que hoy solo llama al modelo. Tu trabajo es convertirlo en un asistente **fundamentado en el documento** y ponerle una cara.

### Qué tiene que hacer la app

1. **Responder a partir del documento** (`data/sample.md`, o uno tuyo): recupera los fragmentos relevantes y construye la respuesta a partir de ellos (RAG).
2. **Ser honesta:** si el documento no contiene la respuesta, debe decir **"No lo sé"** en lugar de inventar.
3. **Citar la fuente:** devolver qué fragmento(s) ha usado para responder.
4. **Tener frontend:** una interfaz de chat simple (puedes _vibe-codearla_ con Cursor) que consuma el backend.
5. **Estar desplegada:** en Vercel, Render, Hugging Face Spaces o donde prefieras, con un enlace público que funcione.

### Bonus (opcional, suma pero no es obligatorio)

- Una mini-evaluación: 3–5 preguntas con su respuesta esperada, mostrando que el sistema acierta (y que dice "No lo sé" cuando toca).
- Streaming de la respuesta, historial de conversación, o lo que consideres que aporta.

---

## Cómo empezar

1. **Haz un fork** de este repositorio a tu cuenta de GitHub.
2. Clónalo y ábrelo en tu editor (Cursor, VS Code…).
3. Levanta el backend siguiendo [`api/README.md`](api/README.md).
4. Implementa el RAG y el frontend (ver [`frontend/README.md`](frontend/README.md)).
5. Despliega y comprueba el enlace público en una ventana de incógnito.

> Necesitarás una API key de un proveedor de LLM (por defecto OpenAI). Si tienes cualquier problema de acceso a modelos, escríbenos y lo resolvemos — no queremos que el coste sea una barrera.

---

## Entrega

Sigue las instrucciones de [`docs/SUBMISSION.md`](docs/SUBMISSION.md). En resumen, envíanos:

- El enlace a **tu repositorio** (el fork con tu trabajo).
- El **enlace público** de la app desplegada.
- Un **README** breve explicando tus decisiones y qué mejorarías con más tiempo.
- Un vídeo de **5 minutos** (Loom o similar) enseñando la app y el código — **o** dínoslo y lo vemos en directo en la siguiente entrevista.

**Tiempo estimado:** 3–4 horas. **Plazo:** una semana desde que recibes el reto (si necesitas otra fecha, dínoslo sin problema).

---

## Qué valoramos

- Que **funcione y esté desplegada** (mejor algo sencillo y sólido que muchas features a medias).
- **Claridad del razonamiento**: por qué tomaste cada decisión.
- **Rigor**: que la app sea honesta ("No lo sé"), que cite fuentes, y que hayas pensado cómo comprobar que responde bien.
- **Criterio con la IA**: usar Cursor/Copilot está perfecto; lo que miramos es que entiendas y sepas **explicar y modificar** el código resultante.

En la siguiente entrevista te pediremos que nos **expliques tu código** y hagas algún **cambio en vivo**, así que asegúrate de conocer bien lo que entregas.

¡Mucha suerte! Cualquier duda, estamos a un email de distancia.

— Equipo de Nektiu
