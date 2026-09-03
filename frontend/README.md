# Frontend

Aquí construyes la interfaz de chat. No hay nada de partida a propósito: queremos ver cómo lo montas.

## Qué pedimos

- Una interfaz de chat sencilla: caja de texto, envío, e historial de mensajes en pantalla.
- Que llame a tu backend (`POST /api/chat`) y muestre la respuesta y, si las hay, las fuentes citadas.
- Cuidado básico de UX: estado de "cargando", y que no se rompa si el backend falla.

## Sugerencias (no obligatorias)

- Next.js o React son una opción natural y despliegan muy bien en Vercel. Usa lo que domines.
- Puedes _vibe-codear_ el frontend con Cursor/Copilot — está perfecto. Recuerda que después
  tendrás que explicar y modificar el código, así que revisa lo que genera.
- Configura la URL del backend por variable de entorno (p. ej. `NEXT_PUBLIC_API_URL`) para
  que funcione tanto en local como desplegado.

## Despliegue

Despliega donde te resulte cómodo (Vercel, Render, Netlify, Hugging Face Spaces…). Lo único
imprescindible: un **enlace público** que funcione en incógnito. Recuerda no exponer tu API
key en el frontend — las llamadas al LLM deben ir a través de tu backend.
