🧳 Asistente de Valija con Clima Real

Genera una lista de equipaje a partir del clima real del destino (Open-Meteo).
Permite perfiles (con niños/negocios), optimización para carry-on, recálculo de cantidades por días, filtro y exportar TXT/CSV.
Opcionalmente usa OpenAI para crear la lista; si no hay crédito/clave, cae en una lógica local por reglas.

Demo: https://appvalija-production.up.railway.app/

✨ Funcionalidades

🔎 Geocoding + pronóstico diario (Open-Meteo).

🗓️ Vista previa de días seleccionados según rango de fechas.

🧠 Generación de lista con OpenAI (opcional) o reglas locales (fallback automático).

👶👔 Perfiles: con niños / negocios (agrega ítems específicos).

🧦 Cantidades ajustadas por días, lavadora y carry-on.

🌧️ Detección de días lluviosos (≥40% prob. precipitación).

🇦🇷 Ajuste de documentación (en Argentina oculta pasaporte y sugiere DNI).

✅ Checklists por categoría + Marcar / Desmarcar / Resetear (debajo del progreso).

🔁 “Regenerar” con enfoques: Más detalle, Minimalista (carry-on), Enfocar actividades.

🔎 Filtro por texto.

⬇️ Exportar lista a TXT y CSV.

🧰 Stack técnico

Python 3.10+, Streamlit

Open-Meteo APIs (geocoding + forecast)

OpenAI SDK (opcional, chat.completions)

Requests, Pandas

Railway (deploy)

Cache con st.cache_data y st.cache_resource

🗺️ Cómo funciona

Destino + fechas → geocoding (Open-Meteo) → pronóstico diario.

Resumen climático (promedios, días con lluvia).

Generación:

Si hay OPENAI_API_KEY (y modo lo permite): OpenAI genera JSON con categorías y cantidades.

Si no, reglas locales según temperatura, lluvia, actividades y perfiles.

Normalización: asegura cantidades coherentes, agrega extras de perfil, ajusta documentación local y deduplica.

UI de checklists, progreso, filtro y exportación.
