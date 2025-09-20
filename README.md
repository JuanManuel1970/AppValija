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

OpenAI SDK (opcional, chat.completions), Pandas, Requests

Railway (deploy)

Caching con st.cache_data y st.cache_resource

🗺️ Cómo funciona

Destino + fechas → geocoding (Open-Meteo) → pronóstico diario.

Resumen climático (promedios, días con lluvia).

Generación:

Si hay OPENAI_API_KEY (y modo lo permite): OpenAI genera JSON con categorías y cantidades.

Si no, reglas locales según temperatura, lluvia, actividades y perfiles.

Normalización: asegura cantidades coherentes, agrega extras de perfil, ajusta documentación local y deduplica.

UI de checklists, progreso, filtro y exportación.

🚀 Ejecutar localmente

Requisitos: Python 3.10+ y pip.

# 1) Clonar e instalar
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# 2) (Opcional) Variables de entorno → archivo .env en la raíz
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini
# LOGLEVEL=INFO

# 3) Ejecutar
streamlit run streamlit_app.py


Si no configurás OPENAI_API_KEY, la app usa la lógica local por reglas.

⚙️ Variables de entorno

En Railway o .env:

# Opcional para IA (si no está, usa reglas locales)
OPENAI_API_KEY=sk-********************************
OPENAI_MODEL=gpt-4o-mini

# Logs (INFO/DEBUG/WARNING)
LOGLEVEL=INFO


Importante (Railway): OPENAI_API_KEY debe contener solo la key en una sola línea.
No mezcles OPENAI_MODEL dentro del mismo valor (evita saltos de línea).

🧪 Actividades (presets)

En el formulario: multiselect con playa, trekking, salidas nocturnas, nieve, montaña + “otros”.
Las reglas locales reconocen expresiones como “trek/sender/montaña”, “noche/resto/eleg” y activan ítems contextuales.

📦 Exportar

TXT con encabezado (destino, fechas, días, perfiles/flags).

CSV con columnas: destino, fechas, días, categoría, ítem, estado (empacado).

🧯 Observabilidad & performance (Railway)

LOGLEVEL=INFO para ver hitos (geocoding, forecast, IA OK/fallback).

Cache de geocoding (6h) y forecast (30m) para reducir latencia y consumo.

Si el servicio “duerme”, podés mantenerlo vivo con un ping externo cada 15–30 min.

📚 Endpoints/Servicios usados

Open-Meteo Geocoding: https://geocoding-api.open-meteo.com/v1/search

Open-Meteo Forecast: https://api.open-meteo.com/v1/forecast

OpenAI Chat Completions (opcional)

🧾 Requisitos

requirements.txt:

streamlit==1.38.0
pandas==2.2.2
python-dotenv==1.0.1
requests==2.32.3
openai==1.43.0
httpx==0.27.2

🔍 Roadmap

Más reglas por actividad (deportes acuáticos, camping, bici).

Guardado/restauración de listas personalizadas.

Internacionalización ampliada (más idiomas).

Captura opcional de email para enviarse la lista.

🤝 Transparencia (ayuda de IA)

Este proyecto lo desarrollé con bastante ayuda de IA (ChatGPT – GPT-5 Thinking) como pair programmer para:

refactors y mejoras de UX,

prompts para la integración con OpenAI,

solución de bugs (deploy en Railway, diferencias de SDK, manejo de session_state en Streamlit),

redacción de este README.

Todo el código fue revisado y adaptado por mí antes del deploy.

🧑‍💻 Autor

Juan Manuel — GitHub

📄 Licencia

MIT (o la que prefieras). Si usás MIT, agregá un LICENSE al repo.
