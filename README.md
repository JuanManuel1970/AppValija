"Este proyecto lo desarrollé con bastante ayuda de IA (ChatGPT – GPT-5 Thinking) como pair programmer para:

mejoras de UX y prompts para OpenAI,

refactors y manejo de session_state en Streamlit,

solución de bugs de deploy en Railway,

y redacción de este README.

Todo el código fue revisado y adaptado por mí antes del deploy."




# 🧳 Asistente de Valija con Clima Real
Genera una **lista de equipaje** a partir del **clima real** del destino (Open-Meteo).  
Permite perfiles (con niños/negocios), optimización para **carry-on**, recálculo de **cantidades por días**, filtro y **exportar TXT/CSV**.  
Opcionalmente usa **OpenAI** para crear la lista; si no hay crédito/clave, cae en una **lógica local por reglas**.

**Demo:** https://appvalija-production.up.railway.app/

---

## ✨ Funcionalidades

- 🔎 **Geocoding** + **pronóstico diario** (Open-Meteo).
- 🗓️ Vista previa de **días seleccionados** según rango de fechas.
- 🧠 Generación de lista con **OpenAI (opcional)** o **reglas locales** (fallback automático).
- 👶👔 **Perfiles**: con niños / negocios (agrega ítems específicos).
- 🧦 **Cantidades** ajustadas por días, lavadora y carry-on.
- 🌧️ Detección de **días lluviosos** (≥40% prob. precipitación).
- 🇦🇷 Ajuste de **documentación** (en Argentina oculta pasaporte y sugiere DNI).
- ✅ Checklists por categoría + **Marcar / Desmarcar / Resetear** (debajo del progreso).
- 🔁 “**Regenerar**” con enfoques: *Más detalle*, *Minimalista (carry-on)*, *Enfocar actividades*.
- 🔎 **Filtro** por texto.
- ⬇️ **Exportar** lista a **TXT** y **CSV**.

---

## 🧰 Stack técnico

- 🐍 **Python 3.10+** + **Streamlit**
- 🌦 **Open-Meteo API** (geocoding + forecast)
- 🤖 **OpenAI SDK** (opcional, `chat.completions`)
- 📦 **Pandas**, **Requests**
- 🚀 **Railway** (deploy y logs)
- ⚡ Cache con `st.cache_data` y `st.cache_resource`

---

## 🚀 Ejecución local

```bash
python -m venv .venv
# Activar entorno:
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt

# Opcional: crear .env con
# OPENAI_API_KEY=sk-xxxx
# OPENAI_MODEL=gpt-4o-mini
# LOGLEVEL=INFO

streamlit run streamlit_app.py

'''


