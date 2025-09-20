import os
import json
import datetime as dt
from typing import List, Dict

import requests
import streamlit as st

# ====== Config ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # opcional
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ====== Utils ======
def geocode_city(city: str) -> Dict:
    """Usa Open-Meteo Geocoding API para obtener lat/lon."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "es", "format": "json"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError("No se encontró la ciudad.")
    res = data["results"][0]
    return {
        "name": res.get("name"),
        "country": res.get("country"),
        "lat": res["latitude"],
        "lon": res["longitude"],
        "timezone": res.get("timezone", "auto"),
    }

def fetch_forecast(lat: float, lon: float, start: dt.date, end: dt.date, timezone="auto") -> Dict:
    """Consulta pronóstico diario (t max/min, prob precip, weathercode)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
        "timezone": timezone,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

WEATHER_CODE_MAP = {
    # Mapa resumido (Open-Meteo weathercode)
    0: "despejado", 1: "mayormente despejado", 2: "parcialmente nublado", 3: "nublado",
    45: "niebla", 48: "niebla escarchada",
    51: "llovizna débil", 53: "llovizna", 55: "llovizna intensa",
    56: "llovizna helada débil", 57: "llovizna helada",
    61: "lluvia débil", 63: "lluvia", 65: "lluvia fuerte",
    66: "lluvia helada débil", 67: "lluvia helada",
    71: "nieve débil", 73: "nieve", 75: "nieve fuerte",
    77: "granos de nieve", 80: "chubascos débiles", 81: "chubascos", 82: "chubascos fuertes",
    85: "chubascos de nieve débiles", 86: "chubascos de nieve fuertes",
    95: "tormenta", 96: "tormenta con granizo débil", 99: "tormenta con granizo fuerte",
}

def summarize_weather(daily: Dict) -> str:
    """Crea un resumen legible del clima en el rango elegido."""
    days = []
    for i, date in enumerate(daily["time"]):
        tmin = daily["temperature_2m_min"][i]
        tmax = daily["temperature_2m_max"][i]
        pprec = daily.get("precipitation_probability_max", [None]*len(daily["time"]))[i]
        code = daily.get("weathercode", [None]*len(daily["time"]))[i]
        desc = WEATHER_CODE_MAP.get(code, "condiciones variables")
        days.append(f"{date}: {desc}, {tmin:.0f}–{tmax:.0f}°C, prob. precip {pprec if pprec is not None else '–'}%")
    # Promedios rápidos
    avg_min = sum(daily["temperature_2m_min"]) / len(daily["temperature_2m_min"])
    avg_max = sum(daily["temperature_2m_max"]) / len(daily["temperature_2m_max"])
    wet_days = sum(1 for p in daily.get("precipitation_probability_max", []) if p is not None and p >= 40)
    summary = (
        f"Promedio térmico: mín {avg_min:.1f}°C / máx {avg_max:.1f}°C. "
        f"Días con alta chance de lluvia (≥40%): {wet_days}."
    )
    return summary + "\n" + "\n".join(days)

def rule_based_packing(avg_min: float, avg_max: float, wet_days: int, activities: List[str]) -> Dict[str, List[str]]:
    """Fallback sin OpenAI: arma lista básica según clima + actividades."""
    base = {
        "Ropa": [],
        "Calzado": [],
        "Higiene y salud": ["Cepillo/pasta", "Desodorante", "Medicación personal", "Protector solar"],
        "Tecnología": ["Celular + cargador", "Power bank", "Adaptador de enchufe (si aplica)"],
        "Documentación": ["DNI/Pasaporte", "Tarjeta de crédito/débito", "Reserva/Seguro"],
        "Varios": ["Botella reutilizable", "Gafas de sol", "Mochila día"],
    }
    # Temperatura
    if avg_max >= 27:
        base["Ropa"] += ["Remeras livianas x4–6", "Shorts x2–3", "Traje de baño", "Gorra/sombrero"]
        base["Calzado"] += ["Zapatillas cómodas", "Ojotas"]
    elif avg_min <= 8:
        base["Ropa"] += ["Campera abrigo", "Buzo/suéter x2", "Remeras térmicas", "Pantalones largos x2–3", "Buff/gorro/guantes"]
        base["Calzado"] += ["Zapatillas cerradas", "Medias térmicas"]
    else:
        base["Ropa"] += ["Campera liviana", "Buzo/suéter", "Remeras x4–5", "Pantalones x2–3"]
        base["Calzado"] += ["Zapatillas cómodas"]
    # Lluvia
    if wet_days >= 1:
        base["Varios"] += ["Pilotín/poncho", "Paraguas plegable", "Cubre mochila"]
        base["Calzado"] += ["Zapatillas que sequen rápido"]
    # Actividades
    acts = [a.lower() for a in activities]
    if any("playa" in a or "pileta" in a for a in acts):
        base["Varios"] += ["Toalla de playa", "After sun"]
    if any("trek" in a or "sender" in a or "montaña" in a for a in acts):
        base["Varios"] += ["Bastones trekking (opcional)", "Botiquín básico", "Repelente"]
        base["Calzado"] += ["Zapatillas/boots trekking"]
        base["Ropa"] += ["Campera impermeable respirable"]
    if any("noche" in a or "resto" in a or "eleg" in a for a in acts):
        base["Ropa"] += ["1 outfit más arreglado"]
        base["Calzado"] += ["Zapatos/zapatillas urbanas limpias"]
    return base

def generate_packing_with_openai(weather_brief: str, city: str, days: int, activities: List[str]) -> Dict[str, List[str]]:
    """Si hay API key, pide a OpenAI una lista argumentada y agrupada por categorías."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada.")

    import openai  # requiere paquete openai>=1.40
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
Eres un asistente de viajes. Usuario viaja a {city} por {days} días.
Resumen del clima real (Open-Meteo): 
{weather_brief}

Actividades de interés: {', '.join(activities) if activities else 'no especificadas'}.

Tarea:
1) Propón una lista de equipaje optimizada por clima, agrupada en categorías (Ropa, Calzado, Higiene y salud, Tecnología, Documentación, Varios).
2) Ajusta cantidades aproximadas pensando en {days} días.
3) Sé concreto y evita ítems redundantes. Responde en español en JSON con la forma:
{{
  "Ropa": ["...", "..."],
  "Calzado": ["..."],
  "Higiene y salud": ["..."],
  "Tecnología": ["..."],
  "Documentación": ["..."],
  "Varios": ["..."]
}}
No incluyas texto fuera del JSON.
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    content = resp.choices[0].message.content.strip()
    # Intentar parsear JSON
    try:
        data = json.loads(content)
        # Validación mínima
        if not isinstance(data, dict) or not data:
            raise ValueError("Respuesta JSON vacía.")
        return data
    except Exception:
        # Si no es JSON puro, hace un salvage simple
        return {"Ropa": [content]}

def export_txt(packing: Dict[str, List[str]]) -> str:
    lines = []
    for cat, items in packing.items():
        lines.append(f"== {cat} ==")
        for it in items:
            lines.append(f"- {it}")
        lines.append("")
    return "\n".join(lines)

# ====== UI ======
st.set_page_config(page_title="Asistente de Valija", page_icon="🧳", layout="centered")
st.title("🧳 Asistente de Valija con Clima Real")
st.caption("Destino + fechas → clima real (Open-Meteo) → lista de equipaje. Hecho por Juanma 😉")

with st.form("trip_form"):
    city = st.text_input("¿A dónde viajás?", placeholder="Ej: Río de Janeiro")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha de inicio", value=dt.date.today() + dt.timedelta(days=7))
    with col2:
        end_date = st.date_input("Fecha de regreso", value=dt.date.today() + dt.timedelta(days=14))
    activities = st.text_input("Actividades (opcional, separá por coma)", placeholder="playa, trekking, salidas nocturnas")
    submit = st.form_submit_button("Generar")

if submit:
    try:
        if not city or start_date > end_date:
            st.error("Completá la ciudad y verificá el rango de fechas.")
        else:
            geo = geocode_city(city)
            forecast = fetch_forecast(geo["lat"], geo["lon"], start_date, end_date, geo["timezone"])
            daily = forecast["daily"]
            weather_brief = summarize_weather(daily)
            st.subheader(f"📍 {geo['name']}, {geo['country']}")
            st.code(weather_brief)

            avg_min = sum(daily["temperature_2m_min"]) / len(daily["temperature_2m_min"])
            avg_max = sum(daily["temperature_2m_max"]) / len(daily["temperature_2m_max"])
            wet_days = sum(1 for p in daily.get("precipitation_probability_max", []) if p is not None and p >= 40)
            acts = [a.strip() for a in activities.split(",")] if activities else []

            # Packing con OpenAI si hay API key, si no reglas:
            try:
                packing = generate_packing_with_openai(weather_brief, geo["name"], (end_date - start_date).days + 1, acts)
                st.success("Lista generada con OpenAI ✅")
            except Exception:
                packing = rule_based_packing(avg_min, avg_max, wet_days, acts)
                st.info("Usando lógica local (sin OpenAI) ✅")

            st.subheader("✅ Lista sugerida para la valija")
            for cat, items in packing.items():
                with st.expander(cat, expanded=True):
                    for it in items:
                        st.write(f"- {it}")

            txt = export_txt(packing)
            st.download_button("⬇️ Descargar lista .txt", data=txt, file_name="lista_valija.txt", mime="text/plain")

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
