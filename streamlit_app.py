# streamlit_app.py
import os
import json
import logging
import unicodedata
import datetime as dt
from typing import List, Dict

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI  # SDK nuevo

load_dotenv(override=True)

# ================== Config inicial ==================
st.set_page_config(page_title="Asistente de armado de Valija", page_icon="🧳", layout="centered")

# ==== Logging básico a stdout (Railway lo capta) ====
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("valija")

# ========= Proveedores de clima y límites típicos =========
PROVIDERS = {
    "open-meteo": {"label": "Open-Meteo (gratis)", "limit_days": 16, "needs_key": False},
    "visualcrossing": {"label": "Visual Crossing (clave)", "limit_days": 15, "needs_key": True},
    "weatherbit": {"label": "Weatherbit (clave)", "limit_days": 16, "needs_key": True},
}

def get_provider_from_env() -> str:
    p = (os.getenv("WEATHER_PROVIDER") or "open-meteo").strip().lower()
    return p if p in PROVIDERS else "open-meteo"

def get_provider_limit(provider: str) -> int:
    return PROVIDERS.get(provider, PROVIDERS["open-meteo"])["limit_days"]

# ================== Utilidades ==================
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split())

def item_key(cat: str, it: str) -> str:
    return f"{normalize_text(cat)}:{normalize_text(it)}".lower()

def make_key(cat: str, it: str, idx: int) -> str:
    """Key única para cada widget (texto normalizado + índice)."""
    return f"{item_key(cat, it)}::{idx}"

def dedupe_packing(packing: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Elimina duplicados por categoría (case/espacios-insensible)."""
    out = {}
    for cat, items in packing.items():
        seen = set()
        new = []
        for it in items:
            norm = normalize_text(it).lower()
            if norm in seen:
                continue
            seen.add(norm)
            new.append(it)
        out[cat] = new
    return out

def ddmmyyyy(iso_date: str) -> str:
    y, m, d = iso_date.split("-")
    return f"{d}/{m}/{y}"

# ================== Sesión HTTP + Cache ==================
@st.cache_resource
def http():
    s = requests.Session()
    s.headers.update({"User-Agent": "ValijaApp/1.1 (+https://railway.app)"})
    return s

# ================== Geocoding & clima (Open-Meteo geocoding) ==================
def geocode_city(city: str) -> Dict:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 10, "language": "es", "format": "json"}
    r = http().get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        alt = strip_accents(city)
        if alt != city:
            params["name"] = alt
            r = http().get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
    if not data.get("results"):
        raise ValueError("No se encontró la ciudad. Probá 'Ciudad, País' (ej: Mar del Plata, Argentina).")
    # elegir mejor resultado
    def name_clean(r_): return strip_accents(r_.get("name", "")).lower().strip()
    q = strip_accents(city).lower().strip()
    exact = [r_ for r_ in data["results"] if name_clean(r_) == q]
    res = exact[0] if exact else max(data["results"], key=lambda r_: r_.get("population", 0))
    return {
        "name": res.get("name"),
        "country": res.get("country"),
        "lat": res["latitude"],
        "lon": res["longitude"],
        "timezone": res.get("timezone", "auto"),
    }

@st.cache_data(ttl=60*60*6, show_spinner=False)  # 6 hs
def geocode_city_cached(city: str) -> Dict:
    return geocode_city(city)

# ================== Clima: wrappers por proveedor ==================
# --------- Open-Meteo ---------
def clamp_to_limit(start: dt.date, end: dt.date, limit_days: int) -> tuple[dt.date, dt.date]:
    today = dt.date.today()
    api_end = min(end, today + dt.timedelta(days=limit_days))
    api_start = min(start, api_end)
    return api_start, api_end

def fetch_forecast_openmeteo(lat: float, lon: float, start: dt.date, end: dt.date, timezone="auto") -> Dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
        "timezone": timezone,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    r = http().get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60*30, show_spinner=False)  # 30 min
def fetch_forecast_openmeteo_cached(lat: float, lon: float, start: dt.date, end: dt.date, timezone="auto") -> Dict:
    return fetch_forecast_openmeteo(lat, lon, start, end, timezone)

@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_forecast_openmeteo_days_cached(lat: float, lon: float, days: int, timezone="auto") -> Dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
        "timezone": timezone,
        "forecast_days": max(1, min(16, int(days))),
    }
    r = http().get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def slice_daily_by_date(forecast: Dict, start: dt.date, end: dt.date) -> Dict:
    daily = forecast.get("daily", {})
    times = daily.get("time", [])
    if not times:
        return forecast
    mask = [(start.isoformat() <= t <= end.isoformat()) for t in times]
    def _f(arr): return [v for v, keep in zip(arr, mask) if keep]
    sliced = {k: _f(v) if isinstance(v, list) and len(v) == len(times) else v for k, v in daily.items()}
    out = dict(forecast)
    out["daily"] = sliced
    return out

# --------- Visual Crossing ---------
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_forecast_visualcrossing_cached(lat: float, lon: float, start: dt.date, end: dt.date, api_key: str) -> Dict:
    base = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    # timeline/{lat},{lon}/{start}/{end}
    url = f"{base}/{lat},{lon}/{start.isoformat()}/{end.isoformat()}"
    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": api_key.strip(),
        "contentType": "json",
    }
    r = http().get(url, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    # Convertir a estructura "daily" compatible
    days = data.get("days", [])
    daily = {
        "time": [d["datetime"] for d in days],
        "temperature_2m_min": [d.get("tempmin") for d in days],
        "temperature_2m_max": [d.get("tempmax") for d in days],
        "precipitation_probability_max": [d.get("precipprob") for d in days],  # %
        "weathercode": [None for _ in days],  # no mapeamos códigos VC -> WMO
        "weatherdesc": [d.get("conditions", "—") for d in days],
    }
    return {"daily": daily}

# --------- Weatherbit ---------
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_forecast_weatherbit_cached(lat: float, lon: float, start: dt.date, end: dt.date, api_key: str) -> Dict:
    # Weatherbit pronostica hacia adelante; pedimos desde HOY hasta máximo 16 días y cortamos por [start,end]
    today = dt.date.today()
    to_date = min(end, today + dt.timedelta(days=16))
    days_needed = max(1, (to_date - today).days + 1)
    url = "https://api.weatherbit.io/v2.0/forecast/daily"
    params = {
        "lat": lat, "lon": lon, "days": min(16, days_needed),
        "units": "M", "key": api_key.strip(),
    }
    r = http().get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    arr = js.get("data", [])
    # Convertir y luego filtrar por rango deseado:
    all_daily = {
        "time": [d["valid_date"] for d in arr],
        "temperature_2m_min": [d.get("min_temp") for d in arr],
        "temperature_2m_max": [d.get("max_temp") for d in arr],
        "precipitation_probability_max": [d.get("pop") for d in arr],  # %
        "weathercode": [None for _ in arr],  # no mapeamos códigos WB -> WMO
        "weatherdesc": [d.get("weather", {}).get("description", "—") for d in arr],
    }
    tmp = {"daily": all_daily}
    return slice_daily_by_date(tmp, start, end)

# ================== Mapas y helpers de clima ==================
WEATHER_CODE_MAP = {
    0: "despejado", 1: "mayormente despejado", 2: "parcialmente nublado", 3: "nublado",
    45: "niebla", 48: "niebla escarchada", 51: "llovizna débil", 53: "llovizna", 55: "llovizna intensa",
    56: "llovizna helada débil", 57: "llovizna helada", 61: "lluvia débil", 63: "lluvia", 65: "lluvia fuerte",
    66: "lluvia helada débil", 67: "lluvia helada", 71: "nieve débil", 73: "nieve", 75: "nieve fuerte",
    77: "granos de nieve", 80: "chubascos débiles", 81: "chubascos", 82: "chubascos fuertes",
    85: "chubascos de nieve débiles", 86: "chubascos de nieve fuertes",
    95: "tormenta", 96: "tormenta con granizo débil", 99: "tormenta con granizo fuerte",
}

def summarize_weather(daily: Dict) -> str:
    times = daily.get("time", [])
    if not times:
        return "No hay datos de pronóstico disponibles por fuera del rango público."
    lines = []
    for i, date in enumerate(times):
        tmin = daily["temperature_2m_min"][i]
        tmax = daily["temperature_2m_max"][i]
        pprec = daily.get("precipitation_probability_max", [None]*len(times))[i]
        # Descripción: preferir weatherdesc si existe
        desc = daily.get("weatherdesc", [])
        if desc:
            dsc = desc[i] or "—"
        else:
            code = daily.get("weathercode", [None]*len(times))[i]
            dsc = WEATHER_CODE_MAP.get(code, "—")
        lines.append(f"{ddmmyyyy(date)}: {dsc}, {tmin:.0f}–{tmax:.0f}°C, precip {pprec if pprec is not None else '—'}%")
    avg_min = sum(v for v in daily["temperature_2m_min"] if v is not None) / len([v for v in daily["temperature_2m_min"] if v is not None])
    avg_max = sum(v for v in daily["temperature_2m_max"] if v is not None) / len([v for v in daily["temperature_2m_max"] if v is not None])
    wet_days = sum(1 for p in daily.get("precipitation_probability_max", []) if p is not None and p >= 40)
    header = f"Promedio térmico: mín {avg_min:.1f}°C / máx {avg_max:.1f}°C. Días con alta chance de lluvia (≥40%): {wet_days}."
    return header + "\n" + "\n".join(lines)

def forecast_to_df_full_range(daily: Dict, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Construye el DF para TODO el rango del viaje. Completa 'sin datos' donde no hay pronóstico."""
    date_list = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    avail = {}
    times = daily.get("time", [])
    for i, iso in enumerate(times):
        avail[iso] = {
            "min": daily["temperature_2m_min"][i],
            "max": daily["temperature_2m_max"][i],
            "pp": daily.get("precipitation_probability_max", [None]*len(times))[i],
            "code": daily.get("weathercode", [None]*len(times))[i] if daily.get("weathercode") else None,
            "desc": daily.get("weatherdesc", [None]*len(times))[i] if daily.get("weatherdesc") else None,
        }

    rows = []
    for d in date_list:
        iso = d.isoformat()
        if iso in avail:
            tmin = avail[iso]["min"]
            tmax = avail[iso]["max"]
            pp = avail[iso]["pp"]
            code = avail[iso]["code"]
            desc = avail[iso]["desc"] or WEATHER_CODE_MAP.get(code, "—")
            umbrella = "🌧️" if (pp is not None and isinstance(pp, (int, float)) and pp >= 40) else ""
            rows.append({
                "Fecha": ddmmyyyy(iso),
                "Mín (°C)": round(tmin) if isinstance(tmin, (int, float)) else "—",
                "Máx (°C)": round(tmax) if isinstance(tmax, (int, float)) else "—",
                "Lluvia %": pp if pp is not None else "",
                "🌧️": umbrella,
                "Estado": desc,
            })
        else:
            rows.append({
                "Fecha": ddmmyyyy(iso),
                "Mín (°C)": "—",
                "Máx (°C)": "—",
                "Lluvia %": "",
                "🌧️": "",
                "Estado": "(sin datos)",
            })
    df = pd.DataFrame(rows)
    return df[["Fecha", "Mín (°C)", "Máx (°C)", "Lluvia %", "🌧️", "Estado"]]

# ================== Lógica local (reglas más finas) ==================
def rule_based_packing(
    avg_min: float, avg_max: float, wet_days: int, activities: List[str], days: int,
    *, carry_on: bool=False, laundry: bool=False, detail_level: int=3, profiles: List[str]=None
) -> Dict[str, List[str]]:
    profiles = profiles or []
    acts = [a.lower() for a in activities]

    def q(n):
        factor = 1.0
        if laundry: factor *= 0.65
        if carry_on: factor *= 0.85
        return max(1, int(round(n * factor)))

    base = {
        "Ropa": [
            f"Remeras x{max(3, q(days*0.8))}",
            f"Pantalones x{max(2, q(days/3))}",
            f"Ropa interior x{max(5, q(days*1.0))}",
            f"Medias x{max(5, q(days*1.0))}",
            f"Ropa de dormir x{max(1, q(days/5))}" if detail_level >= 3 else "Ropa de dormir x1",
        ],
        "Calzado": ["Zapatillas cómodas"],
        "Higiene y salud": ["Cepillo + pasta (viaje)", "Desodorante", "Medicación personal", "Protector solar (100 ml)"],
        "Tecnología": ["Celular + cargador", "Power bank", "Adaptador (si aplica)"],
        "Documentación": ["DNI/Pasaporte", "Tarjeta de crédito/débito", "Reserva/Seguro"],
        "Varios": ["Botella reutilizable", "Gafas de sol", "Mochila de día"],
    }

    very_hot = avg_max >= 30
    hot = 27 <= avg_max < 30
    cold = avg_min <= 5
    very_cold = avg_min <= -2

    if very_hot:
        base["Ropa"] += [f"Traje de baño x{max(1, q(days/5))}", f"Shorts/Pollera x{max(1, q(days/3))}", "Gorra/sombrero"]
        base["Calzado"] += ["Ojotas"]
        base["Varios"] += ["Toalla de playa", "After sun", "Repelente"]
    elif hot:
        base["Ropa"] += [f"Prendas livianas x{max(1, q(days/4))}", "Campera liviana"]
        base["Calzado"] += ["Zapatillas ventiladas"]
    elif very_cold:
        base["Ropa"] += [
            "Campera técnica (abrigo)",
            f"Remera térmica x{max(2, q(days/4))}",
            f"Pantalón térmico x{max(1, q(days/4))}",
            f"Buzo/Suéter x{max(2, q(days/5))}",
            "Gorro/Bufanda/Guantes",
        ]
        base["Calzado"] += ["Calzado cerrado abrigado"]
        base["Varios"] += ["Crema labial", "Termo (opcional)"]
    elif cold:
        base["Ropa"] += [
            "Campera de abrigo",
            f"Remera térmica x{max(1, q(days/5))}",
            f"Buzo/Suéter x{max(2, q(days/6))}",
            "Gorro/Guantes",
        ]
        base["Calzado"] += ["Zapatillas cerradas"]
    else:
        base["Ropa"] += ["Campera liviana", f"Buzo/Suéter x{max(1, q(days/6))}"]

    if wet_days >= 2:
        base["Varios"] += ["Pilotín/poncho", "Paraguas plegable", "Cubre mochila"]
        base["Calzado"] += ["Calzado que seque rápido"]

    # Actividades
    if any(x in a for a in acts for x in ("trek", "sender", "montaña")):
        base["Ropa"] += ["Campera impermeable respirable"]
        base["Calzado"] += ["Zapatillas/boots de trekking"]
        base["Varios"] += ["Bastones (opcional)", "Botiquín básico", "Repelente"]
    if any(x in a for a in acts for x in ("noche", "resto", "eleg")):
        base["Ropa"] += ["1 outfit arreglado"]
        base["Calzado"] += ["Zapatos/zapatillas urbanas limpias"]
    if any("playa" in a for a in acts):
        base["Ropa"] += [f"Traje de baño x{max(1, q(max(1, days//5)))}"]
        base["Calzado"] += ["Ojotas"]
        base["Varios"] += ["Toalla de playa"]

    # Perfiles
    if "con niños" in profiles:
        base["Varios"] += ["Toallitas húmedas", "Snacks", "Entretenimiento (libros/juguetes)"]
    if "negocios" in profiles:
        base["Tecnología"] += ["Notebook + cargador", "Adaptadores/HDMI"]
        base["Ropa"] += ["Camisa/Blusa formal", "Saco/Blazer"]

    # Detalle
    if detail_level >= 4:
        base["Varios"] += ["Cinta para ampollas", "Cortaúñas", "Mini costurero"]
    if detail_level >= 5:
        base["Varios"] += ["Cubes/bolsas de compresión", "Balanza de equipaje"]

    if carry_on:
        base["Varios"] += ["Neceser 1 L (líquidos <100 ml)"]

    return base

# ================== Normalizador de cantidades + extras ==================
def ensure_quantities_and_extras(
    packing: Dict[str, List[str]],
    *,
    days: int,
    laundry: bool,
    carry_on: bool,
    avg_min: float,
    avg_max: float,
    profiles: List[str],
    country: str
) -> Dict[str, List[str]]:

    def add_unique(cat, item):
        if item not in packing.setdefault(cat, []):
            packing[cat].append(item)

    factor = 1.0
    if laundry: factor *= 0.65
    if carry_on: factor *= 0.85

    underwear = max(1, round(days * factor))
    shirts    = max(3, round(days * 0.8 * factor))
    socks     = max(3, round(days * factor))
    pants     = max(2, round(days / 3 * factor))

    normalized: Dict[str, List[str]] = {}
    for cat, items in packing.items():
        new_items = []
        for it in items:
            t = normalize_text(it).lower()
            if "interior" in t or "underwear" in t or "calzonc" in t:
                it = f"Ropa interior x{underwear}"
            elif "remera" in t or "camis" in t or "t-shirt" in t:
                it = f"Remeras x{shirts}"
            elif "media" in t or "calcet" in t or "socks" in t:
                it = f"Medias x{socks}"
            elif "pantal" in t:
                it = f"Pantalones x{pants}"
            new_items.append(it)
        normalized[cat] = new_items
    packing = normalized

    # Asegurar ropa interior
    if not any("interior" in i.lower() for i in packing.get("Ropa", [])):
        add_unique("Ropa", f"Ropa interior x{underwear}")

    # Extras por perfil
    if "con niños" in profiles:
        add_unique("Varios", "Toallitas húmedas")
        add_unique("Varios", "Snacks para el viaje")
        add_unique("Varios", "Entretenimiento para niños")
    if "negocios" in profiles:
        add_unique("Ropa", "Camisa/Blusa formal")
        add_unique("Ropa", "Saco/Blazer")
        add_unique("Tecnología", "Notebook + cargador")

    # Documentación local para Argentina
    if country and country.lower() in ["argentina", "arg"]:
        if "Documentación" in packing:
            packing["Documentación"] = [it for it in packing["Documentación"] if "pasaporte" not in it.lower()]
            add_unique("Documentación", "DNI")

    return packing

# ================== OpenAI (opcional) ==================
def generate_packing_with_openai(weather_brief: str, city: str, days: int, activities: List[str], options: dict) -> Dict[str, List[str]]:
    raw_key = os.getenv("OPENAI_API_KEY", "")
    api_key = raw_key.strip().splitlines()[0] if raw_key else ""
    if not api_key or not api_key.startswith(("sk-", "rk-")):
        raise RuntimeError("OPENAI_API_KEY no configurada.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
You are a travel packing assistant. Return ONLY a valid JSON with exact keys:
["Ropa","Calzado","Higiene y salud","Tecnología","Documentación","Varios"].

Trip: {days} days in {city}.
Real-weather summary:
{weather_brief}

Activities: {', '.join(activities) if activities else 'not specified'}.
Traveler profiles: {', '.join(options.get('profiles', [])) or 'standard tourist'}.
Detail level: {options.get('detail_level', 3)} (1–5).
Carry-on: {"yes" if options.get("carry_on") else "no"}.
Laundry: {"yes" if options.get("laundry") else "no"}.
Language: {options.get("language", "es")}.

Rules:
- Scale QUANTITIES by days/laundry (e.g., "Remeras x7", "Medias x10", "Ropa interior x10").
- Reflect temperature/rain (thermals for cold, swimwear for hot, rain gear if wet days ≥ 2).
- Prefer compact/light items if carry-on is yes.
- Output in requested language. No text outside JSON.
"""

    messages = [
        {"role": "system", "content": "Eres un asistente de viajes experto. Devuelve siempre JSON válido con cantidades."},
        {"role": "user", "content": prompt},
    ]

    def try_parse_json(text: str) -> Dict[str, list]:
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{"); end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
            raise

    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.3,
        response_format={"type": "json_object"},
    )
    return try_parse_json(resp.choices[0].message.content.strip())

def export_txt(packing: Dict[str, List[str]]) -> str:
    lines = []
    for cat, items in packing.items():
        lines.append(f"== {cat} ==")
        for it in items:
            lines.append(f"- {it}")
        lines.append("")
    return "\n".join(lines)

# ================== UI ==================
st.title("🧳 Asistente de Valija con Clima Real")
st.caption("Destino + fechas → clima → lista de equipaje. Hecho por Juanma 😉")

with st.sidebar:
    st.header("⚙️ Preferencias")
    # Proveedor de clima
    providers_labels = [PROVIDERS[k]["label"] for k in PROVIDERS]
    providers_keys = list(PROVIDERS.keys())
    default_key = get_provider_from_env()
    provider_idx = providers_keys.index(default_key)
    provider_choice = st.selectbox("Proveedor de clima", providers_labels, index=provider_idx)
    provider = providers_keys[providers_labels.index(provider_choice)]
    # Opciones
    detail_level = st.slider("Nivel de detalle", 1, 5, 3)
    traveler_profiles = st.multiselect("Perfil del viaje", ["solo", "en pareja", "con niños", "negocios", "mochilero"], default=[])
    carry_on = st.toggle("Optimizar equipaje de mano")
    laundry = st.toggle("Tendré lavadora")
    language = st.selectbox("Idioma de la lista", ["es", "en"], index=0)
    gen_mode = st.selectbox("Generación", ["Automático", "Forzar local", "Forzar OpenAI"], index=0)
    apply_sidebar = st.button("🔄 Aplicar cambios")

with st.form("trip_form"):
    city = st.text_input("¿A dónde viajás?", placeholder="Ej: Mar del Plata, Argentina")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Fecha de inicio", value=dt.date.today() + dt.timedelta(days=7), format="DD/MM/YYYY")
    with c2:
        end_date = st.date_input("Fecha de regreso", value=dt.date.today() + dt.timedelta(days=14), format="DD/MM/YYYY")

    # Vista previa de cantidad de días
    days_preview = (end_date - start_date).days + 1 if end_date >= start_date else 0
    st.caption(f"Días seleccionados: **{days_preview}**" if days_preview > 0 else "Elegí un rango válido (fin ≥ inicio).")

    # Aviso según límite del proveedor elegido
    limit_date = dt.date.today() + dt.timedelta(days=get_provider_limit(provider))
    if end_date > limit_date:
        st.caption(
            f"⚠️ El pronóstico de {PROVIDERS[provider]['label'].split(' (')[0]} cubre hasta "
            f"{get_provider_limit(provider)} días desde hoy ({limit_date.strftime('%d/%m/%Y')}). "
            "Se mostrará clima hasta esa fecha; el resto de los días del viaje se verán como “(sin datos)”. "
            "Las cantidades se calcularán para todos tus días."
        )

    # Actividades
    ACTIVITY_PRESETS = ["playa", "trekking", "salidas nocturnas", "nieve", "montaña"]
    acts_selected = st.multiselect("Actividades (elegí de la lista)", ACTIVITY_PRESETS, default=[])
    acts_extra = st.text_input("…o agregá otras (separá por coma)", placeholder="city tour, snorkel, eventos")
    activities = ", ".join(acts_selected + ([acts_extra] if acts_extra else []))

    submit = st.form_submit_button("Generar")

# ================== Orquestación ==================
def compute_and_store(city, start_date, end_date, activities, detail_level, traveler_profiles, carry_on, laundry, language, gen_mode, provider):
    log.info(f"compute_and_store city={city} {start_date}→{end_date} days={ (end_date - start_date).days + 1 } mode={gen_mode} provider={provider}")
    geo = geocode_city_cached(city)

    limit_days = get_provider_limit(provider)
    api_start, api_end = clamp_to_limit(start_date, end_date, limit_days)

    # Elegir proveedor
    daily = {}
    available_start = available_end = None
    try:
        if provider == "open-meteo":
            try:
                resp = fetch_forecast_openmeteo_cached(geo["lat"], geo["lon"], api_start, api_end, "auto")
            except requests.HTTPError as e:
                if getattr(e, "response", None) is not None and e.response.status_code == 400:
                    days_needed = (min(api_end, dt.date.today() + dt.timedelta(days=limit_days)) - dt.date.today()).days + 1
                    alt = fetch_forecast_openmeteo_days_cached(geo["lat"], geo["lon"], days_needed, "auto")
                    resp = slice_daily_by_date(alt, api_start, api_end)
                else:
                    raise
            daily = resp.get("daily", {})

        elif provider == "visualcrossing":
            key = os.getenv("VISUALCROSSING_KEY", "").strip()
            if not key:
                st.info("Falta VISUALCROSSING_KEY; usando Open-Meteo.", icon="ℹ️")
                resp = fetch_forecast_openmeteo_cached(geo["lat"], geo["lon"], api_start, api_end, "auto")
                daily = resp.get("daily", {})
            else:
                resp = fetch_forecast_visualcrossing_cached(geo["lat"], geo["lon"], api_start, api_end, key)
                daily = resp.get("daily", {})

        elif provider == "weatherbit":
            key = os.getenv("WEATHERBIT_KEY", "").strip()
            if not key:
                st.info("Falta WEATHERBIT_KEY; usando Open-Meteo.", icon="ℹ️")
                resp = fetch_forecast_openmeteo_cached(geo["lat"], geo["lon"], api_start, api_end, "auto")
                daily = resp.get("daily", {})
            else:
                resp = fetch_forecast_weatherbit_cached(geo["lat"], geo["lon"], api_start, api_end, key)
                daily = resp.get("daily", {})

        times = daily.get("time", [])
        if times:
            available_start = dt.date.fromisoformat(times[0])
            available_end = dt.date.fromisoformat(times[-1])
    except Exception:
        log.exception("Fallo obteniendo pronóstico; continuamos sin datos.")
        daily = {"time": []}

    # Métricas climáticas para reglas
    times = daily.get("time", [])
    if times:
        tmins = [v for v in daily["temperature_2m_min"] if isinstance(v, (int, float))]
        tmaxs = [v for v in daily["temperature_2m_max"] if isinstance(v, (int, float))]
        avg_min = sum(tmins) / len(tmins) if tmins else 18.0
        avg_max = sum(tmaxs) / len(tmaxs) if tmaxs else 24.0
        wet_days = sum(1 for p in daily.get("precipitation_probability_max", []) if isinstance(p, (int, float)) and p >= 40)
    else:
        avg_min, avg_max, wet_days = 18.0, 24.0, 0

    acts = [normalize_text(a) for a in activities.split(",")] if activities else []
    trip_days = (end_date - start_date).days + 1

    opts = {"detail_level": detail_level, "profiles": traveler_profiles, "carry_on": carry_on, "laundry": laundry, "language": language}

    use_openai = (gen_mode == "Forzar OpenAI") or (gen_mode == "Automático" and os.getenv("OPENAI_API_KEY"))

    if use_openai and gen_mode != "Forzar local":
        try:
            weather_brief = summarize_weather(daily) if times else "No hay datos de pronóstico disponibles."
            packing = generate_packing_with_openai(weather_brief, geo["name"], trip_days, acts, opts)
            st.success("Lista generada con OpenAI ✅")
        except Exception as e:
            log.exception("OpenAI falló; usando reglas locales")
            st.info("Usando lógica local (sin OpenAI) ✅")
            st.caption(f"Motivo: {e}")
            packing = rule_based_packing(avg_min, avg_max, wet_days, acts, trip_days,
                                         carry_on=carry_on, laundry=laundry, detail_level=detail_level, profiles=traveler_profiles)
    else:
        st.info("Generación local (reglas) ✅")
        packing = rule_based_packing(avg_min, avg_max, wet_days, acts, trip_days,
                                     carry_on=carry_on, laundry=laundry, detail_level=detail_level, profiles=traveler_profiles)

    # Ajustes SIEMPRE: cantidades + extras + doc local + dedupe
    packing = ensure_quantities_and_extras(
        packing,
        days=trip_days, laundry=laundry, carry_on=carry_on,
        avg_min=avg_min, avg_max=avg_max,
        profiles=traveler_profiles, country=geo.get("country", "")
    )
    packing = dedupe_packing(packing)

    st.session_state["packing"] = packing
    st.session_state["meta"] = {
        "geo": geo,
        "avg_min": avg_min, "avg_max": avg_max, "wet_days": wet_days,
        "trip_days": trip_days, "acts": acts, "opts": opts, "date_range": (start_date, end_date),
        "activities_raw": activities or "", "city_raw": city, "daily": daily,
        "api_date_range": (api_start, api_end),
        "available_range": (available_start, available_end),
        "forecast_limited": (not times) or (available_start and available_start > start_date) or (available_end and available_end < end_date),
        "provider": provider,
    }

# ============== Acciones de usuario (submit / aplicar) ==================
if submit:
    if not city or start_date > end_date:
        st.error("Completá la ciudad y verificá el rango de fechas.")
    else:
        try:
            compute_and_store(city, start_date, end_date, activities, detail_level, traveler_profiles, carry_on, laundry, language, gen_mode, provider)
            st.success("Preferencias aplicadas ✅")
        except Exception as e:
            log.exception("Fallo en compute_and_store")
            st.error(f"Ocurrió un error: {e}")

if apply_sidebar and ("packing" in st.session_state) and ("meta" in st.session_state):
    m = st.session_state["meta"]
    try:
        compute_and_store(m["city_raw"], m["date_range"][0], m["date_range"][1], m["activities_raw"],
                          detail_level, traveler_profiles, carry_on, laundry, language, gen_mode, provider)
        st.success("Preferencias aplicadas ✅")
    except Exception as e:
        log.exception("Fallo al aplicar cambios")
        st.error(f"Ocurrió un error al aplicar cambios: {e}")

# ================== Render principal ==================
if "packing" in st.session_state and st.session_state["packing"]:
    packing = st.session_state["packing"]
    m = st.session_state["meta"]

    st.subheader(f"📍 {m['geo']['name']}, {m['geo']['country']}")
    prov_name = PROVIDERS[m.get("provider", "open-meteo")]["label"].split(" (")[0]
    st.markdown(
        f"**Fechas:** {m['date_range'][0].strftime('%d/%m/%Y')} → {m['date_range'][1].strftime('%d/%m/%Y')} "
        f"&nbsp;•&nbsp; **Días:** {m['trip_days']} &nbsp;•&nbsp; **Clima:** {prov_name}"
    )

    # Aviso si el pronóstico fue recortado o faltan días (sin datos)
    if m.get("forecast_limited"):
        avail_end = m.get("available_range", (None, None))[1]
        if avail_end:
            limit_str = avail_end.strftime('%d/%m/%Y')
        else:
            limit_str = (dt.date.today() + dt.timedelta(days=get_provider_limit(m.get("provider","open-meteo")))).strftime('%d/%m/%Y')
        st.warning(
            f"🔎 El pronóstico muestra datos hasta **{limit_str}**. "
            "Para el resto de los días se verá **(sin datos)** en la tabla, "
            f"pero la lista y las cantidades se generan igualmente para **{m['trip_days']}** días.",
            icon="ℹ️",
        )

    # Tabla con TODO el rango del viaje; completa '(sin datos)' más allá del límite
    df_forecast = forecast_to_df_full_range(m["daily"], m["date_range"][0], m["date_range"][1])
    st.dataframe(df_forecast, hide_index=True, use_container_width=True)

    # ===== Acciones en bloque =====
    pending = st.session_state.pop("__bulk_action__", None)
    if pending:
        action = pending.get("type")
        if action in {"mark_all", "unmark_all", "reset"}:
            for cat, items in packing.items():
                for idx, it in enumerate(items):
                    k = make_key(cat, it, idx)
                    if action == "mark_all":
                        st.session_state[k] = True
                    elif action == "unmark_all":
                        st.session_state[k] = False
                    elif action == "reset":
                        st.session_state.pop(k, None)

    # Checkboxes por categoría
    st.subheader("✅ Lista sugerida para la valija")
    for cat, items in packing.items():
        with st.expander(cat, expanded=True):
            for idx, it in enumerate(items):
                k = make_key(cat, it, idx)
                if k not in st.session_state:
                    st.session_state[k] = False
                st.checkbox(normalize_text(it), key=k)

    # Progreso
    total = sum(len(v) for v in packing.values())
    done = sum(
        1 for cat, items in packing.items() for idx, it in enumerate(items)
        if st.session_state.get(make_key(cat, it, idx), False)
    )
    pct = (done / total) if total else 0
    st.write(f"Progreso: **{done}/{total}** ({round(pct*100)}%)")
    st.progress(pct)

    # Botones Marcar/Desmarcar/Reset
    cma, cmb, cmc = st.columns(3)
    with cma:
        if st.button("Marcar todo"):
            st.session_state["__bulk_action__"] = {"type": "mark_all"}
            st.rerun()
    with cmb:
        if st.button("Desmarcar todo"):
            st.session_state["__bulk_action__"] = {"type": "unmark_all"}
            st.rerun()
    with cmc:
        if st.button("Resetear selección"):
            st.session_state["__bulk_action__"] = {"type": "reset"}
            st.rerun()

    # Agregar ítem manual
    with st.form("add_item_form", clear_on_submit=True):
        new_item = st.text_input("Agregar ítem manual", placeholder="Ej: Adaptador USB-C")
        add_submit = st.form_submit_button("Añadir")
    if add_submit and new_item:
        new_item_n = normalize_text(new_item)
        already = any(normalize_text(x).lower() == new_item_n.lower() for x in packing.setdefault("Varios", []))
        if not already:
            packing["Varios"].append(new_item_n)
            packing = dedupe_packing(packing)
            st.session_state["packing"] = packing
            st.success(f"Añadido: {new_item_n}")
            st.rerun()
        else:
            st.info("Ese ítem ya existe en 'Varios'.")

    # Filtro
    query = st.text_input("🔎 Filtrar ítems", placeholder="paraguas, térmica, cargador…")
    if query:
        q = normalize_text(query).lower()
        filtered = {cat: [it for it in items if q in normalize_text(it).lower()] for cat, items in packing.items()}
        st.write("Resultados del filtro:")
        for cat, items in filtered.items():
            if items:
                with st.expander(f"{cat} (filtrado)", expanded=True):
                    for it in items:
                        st.write(f"- {it}")

    # Exportar con metadatos
    header = (
        f"Destino: {m['geo']['name']}, {m['geo']['country']}\n"
        f"Fechas: {m['date_range'][0].strftime('%d/%m/%Y')} -> {m['date_range'][1].strftime('%d/%m/%Y')}\n"
        f"Días: {m['trip_days']} | Perfiles: {', '.join(m['opts']['profiles']) or '-'} | "
        f"Carry-on: {m['opts']['carry_on']} | Laundry: {m['opts']['laundry']}\n\n"
    )
    txt = header + export_txt(packing)
    st.download_button("⬇️ Descargar lista .txt", data=txt, file_name="lista_valija.txt", mime="text/plain")

    rows = []
    for cat, items in packing.items():
        for idx, it in enumerate(items):
            rows.append({
                "Destino": f"{m['geo']['name']}, {m['geo']['country']}",
                "Desde": m["date_range"][0].strftime("%d/%m/%Y"),
                "Hasta": m["date_range"][1].strftime("%d/%m/%Y"),
                "Dias": m["trip_days"],
                "Categoria": cat,
                "Item": normalize_text(it),
                "Empacado": st.session_state.get(make_key(cat, it, idx), False)
            })
    df = pd.DataFrame(rows)
    st.download_button(
        "⬇️ Descargar lista .csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="lista_valija.csv",
        mime="text/csv"
    )

    # Regenerar (con enfoque)
    c1, c2, c3 = st.columns(3)
    with c1:
        regen_focus = st.selectbox("Enfoque de regeneración", ["Más detalle", "Minimalista (carry-on)", "Enfocar actividades"], index=0)
    with c2:
        regen_btn = st.button("🔁 Regenerar")
    with c3:
        st.caption("Sube/baja detalle y ajusta reglas según enfoque.")

    if regen_btn:
        try:
            if regen_focus == "Más detalle":
                m["opts"]["detail_level"] = min(5, m["opts"]["detail_level"] + 1)
            elif regen_focus == "Minimalista (carry-on)":
                m["opts"]["carry_on"] = True
                m["opts"]["detail_level"] = max(2, m["opts"]["detail_level"] - 1)
            elif regen_focus == "Enfocar actividades" and m["acts"]:
                acts_boosted = m["acts"] + m["acts"]
                m["activities_raw"] = ", ".join(acts_boosted)

            compute_and_store(
                m["city_raw"], m["date_range"][0], m["date_range"][1],
                m["activities_raw"], m["opts"]["detail_level"], m["opts"]["profiles"],
                m["opts"]["carry_on"], m["opts"]["laundry"], m["opts"]["language"], gen_mode, provider
            )
            st.success("Lista regenerada ✅")
        except Exception as e:
            log.exception("Fallo al regenerar")
            st.info("Usando lógica local (sin OpenAI) ✅")
            st.caption(f"Motivo: {e}")
