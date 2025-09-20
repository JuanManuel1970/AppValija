import os
import json
import unicodedata
import datetime as dt
from typing import List, Dict
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

# ===== Config inicial de Streamlit (primera instrucción) =====
st.set_page_config(page_title="Asistente de armado de Valija", page_icon="🧳", layout="centered")

# ===== Estado global =====
if "packing" not in st.session_state:
    st.session_state.packing = None
if "result_meta" not in st.session_state:
    st.session_state.result_meta = {}
if "checked" not in st.session_state:
    st.session_state.checked = {}

# ================== Utilidades ==================
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_text(s: str) -> str:
    # quita espacios de extremos y colapsa espacios internos
    return " ".join((s or "").strip().split())

def item_key(cat: str, it: str) -> str:
    # clave normalizada y en minúscula para evitar desalineaciones
    return f"{normalize_text(cat)}:{normalize_text(it)}".lower()

def choose_best_place(query: str, results: list) -> Dict:
    """
    Elige el mejor resultado de geocoding:
    1) match exacto por nombre (sin acentos)
    2) nombre que contenga el query
    3) mayor población
    """
    if not results:
        return {}
    q_clean = strip_accents(query).lower().strip()

    def name_clean(r):
        return strip_accents(r.get("name", "")).lower().strip()

    exact = [r for r in results if name_clean(r) == q_clean]
    if exact:
        return exact[0]

    contains = [r for r in results if q_clean in name_clean(r)]
    if contains:
        return contains[0]

    return max(results, key=lambda r: r.get("population", 0))

# ================== Geocoding y clima ==================
def geocode_city(city: str) -> Dict:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 5, "language": "es", "format": "json"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        # Intentar sin acentos como fallback
        alt = strip_accents(city)
        if alt != city:
            params["name"] = alt
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
    if not data.get("results"):
        raise ValueError("No se encontró la ciudad. Probá con 'Ciudad, País' (ej: Neuquén, Argentina).")
    res = choose_best_place(city, data["results"])
    return {
        "name": res.get("name"),
        "country": res.get("country"),
        "lat": res["latitude"],
        "lon": res["longitude"],
        "timezone": res.get("timezone", "auto"),
    }

def fetch_forecast(lat: float, lon: float, start: dt.date, end: dt.date, timezone="auto") -> Dict:
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
    0: "despejado", 1: "mayormente despejado", 2: "parcialmente nublado", 3: "nublado",
    45: "niebla", 48: "niebla escarchada", 51: "llovizna débil", 53: "llovizna", 55: "llovizna intensa",
    56: "llovizna helada débil", 57: "llovizna helada", 61: "lluvia débil", 63: "lluvia", 65: "lluvia fuerte",
    66: "lluvia helada débil", 67: "lluvia helada", 71: "nieve débil", 73: "nieve", 75: "nieve fuerte",
    77: "granos de nieve", 80: "chubascos débiles", 81: "chubascos", 82: "chubascos fuertes",
    85: "chubascos de nieve débiles", 86: "chubascos de nieve fuertes",
    95: "tormenta", 96: "tormenta con granizo débil", 99: "tormenta con granizo fuerte",
}

def summarize_weather(daily: Dict) -> str:
    lines = []
    for i, date in enumerate(daily["time"]):
        tmin = daily["temperature_2m_min"][i]
        tmax = daily["temperature_2m_max"][i]
        pprec = daily.get("precipitation_probability_max", [None]*len(daily["time"]))[i]
        code = daily.get("weathercode", [None]*len(daily["time"]))[i]
        desc = WEATHER_CODE_MAP.get(code, "condiciones variables")
        lines.append(f"{date}: {desc}, {tmin:.0f}–{tmax:.0f}°C, prob. precip {pprec if pprec is not None else '–'}%")

    avg_min = sum(daily["temperature_2m_min"]) / len(daily["temperature_2m_min"])
    avg_max = sum(daily["temperature_2m_max"]) / len(daily["temperature_2m_max"])
    wet_days = sum(1 for p in daily.get("precipitation_probability_max", []) if p is not None and p >= 40)
    header = f"Promedio térmico: mín {avg_min:.1f}°C / máx {avg_max:.1f}°C. Días con alta chance de lluvia (≥40%): {wet_days}."
    return header + "\n" + "\n".join(lines)

# ================== Lógica local (fallback) ==================
def rule_based_packing(avg_min: float, avg_max: float, wet_days: int, activities: List[str], days: int) -> Dict[str, List[str]]:
    shirts = max(5, round(days * 0.8))
    pants = max(2, round(days / 3))
    underwear = max(7, days)
    socks = max(7, days)

    packing = {
        "Ropa": [f"Remeras x{shirts}", f"Pantalones x{pants}", f"Ropa interior x{underwear}", f"Medias x{socks}"],
        "Calzado": ["Zapatillas cómodas"],
        "Higiene y salud": ["Cepillo/pasta", "Desodorante", "Medicación personal", "Protector solar"],
        "Tecnología": ["Celular + cargador", "Power bank", "Adaptador de enchufe (si aplica)"],
        "Documentación": ["DNI/Pasaporte", "Tarjeta de crédito/débito", "Reserva/Seguro"],
        "Varios": ["Botella reutilizable", "Gafas de sol", "Mochila de día"],
    }

    if avg_max >= 27:
        packing["Ropa"] += ["Traje de baño", "Gorra/sombrero"]
        packing["Calzado"] += ["Ojotas"]
        packing["Varios"] += ["Toalla de playa", "After sun"]
    elif avg_min <= 8:
        packing["Ropa"] += ["Campera de abrigo", "Buzo/suéter x2", "Remera térmica x2", "Pantalón térmico (opcional)", "Gorro/Buff/Guantes"]
        packing["Calzado"] += ["Zapatillas cerradas"]
    else:
        packing["Ropa"] += ["Campera liviana", "Buzo/suéter"]

    if wet_days >= 1:
        packing["Varios"] += ["Pilotín/poncho", "Paraguas plegable", "Cubre mochila"]
        packing["Calzado"] += ["Calzado que seque rápido"]

    acts = [a.lower() for a in activities]
    if any("trek" in a or "sender" in a or "montaña" in a for a in acts):
        packing["Ropa"] += ["Campera impermeable respirable"]
        packing["Calzado"] += ["Zapatillas/boots de trekking"]
        packing["Varios"] += ["Bastones (opcional)", "Botiquín básico", "Repelente"]
    if any("noche" in a or "resto" in a or "eleg" in a for a in acts):
        packing["Ropa"] += ["1 outfit arreglado"]
        packing["Calzado"] += ["Zapatos/zapatillas urbanas limpias"]

    return packing

# ================== OpenAI (IA) ==================
def generate_packing_with_openai(weather_brief: str, city: str, days: int, activities: List[str], options: dict) -> Dict[str, List[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no configurada.")

    import openai
    client = openai.OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
You are a travel packing assistant. Produce a packing list ONLY as a valid JSON object.

Trip: {days} days in {city}.
Real-weather summary:
{weather_brief}

Activities: {', '.join(activities) if activities else 'not specified'}.
Traveler profiles: {', '.join(options.get('profiles', [])) or 'standard tourist'}.
Constraints:
- Detail level (1-5): {options.get('detail_level', 3)}
- Carry-on optimization: {"yes" if options.get("carry_on") else "no"}
- Laundry available: {"yes" if options.get("laundry") else "no"}
- Language: {options.get("language", "es")}

Rules:
- Return JSON with EXACT KEYS: ["Ropa", "Calzado", "Higiene y salud", "Tecnología", "Documentación", "Varios"].
- The number of items must scale with "detail level". At 5, include sub-items and spares.
- If carry-on is YES, reduce bulky items and prefer lightweight/quick-dry; suggest packing cubes.
- If laundry is YES, reduce quantities accordingly.
- If kids, add kid-specific items (snacks, wipes, entertainment).
- If business, include formal outfit and laptop accessories.
- If backpacker, prioritize lightweight, multi-use items.
- Output MUST be in requested language ('es' or 'en').
- DO NOT include any text outside JSON. No comments.

Example shape:
{{
  "Ropa": ["Remeras x5", "Pantalones x2"],
  "Calzado": ["Zapatillas cómodas"],
  "Higiene y salud": ["Cepillo de dientes", "Desodorante"],
  "Tecnología": ["Celular + cargador"],
  "Documentación": ["DNI/Pasaporte"],
  "Varios": ["Botella de agua"]
}}
"""

    messages = [
        {"role": "system", "content": "Eres un asistente de viajes experto en equipaje. Devuelve SIEMPRE un JSON válido y bien formado."},
        {"role": "user", "content": prompt},
    ]

    def try_parse_json(text: str) -> Dict[str, list]:
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end+1]
                return json.loads(candidate)
            raise

    # 1) Intento con response_format
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content.strip()
        data = try_parse_json(content)
        return data
    except Exception as e1:
        # 2) Reintento sin response_format
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
            )
            content = resp.choices[0].message.content.strip()
            data = try_parse_json(content)
            return data
        except Exception as e2:
            raise RuntimeError(f"Error OpenAI: {e2}") from e1

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
st.caption("Destino + fechas → clima (Open-Meteo) → lista de equipaje. Hecho por Juanma 😉")

# Sidebar de preferencias (con botón para aplicar cambios y regenerar)
with st.sidebar:
    st.header("⚙️ Preferencias")
    detail_level = st.slider("Nivel de detalle", 1, 5, 3, help="Más alto = más ítems y subdetalles")
    traveler_profiles = st.multiselect(
        "Perfil del viaje",
        ["solo", "en pareja", "con niños", "negocios", "mochilero"],
        default=[]
    )
    carry_on = st.toggle("Optimizar para equipaje de mano (10 kg máx.)")
    laundry = st.toggle("Tendré lavadora durante el viaje")
    language = st.selectbox("Idioma de la lista", ["es", "en"], index=0)

    apply_sidebar = st.button("🔄 Aplicar cambios")

# Form de entrada principal
with st.form("trip_form"):
    city = st.text_input("¿A dónde viajás?", placeholder="Ej: Neuquén, Argentina")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha de inicio", value=dt.date.today() + dt.timedelta(days=7))
    with col2:
        end_date = st.date_input("Fecha de regreso", value=dt.date.today() + dt.timedelta(days=14))
    activities = st.text_input("Actividades (opcional, separá por coma)", placeholder="playa, trekking, salidas nocturnas")
    submit = st.form_submit_button("Generar")

# ====== Submit: calcula y guarda en session_state (persistente) ======
def compute_and_store(city, start_date, end_date, activities, detail_level, traveler_profiles, carry_on, laundry, language):
    geo = geocode_city(city)
    forecast = fetch_forecast(geo["lat"], geo["lon"], start_date, end_date, geo["timezone"])
    daily = forecast["daily"]
    weather_brief = summarize_weather(daily)

    avg_min = sum(daily["temperature_2m_min"]) / len(daily["temperature_2m_min"])
    avg_max = sum(daily["temperature_2m_max"]) / len(daily["temperature_2m_max"])
    wet_days = sum(1 for p in daily.get("precipitation_probability_max", []) if p is not None and p >= 40)
    acts = [normalize_text(a) for a in activities.split(",")] if activities else []
    trip_days = (end_date - start_date).days + 1

    opts = {
        "detail_level": detail_level,
        "profiles": traveler_profiles,
        "carry_on": carry_on,
        "laundry": laundry,
        "language": language,
    }

    try:
        packing = generate_packing_with_openai(weather_brief, geo["name"], trip_days, acts, opts)
        st.success("Lista generada con OpenAI ✅")
    except Exception as e:
        st.info("Usando lógica local (sin OpenAI) ✅")
        st.caption(f"Motivo: {e}")
        packing = rule_based_packing(avg_min, avg_max, wet_days, acts, trip_days)

    st.session_state.packing = packing
    st.session_state.result_meta = {
        "geo": geo,
        "weather_brief": weather_brief,
        "avg_min": avg_min,
        "avg_max": avg_max,
        "wet_days": wet_days,
        "trip_days": trip_days,
        "acts": acts,
        "opts": opts,
        "date_range": (start_date, end_date),
        "activities_raw": activities or "",
        "city_raw": city,
    }
    # Reset de checks al regenerar (para evitar “fantasmas”)
    st.session_state.checked = {}

if submit:
    try:
        if not city or start_date > end_date:
            st.error("Completá la ciudad y verificá el rango de fechas.")
        else:
            compute_and_store(city, start_date, end_date, activities, detail_level, traveler_profiles, carry_on, laundry, language)
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

# ====== Aplicar cambios desde la barra lateral ======
if apply_sidebar and st.session_state.packing and st.session_state.result_meta:
    meta = st.session_state.result_meta
    try:
        compute_and_store(
            meta["city_raw"], meta["date_range"][0], meta["date_range"][1],
            meta["activities_raw"], detail_level, traveler_profiles, carry_on, laundry, language
        )
        st.success("Preferencias aplicadas ✅")
    except Exception as e:
        st.error(f"Ocurrió un error al aplicar cambios: {e}")

# ====== Render persistente (no desaparece al tildar/añadir) ======
if st.session_state.packing:
    meta = st.session_state.result_meta
    packing = st.session_state.packing

    st.subheader(f"📍 {meta['geo']['name']}, {meta['geo']['country']}")
    st.code(meta["weather_brief"])

    # --- Botones de Marcar/Desmarcar TODO (ANTES de dibujar checkboxes) ---
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Marcar todo"):
            for cat, items in packing.items():
                for it in items:
                    st.session_state.checked[item_key(cat, it)] = True
            st.rerun()
    with col_b:
        if st.button("Desmarcar todo"):
            for cat, items in packing.items():
                for it in items:
                    st.session_state.checked[item_key(cat, it)] = False
            st.rerun()

    # --- Checkboxes persistentes ---
    st.subheader("✅ Lista sugerida para la valija")
    for cat, items in packing.items():
        with st.expander(cat, expanded=True):
            for it in items:
                k = item_key(cat, it)
                if k not in st.session_state.checked:
                    st.session_state.checked[k] = False
                # usamos la versión normalizada en la clave, y el texto original como etiqueta
                st.session_state.checked[k] = st.checkbox(normalize_text(it), key=k)

    # ===== Progreso (porcentaje y barra) =====
    all_items = sum(len(items) for items in packing.values())
    checked_items = sum(
        1
        for cat, items in packing.items()
        for it in items
        if st.session_state.checked.get(item_key(cat, it), False)
    )
    progress = (checked_items / all_items) if all_items else 0.0
    st.write(f"Progreso: **{checked_items} / {all_items}** ítems ({round(progress*100)}%)")
    st.progress(progress)

    # Form para agregar ítems manualmente (no rompe nada al enviar)
    with st.form("add_item_form", clear_on_submit=True):
        new_item = st.text_input("Agregar ítem manual", placeholder="Ej: Adaptador USB-C")
        add_submit = st.form_submit_button("Añadir")
    if add_submit and new_item:
        new_item_norm = normalize_text(new_item)
        # evitar duplicado (case/espacios)
        already = any(normalize_text(x).lower() == new_item_norm.lower() for x in packing.setdefault("Varios", []))
        if not already:
            packing["Varios"].append(new_item_norm)
            st.session_state.packing = packing  # guardar cambio
            st.session_state.checked.setdefault(item_key("Varios", new_item_norm), False)
            st.success(f"Añadido: {new_item_norm}")
            st.rerun()
        else:
            st.info("Ese ítem ya existe en 'Varios'.")

    # Filtro de búsqueda
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

    # Exportar TXT y CSV
    txt = export_txt(packing)
    st.download_button("⬇️ Descargar lista .txt", data=txt, file_name="lista_valija.txt", mime="text/plain")

    rows = []
    for cat, items in packing.items():
        for it in items:
            rows.append({
                "Categoria": cat,
                "Item": normalize_text(it),
                "Empacado": st.session_state.checked.get(item_key(cat, it), False)
            })
    df = pd.DataFrame(rows)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar lista .csv", data=csv, file_name="lista_valija.csv", mime="text/csv")

    # Regenerar más detallado (sube el nivel y re-llama IA/fallback)
    if st.button("🔁 Regenerar con más detalle"):
        meta["opts"]["detail_level"] = min(5, meta["opts"]["detail_level"] + 1)
        try:
            compute_and_store(
                meta["city_raw"], meta["date_range"][0], meta["date_range"][1],
                meta["activities_raw"], meta["opts"]["detail_level"], meta["opts"]["profiles"],
                meta["opts"]["carry_on"], meta["opts"]["laundry"], meta["opts"]["language"]
            )
            st.success("Lista (más detallada) generada ✅")
        except Exception as e:
            st.info("Usando lógica local (sin OpenAI) ✅")
            st.caption(f"Motivo: {e}")