# app.py — AgriIntel Premium (A3 PDF, 2-charts/page, Cover, Gemini, Unicode-safe, Drive models)

import os
import io
import tempfile
import requests
import datetime as dt
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from fpdf import FPDF
import google.generativeai as genai
from matplotlib import font_manager  # Unicode font path
import matplotlib.pyplot as plt
import textwrap

# =========================================================
# PAGE CONFIG + THEME (Agriculture style)
# =========================================================
st.set_page_config(page_title="AgriIntel Premium • Agriculture Intelligence", layout="wide", page_icon="🌱")

APP_NAME = "AgriIntel Premium"

PRIMARY = "#166534"     # green-800
ACCENT  = "#22c55e"     # green-500
SAND    = "#f8fafc"     # light background
BORDER  = "#e5e7eb"     # gray-200
TEXT    = "#0f172a"     # slate-900

st.markdown(f"""
    <style>
        html, body, [class*="css"] {{
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial;
            color: {TEXT};
        }}
        .agri-hero {{
            background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
            padding: 20px 24px; border-radius: 20px; color: white;
            margin: 8px 0 18px; box-shadow: 0 10px 30px rgba(22,101,52,.25);
        }}
        .agri-card {{
            background: white; border: 1px solid {BORDER}; border-radius: 16px;
            padding: 16px; box-shadow: 0 6px 20px rgba(0,0,0,.04);
        }}
        .agri-metric {{
            background: {SAND}; border: 1px solid {BORDER}; border-radius: 16px; padding: 12px 14px;
        }}
        .btn-primary button {{ background: {PRIMARY}; color:#fff; border-radius: 12px; height:44px; border:0; }}
        .btn-primary button:hover {{ background:#14532d; }}
        .mini-help {{ font-size: 13px; opacity:.9; margin-top:6px; }}
    </style>
""", unsafe_allow_html=True)

# =========================================================
# LANGS & GEMINI
# =========================================================
LANGS = {"English":"en","Hindi":"hi","Marathi":"mr","Tamil":"ta","Telugu":"te","Kannada":"kn"}

GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
USE_GEMINI = bool(GEMINI_KEY)
if USE_GEMINI:
    genai.configure(api_key=GEMINI_KEY)
    gemini = genai.GenerativeModel("gemini-pro")

def gemini_text(prompt, fallback=""):
    if not USE_GEMINI:
        return fallback or "(AI not configured — add GEMINI_API_KEY in Secrets)"
    try:
        resp = gemini.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        # Show real error so you can debug on Streamlit Cloud
        return f"(Gemini error: {e})"

# =========================================================
# LOGIN (demo creds)
# =========================================================
def login():
    st.markdown(f'<div class="agri-hero"><h2 style="margin:0;">🌱 {APP_NAME} — Secure Login</h2><p class="mini-help">Demo: admin@agriintel.app / agriintel123</p></div>', unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", value="")
        pwd = st.text_input("Password", type="password", value="")
        ok = st.form_submit_button("Login", use_container_width=True)
        if ok:
            if email == "admin@agriintel.app" and pwd == "agriintel123":
                st.session_state.auth = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    login()
    st.stop()

# =========================================================
# MODEL LOADING — GOOGLE DRIVE
# =========================================================
MODEL_URLS = {
    "yield": "https://drive.google.com/uc?export=download&id=1EMwJ9wr_s5yMvRtpDTkP4Va2csniqfSv",
    "soil_encoder": "https://drive.google.com/uc?export=download&id=10fo75uk_uY6fYPcUZTXd-6AqolelWwDe",
    "soil": "https://drive.google.com/uc?export=download&id=1tQcpfJ3M8s3m5fuXVZ3ZrKuAWyfMrLhm",
    "fert": "https://drive.google.com/uc?export=download&id=16lWBeuxyKF1FjvIgka8fGEteadqEgrHc",
    "crop": "https://drive.google.com/uc?export=download&id=10y_phgu-8AV-gdH2K47TqOAw37L7vr-b"
}

@st.cache_resource(show_spinner=True)
def load_drive_model(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))

with st.spinner("⚙️ Loading ML models…"):
    try:
        crop_model = load_drive_model(MODEL_URLS["crop"])
        fert_model = load_drive_model(MODEL_URLS["fert"])
        soil_model = load_drive_model(MODEL_URLS["soil"])
        soil_encoder = load_drive_model(MODEL_URLS["soil_encoder"])
        yield_model = load_drive_model(MODEL_URLS["yield"])
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# =========================================================
# SIDEBAR — INPUTS
# =========================================================
with st.sidebar:
    st.header("🧾 Inputs")
    language = st.selectbox("Language", list(LANGS.keys()))
    region = st.text_input("Region / Place", "Pune, India")
    st.divider()
    N = st.number_input("Nitrogen (N)", 0, 300, 90)
    P = st.number_input("Phosphorus (P)", 0, 300, 40)
    K = st.number_input("Potassium (K)", 0, 300, 40)
    temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 75.0)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0, 200.0)

# =========================================================
# HEADER
# =========================================================
st.markdown(f'<div class="agri-hero"><h2 style="margin:0;">🌱 {APP_NAME} — Agriculture Intelligence</h2><p class="mini-help">Crop • Fertilizer • Yield • Soil • Live Weather • Seasonal Outlook • AI Advisory</p></div>', unsafe_allow_html=True)

tip_col, logout_col = st.columns([3,1])
with tip_col:
    st.markdown('<div class="agri-card">Choose your language and region in the sidebar, then click <b>Analyze</b>. The PDF includes a professional cover page, charts, and a detailed AI advisory.</div>', unsafe_allow_html=True)
with logout_col:
    if st.button("Logout", use_container_width=True):
        st.session_state.auth = False
        st.rerun()

# =========================================================
# WEATHER HELPERS — OPEN-METEO
# =========================================================
@st.cache_data(show_spinner=False)
def geocode(place: str):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": place, "count": 1}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("results"):
        d = data["results"][0]
        return d["latitude"], d["longitude"], d["name"], d["country_code"]
    else:
        raise ValueError("Location not found")

@st.cache_data(show_spinner=False)
def get_realtime_and_daily(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 16, "timezone": "auto"
    }, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def get_seasonal_monthly(lat, lon):
    start = dt.date.today().replace(day=1)
    end = (pd.Timestamp(start) + pd.DateOffset(months=3)).date()
    r = requests.get("https://seasonal-api.open-meteo.com/v1/seasonal", params={
        "latitude": lat, "longitude": lon,
        "models": "ecmwf_seas5",
        "monthly": "temperature_2m_mean,precipitation_sum",
        "start_date": start, "end_date": end, "timezone": "auto"
    }, timeout=30)
    r.raise_for_status()
    return r.json()

# =========================================================
# MODEL HELPERS
# =========================================================
def predict_crop():
    X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    return crop_model.predict(X)[0]

def predict_fertilizer(crop):
    feats = getattr(fert_model, "feature_names_in_", [])
    row = {f: 0 for f in feats}
    for k, v in {"N": N, "P": P, "K": K}.items():
        if k in row: row[k] = v
    for f in feats:
        if f == f"crop_{crop}": row[f] = 1
    X = pd.DataFrame([row], columns=feats)
    pred = fert_model.predict(X)[0]
    return {"delta_N": round(pred[0],2), "delta_P": round(pred[1],2), "delta_K": round(pred[2],2)}

def predict_soil():
    pred = soil_model.predict([[N, P, K, ph]])[0]
    return soil_encoder.inverse_transform([pred])[0]

def predict_yield(crop):
    feats = getattr(yield_model, "feature_names_in_", [])
    row = {f: 0 for f in feats}
    for k,v in {"N":N, "P":P, "K":K, "temperature":temperature, "humidity":humidity, "ph":ph, "rainfall":rainfall}.items():
        if k in row: row[k]=v
    for f in feats:
        if f == f"crop_{crop}": row[f]=1
    X = pd.DataFrame([row], columns=feats)
    return round(float(yield_model.predict(X)[0]),2)

# =========================================================
# CHART HELPERS (matplotlib)
# =========================================================
def save_fig(fig) -> str:
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return path

def make_16day_temp_chart(df16: pd.DataFrame) -> str:
    if df16.empty: return ""
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(pd.to_datetime(df16["date"]), df16["t_max"], label="Tmax (°C)")
    ax.plot(pd.to_datetime(df16["date"]), df16["t_min"], label="Tmin (°C)")
    ax.set_ylabel("°C"); ax.set_title("16-Day Temperature Forecast"); ax.legend()
    fig.autofmt_xdate()
    return save_fig(fig)

def make_16day_rain_bar(df16: pd.DataFrame) -> str:
    if df16.empty: return ""
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.bar(pd.to_datetime(df16["date"]), df16["precip_mm"])
    ax.set_title("16-Day Rainfall (mm)"); ax.set_ylabel("mm")
    fig.autofmt_xdate()
    return save_fig(fig)

def make_seasonal_temp_chart(dfm: pd.DataFrame) -> str:
    if dfm.empty: return ""
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(pd.to_datetime(dfm["month"]), dfm["temp_mean"], marker="o")
    ax.set_title("3-Month Seasonal Mean Temperature (°C)"); ax.set_ylabel("°C")
    fig.autofmt_xdate()
    return save_fig(fig)

def make_npk_bar(n, p, k) -> str:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(["N", "P", "K"], [n, p, k])
    ax.set_title("NPK Recommendation (kg/ha)")
    return save_fig(fig)

# =========================================================
# ANALYZE (UI)
# =========================================================
cta_a, cta_b = st.columns([1,2])
with cta_a:
    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze & Fetch Weather", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with cta_b:
    st.caption("Models: Crop • Fertilizer • Yield • Soil  |  Weather: Now • 16 days • 3 months  |  AI Advisory via Gemini")

if analyze_clicked:
    try:
        lat, lon, loc_name, cc = geocode(region)
        realtime = get_realtime_and_daily(lat, lon)
        try:
            seasonal = get_seasonal_monthly(lat, lon)
        except:
            seasonal = None
    except Exception as e:
        st.error(f"Weather lookup failed: {e}")
        st.stop()

    crop = predict_crop()
    fert = predict_fertilizer(crop)
    soil_h = predict_soil()
    y_pred = predict_yield(crop)

    st.session_state.analysis_done = True
    st.session_state._results = {
        "crop": crop, "fert": fert, "soil": soil_h, "yield": y_pred,
        "loc_name": loc_name, "cc": cc,
        "cur": realtime.get("current", {}),
        "daily": realtime.get("daily", {}),
        "seasonal": seasonal,
        "language": language, "region": region,
        "lat": lat, "lon": lon
    }

    # Metrics (compact)
    a,b,c,d = st.columns(4)
    a.markdown(f"<div class='agri-metric'><b>🌿 Recommended Crop</b><br><span style='font-size:22px;'>{crop}</span></div>", unsafe_allow_html=True)
    b.markdown(f"<div class='agri-metric'><b>📈 Predicted Yield</b><br><span style='font-size:22px;'>{y_pred} t/ha</span></div>", unsafe_allow_html=True)
    c.markdown(f"<div class='agri-metric'><b>🧪 Soil Health</b><br><span style='font-size:22px;'>{soil_h}</span></div>", unsafe_allow_html=True)
    d.markdown(f"<div class='agri-metric'><b>📍 Location</b><br><span style='font-size:22px;'>{loc_name} ({cc})</span></div>", unsafe_allow_html=True)

    # Fertilizer table (label negative as reduce)
    fert_display = fert.copy()
    note_lines = []
    for k in fert_display:
        if fert_display[k] < 0:
            note_lines.append(f"{k} is negative → reduce input by {abs(fert_display[k])} kg/ha")
            fert_display[k] = f"{fert_display[k]} (reduce)"
    st.markdown("#### 🧪 Fertilizer Recommendation (kg/ha)")
    st.markdown('<div class="agri-card">', unsafe_allow_html=True)
    st.table(pd.DataFrame([fert_display]))
    if note_lines: st.caption("Note: " + " | ".join(note_lines))
    st.markdown('</div>', unsafe_allow_html=True)

    # Real-time weather
    cur = st.session_state._results["cur"]
    st.markdown("#### 🌤️ Real-time Weather")
    st.markdown('<div class="agri-card">', unsafe_allow_html=True)
    st.write({
        "Temperature (°C)": cur.get("temperature_2m"),
        "Humidity (%)": cur.get("relative_humidity_2m"),
        "Precipitation (mm)": cur.get("precipitation"),
    })
    st.markdown('</div>', unsafe_allow_html=True)

    # 16-day forecast
    daily = st.session_state._results["daily"]
    df_16 = pd.DataFrame({
        "date": daily.get("time", []),
        "t_max": daily.get("temperature_2m_max", []),
        "t_min": daily.get("temperature_2m_min", []),
        "precip_mm": daily.get("precipitation_sum", [])
    })
    st.markdown("#### 📅 16-Day Forecast")
    st.markdown('<div class="agri-card">', unsafe_allow_html=True)
    st.dataframe(df_16, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Seasonal (3 months)
    seasonal = st.session_state._results["seasonal"]
    df_month = pd.DataFrame()
    st.markdown("#### 📈 3-Month Seasonal Averages")
    st.markdown('<div class="agri-card">', unsafe_allow_html=True)
    if seasonal and "monthly" in seasonal:
        try:
            m = seasonal["monthly"]
            df_month = pd.DataFrame({
                "month": m.get("time", []),
                "temp_mean": m.get("temperature_2m_mean", []),
                "precip_sum": m.get("precipitation_sum", [])
            }).head(3)
            st.dataframe(df_month, use_container_width=True)
        except:
            st.info("Seasonal forecast not available.")
    else:
        st.info("Seasonal forecast unavailable for this region/date.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Gemini short + long
    prompt_short = f"""
    Write a short, farmer-friendly advisory in {language}.
    Location: {loc_name}, {cc} (lat {st.session_state._results['lat']}, lon {st.session_state._results['lon']}).
    Crop: {crop}. Soil health: {soil_h}. NPK(kg/ha): N={fert['delta_N']}, P={fert['delta_P']}, K={fert['delta_K']}.
    Yield: {y_pred} t/ha. Current weather: T={cur.get('temperature_2m')}°C, RH={cur.get('relative_humidity_2m')}%, rain={cur.get('precipitation')} mm.
    16-day avg Tmax={(df_16['t_max'].mean() if not df_16.empty else 'NA')}, total rain={(df_16['precip_mm'].sum() if not df_16.empty else 'NA')} mm.
    Bullet points only.
    """
    explanation_short = gemini_text(prompt_short, fallback="(Add GEMINI_API_KEY to enable AI explanation.)")
    st.session_state._explanation_short = explanation_short

    st.markdown("#### 🗣️ AI Advisory (Short)")
    st.markdown('<div class="agri-card">', unsafe_allow_html=True)
    st.write(explanation_short)
    st.markdown('</div>', unsafe_allow_html=True)

    prompt_long = f"""
    Create a comprehensive advisory in {language} with headings and bullet points:
    1) Overview — region {loc_name}, {cc} (lat {st.session_state._results['lat']}, lon {st.session_state._results['lon']}).
    2) Recommended crop and rationale: {crop}.
    3) Soil health: {soil_h} — soil management tips.
    4) Fertilizer plan (kg/ha): N={fert['delta_N']}, P={fert['delta_P']}, K={fert['delta_K']} — timing/splitting advice; if negative, explain reduction.
    5) Yield expectation: {y_pred} t/ha — drivers and risks.
    6) Current weather: T={cur.get('temperature_2m')}°C, RH={cur.get('relative_humidity_2m')}%, rain={cur.get('precipitation')} mm.
    7) 16-day forecast summary: avg Tmax={(df_16['t_max'].mean() if not df_16.empty else 'NA')}, total rain={(df_16['precip_mm'].sum() if not df_16.empty else 'NA')} mm — irrigation/pest implications.
    8) Seasonal outlook (3 months): {df_month.to_dict(orient='records') if not df_month.empty else 'NA'} — planning and risk mitigation.
    9) Operations calendar: sowing, nutrition windows, irrigation schedule, monitoring checklist.
    Keep tone supportive and actionable.
    """
    explanation_long = gemini_text(prompt_long, fallback="(Enable GEMINI_API_KEY to add a detailed AI report.)")
    st.session_state._explanation_long = explanation_long

# =========================================================
# PDF — A3, Minimal Cover, 2 Charts per Page, Safe Width
# =========================================================
def wrap_text(text, width=95):
    lines = []
    for para in (text or "").splitlines():
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width, break_long_words=True, break_on_hyphens=False))
    return lines

if st.session_state.get("analysis_done", False):
    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    pdf_clicked = st.button("📄 Download Premium PDF Report", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if pdf_clicked:
        r   = st.session_state._results
        crop, fert, soil_h, y_pred = r["crop"], r["fert"], r["soil"], r["yield"]
        loc_name, cc, language, region = r["loc_name"], r["cc"], r["language"], r["region"]
        lat, lon = r["lat"], r["lon"]
        cur = r["cur"]
        daily = r["daily"]
        df_16 = pd.DataFrame({
            "date": daily.get("time", []),
            "t_max": daily.get("temperature_2m_max", []),
            "t_min": daily.get("temperature_2m_min", []),
            "precip_mm": daily.get("precipitation_sum", [])
        })
        seasonal = r["seasonal"]
        df_month = pd.DataFrame()
        if seasonal and "monthly" in seasonal:
            m = seasonal["monthly"]
            df_month = pd.DataFrame({
                "month": m.get("time", []),
                "temp_mean": m.get("temperature_2m_mean", []),
                "precip_sum": m.get("precipitation_sum", [])
            }).head(3)

        # Charts (paths)
        path_temp = make_16day_temp_chart(df_16)
        path_rain = make_16day_rain_bar(df_16)
        path_seas = make_seasonal_temp_chart(df_month)
        path_npk  = make_npk_bar(fert['delta_N'], fert['delta_P'], fert['delta_K'])

        # --- Prepare A3 PDF
        pdf = FPDF(format="A3")
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_page()

        # Unicode font BEFORE writing
        font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)

        # Hard-safe margins + width for A3
        pdf.set_left_margin(20)
        pdf.set_right_margin(20)
        SAFE_WIDTH = 250  # A3 width (297) - 40 margins = 257 -> use 250 to be safe

        # -------- Cover Page (Minimal Professional) --------
        pdf.set_font("DejaVu", size=28)
        pdf.cell(SAFE_WIDTH, 16, f"{APP_NAME}", ln=True, align="C")
        pdf.set_font("DejaVu", size=18)
        pdf.cell(SAFE_WIDTH, 12, "Advisory Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("DejaVu", size=12)
        today = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        pdf.multi_cell(SAFE_WIDTH, 8, f"Region: {region}")
        pdf.multi_cell(SAFE_WIDTH, 8, f"Location: {loc_name}, {cc}")
        pdf.multi_cell(SAFE_WIDTH, 8, f"Coordinates: {lat}, {lon}")
        pdf.multi_cell(SAFE_WIDTH, 8, f"Generated: {today}")
        pdf.ln(8)
        for seg in wrap_text("Unified crop, fertilizer, yield & soil insights with live weather, seasonal outlook, and AI advisory.", width=110):
            pdf.multi_cell(SAFE_WIDTH, 7, seg)

        # -------- Summary Page --------
        pdf.add_page()
        pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "Summary", ln=True)
        pdf.set_font("DejaVu", size=12)
        fert_line = (
            f"N={fert['delta_N']} kg/ha | "
            f"P={fert['delta_P']} kg/ha ({'reduce' if fert['delta_P'] < 0 else 'apply'}) | "
            f"K={fert['delta_K']} kg/ha"
        )
        pdf.multi_cell(SAFE_WIDTH, 7, f"Recommended Crop: {crop}")
        pdf.multi_cell(SAFE_WIDTH, 7, f"Predicted Yield: {y_pred} t/ha")
        pdf.multi_cell(SAFE_WIDTH, 7, f"Soil Health: {soil_h}")
        pdf.multi_cell(SAFE_WIDTH, 7, f"Fertilizer Plan: {fert_line}")
        pdf.ln(4)
        pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "Weather (Now)", ln=True)
        pdf.set_font("DejaVu", size=12)
        pdf.multi_cell(SAFE_WIDTH, 7, f"Temperature: {cur.get('temperature_2m')}°C")
        pdf.multi_cell(SAFE_WIDTH, 7, f"Humidity: {cur.get('relative_humidity_2m')}%")
        pdf.multi_cell(SAFE_WIDTH, 7, f"Precipitation: {cur.get('precipitation')} mm")

        # -------- Charts Page 1 (2 charts) --------
        if path_temp or path_rain:
            pdf.add_page()
            pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "Forecast Charts", ln=True)
            if path_temp and os.path.exists(path_temp):
                pdf.ln(2); pdf.image(path_temp, w=SAFE_WIDTH)
            if path_rain and os.path.exists(path_rain):
                pdf.ln(4); pdf.image(path_rain, w=SAFE_WIDTH)

        # -------- Charts Page 2 (2 charts) --------
        if path_seas or path_npk:
            pdf.add_page()
            pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "Seasonal & Nutrition Charts", ln=True)
            if path_seas and os.path.exists(path_seas):
                pdf.ln(2); pdf.image(path_seas, w=SAFE_WIDTH)
            if path_npk and os.path.exists(path_npk):
                pdf.ln(4); pdf.image(path_npk, w=SAFE_WIDTH/2)  # half width looks neat

        # -------- Detailed AI advisory --------
        explanation_long = st.session_state.get("_explanation_long", "")
        pdf.add_page()
        pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "Detailed AI Advisory", ln=True)
        pdf.set_font("DejaVu", size=11)
        for seg in wrap_text(explanation_long, width=120):
            pdf.multi_cell(SAFE_WIDTH, 6.5, seg)

        # Save + download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            tmp.seek(0)
            st.download_button(
                label="⬇️ Download Premium PDF",
                data=tmp.read(),
                file_name=f"{APP_NAME.replace(' ', '_')}_Report_A3.pdf",
                mime="application/pdf"
            )
