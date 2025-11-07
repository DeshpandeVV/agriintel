# app.py — AgriIntel (Agriculture Themed, Gemini report, Unicode-safe PDF)

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
from matplotlib import font_manager  # to locate DejaVu Sans for Unicode PDF
import textwrap

# =========================================================
# PAGE CONFIG + THEME (Agriculture style)
# =========================================================
st.set_page_config(page_title="AgriIntel • Agriculture Intelligence", layout="wide", page_icon="🌱")

PRIMARY = "#166534"     # green-800
ACCENT  = "#22c55e"     # green-500
SAND    = "#f8fafc"     # light background
BORDER  = "#e5e7eb"     # gray-200
TEXT    = "#0f172a"     # slate-900

st.markdown(f"""
    <style>
        html, body, [class*="css"] {{
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
            color: {TEXT};
        }}
        .agri-hero {{
            background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
            padding: 20px 24px;
            border-radius: 20px;
            color: white;
            margin: 8px 0 18px;
            box-shadow: 0 10px 30px rgba(22,101,52,.25);
        }}
        .agri-card {{
            background: white;
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 6px 20px rgba(0,0,0,.04);
        }}
        .agri-metric {{
            background: {SAND};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 14px 16px;
        }}
        .btn-primary button {{
            background: {PRIMARY}; color: #fff; border-radius: 12px; height: 44px; border: 0;
        }}
        .btn-primary button:hover {{ background: #14532d; }}
        .btn-ghost button {{
            background: white; color: {TEXT}; border-radius: 12px; height: 44px; border: 1px solid {BORDER};
        }}
        .mini-help {{
            font-size: 13px; opacity: .9; margin-top: 6px;
        }}
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
        return fallback or f"(AI explanation unavailable: {e})"

# =========================================================
# LOGIN (demo)
# =========================================================
def login():
    st.markdown('<div class="agri-hero"><h2 style="margin:0;">🌱 AgriIntel — Secure Login</h2><p class="mini-help">Demo credentials: admin@agriintel.app / agriintel123</p></div>', unsafe_allow_html=True)
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
    # yield_prediction_model.pkl
    "yield": "https://drive.google.com/uc?export=download&id=1EMwJ9wr_s5yMvRtpDTkP4Va2csniqfSv",
    # soil_label_encoder.pkl
    "soil_encoder": "https://drive.google.com/uc?export=download&id=10fo75uk_uY6fYPcUZTXd-6AqolelWwDe",
    # soil_health_classification_model.pkl
    "soil": "https://drive.google.com/uc?export=download&id=1tQcpfJ3M8s3m5fuXVZ3ZrKuAWyfMrLhm",
    # fertilizer_recommendation_model.pkl
    "fert": "https://drive.google.com/uc?export=download&id=16lWBeuxyKF1FjvIgka8fGEteadqEgrHc",
    # crop_recommendation_model.pkl
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
st.markdown('<div class="agri-hero"><h2 style="margin:0;">🌱 AgriIntel — Agriculture Intelligence</h2><p class="mini-help">Crop • Fertilizer • Yield • Soil • Live Weather • Seasonal Outlook • AI Advisory</p></div>', unsafe_allow_html=True)

tip_col, logout_col = st.columns([3,1])
with tip_col:
    st.markdown('<div class="agri-card">Choose your language and region in the sidebar, then click <b>Analyze</b>. The report will include a detailed AI advisory in your selected language.</div>', unsafe_allow_html=True)
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
# ANALYZE (Agriculture UI)
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

    # Save everything to session for PDF + rendering
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

    # Metrics
    a,b,c,d = st.columns(4)
    a.markdown(f'<div class="agri-metric"><b>🌿 Recommended Crop</b><h3 style="margin:8px 0">{crop}</h3></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="agri-metric"><b>📈 Predicted Yield</b><h3 style="margin:8px 0">{y_pred} t/ha</h3></div>', unsafe_allow_html=True)
    c.markdown(f'<div class="agri-metric"><b>🧪 Soil Health</b><h3 style="margin:8px 0">{soil_h}</h3></div>', unsafe_allow_html=True)
    d.markdown(f'<div class="agri-metric"><b>📍 Location</b><h3 style="margin:8px 0">{loc_name} ({cc})</h3></div>', unsafe_allow_html=True)

    # Fertilizer
    st.markdown("#### 🧪 Fertilizer Recommendation (kg/ha)")
    st.markdown('<div class="agri-card">', unsafe_allow_html=True)
    st.table(pd.DataFrame([fert]))
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

    # Gemini (short) explanation for the page
    prompt_short = f"""
    Write a short, farmer-friendly advisory in {language}.
    Location: {loc_name}, {cc}  (lat {st.session_state._results['lat']}, lon {st.session_state._results['lon']})
    Recommended crop: {crop}. Soil health: {soil_h}.
    Fertilizer (kg/ha): N={fert['delta_N']}, P={fert['delta_P']}, K={fert['delta_K']}.
    Predicted yield: {y_pred} t/ha.
    Current weather: temp={cur.get('temperature_2m')}°C, RH={cur.get('relative_humidity_2m')}%, precip={cur.get('precipitation')} mm.
    16-day forecast summary: avg Tmax={(df_16['t_max'].mean() if not df_16.empty else 'NA')}, total precip={(df_16['precip_mm'].sum() if not df_16.empty else 'NA')} mm.
    Be concise, bullet points only.
    """
    explanation_short = gemini_text(prompt_short, fallback="(Add GEMINI_API_KEY in Secrets to enable AI explanation.)")
    st.session_state._explanation_short = explanation_short

    st.markdown("#### 🗣️ AI Advisory (Short)")
    st.markdown('<div class="agri-card">', unsafe_allow_html=True)
    st.write(explanation_short)
    st.markdown('</div>', unsafe_allow_html=True)

    # Gemini (long) detailed report for PDF
    prompt_long = f"""
    Create a comprehensive advisory in {language} for farmers. Use clear headings and bullet points.
    Include the following sections in detail:
    1) Overview (region: {loc_name}, {cc}; coordinates: {st.session_state._results['lat']}, {st.session_state._results['lon']})
    2) Recommended Crop and Rationale ({crop})
    3) Soil Health Status: {soil_h} — with practical soil management tips
    4) Fertilizer Plan (kg/ha): N={fert['delta_N']}, P={fert['delta_P']}, K={fert['delta_K']} — with timing/splitting advice
    5) Yield Expectation: {y_pred} t/ha — explain key drivers and risks
    6) Current Weather: temp={cur.get('temperature_2m')}°C, RH={cur.get('relative_humidity_2m')}%, precip={cur.get('precipitation')} mm
    7) 16-Day Forecast Summary: avg Tmax={(df_16['t_max'].mean() if not df_16.empty else 'NA')}, total precip={(df_16['precip_mm'].sum() if not df_16.empty else 'NA')} mm — irrigation/pest implications
    8) Seasonal Outlook (3 months): {df_month.to_dict(orient='records') if not df_month.empty else 'NA'} — planning and risk mitigation
    9) Operations Calendar: sowing, nutrient application windows, irrigation schedule, and monitoring checklist
    Keep the tone supportive and actionable. Avoid jargon. Use line breaks sensibly.
    """
    explanation_long = gemini_text(prompt_long, fallback="(Enable GEMINI_API_KEY in Secrets to add a detailed AI report.)")
    st.session_state._explanation_long = explanation_long

# =========================================================
# PDF DOWNLOAD — Agriculture theme, Unicode-safe & wrapped
# =========================================================
def safe_wrap(text, width=95):
    """Wrap long text safely for FPDF multi_cell."""
    lines = []
    for para in (text or "").splitlines():
        para = para.replace("\t", "    ")
        if not para.strip():
            lines.append("")  # preserve blank lines
        else:
            lines.extend(textwrap.wrap(para, width=width, break_long_words=True, break_on_hyphens=False))
    return lines

if st.session_state.get("analysis_done", False):
    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    pdf_clicked = st.button("📄 Generate & Download Detailed PDF", use_container_width=True)
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

        explanation_long = st.session_state.get("_explanation_long", "")

        # --- Prepare PDF
        pdf = FPDF()
        pdf.add_page()

        # ✅ Set Unicode font BEFORE writing anything
        font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)

        # Title
        pdf.cell(0, 10, "AgriIntel Advisory Report", ln=True, align="C")
        pdf.ln(2)
        pdf.set_font("DejaVu", size=10)
        pdf.multi_cell(0, 7, f"Region: {region} | Location: {loc_name}, {cc} | Coordinates: {lat}, {lon}")

        # Key metrics
        pdf.ln(3)
        pdf.set_font("DejaVu", size=11)
        pdf.cell(0, 8, "Summary", ln=True)
        pdf.set_font("DejaVu", size=10)
        pdf.multi_cell(0, 6, f"Recommended Crop: {crop}")
        pdf.multi_cell(0, 6, f"Predicted Yield: {y_pred} t/ha")
        pdf.multi_cell(0, 6, f"Soil Health: {soil_h}")
        pdf.multi_cell(0, 6, f"Fertilizer Plan (kg/ha): N={fert['delta_N']}, P={fert['delta_P']}, K={fert['delta_K']}")

        # Weather now
        pdf.ln(3)
        pdf.set_font("DejaVu", size=11)
        pdf.cell(0, 8, "Weather (Now)", ln=True)
        pdf.set_font("DejaVu", size=10)
        pdf.multi_cell(0, 6, f"Temperature: {cur.get('temperature_2m')}°C")
        pdf.multi_cell(0, 6, f"Humidity: {cur.get('relative_humidity_2m')}%")
        pdf.multi_cell(0, 6, f"Precipitation: {cur.get('precipitation')} mm")

        # 16-day
        if not df_16.empty:
            pdf.ln(3)
            pdf.set_font("DejaVu", size=11); pdf.cell(0, 8, "16-Day Forecast", ln=True)
            pdf.set_font("DejaVu", size=9)
            for _, row in df_16.iterrows():
                text = f"{row['date']} → Tmax {row['t_max']}°C, Tmin {row['t_min']}°C, Precip {row['precip_mm']} mm"
                for seg in safe_wrap(text, width=100):
                    pdf.multi_cell(0, 5, seg)

        # Seasonal
        if not df_month.empty:
            pdf.ln(3)
            pdf.set_font("DejaVu", size=11); pdf.cell(0, 8, "3-Month Seasonal Outlook", ln=True)
            pdf.set_font("DejaVu", size=9)
            for _, row in df_month.iterrows():
                text = f"{row['month']} → Temp Mean {row['temp_mean']}°C, Rainfall {row['precip_sum']} mm"
                for seg in safe_wrap(text, width=100):
                    pdf.multi_cell(0, 5, seg)

        # Detailed AI report (Gemini)
        pdf.ln(3)
        pdf.set_font("DejaVu", size=11); pdf.cell(0, 8, "Detailed AI Advisory", ln=True)
        pdf.set_font("DejaVu", size=10)
        for seg in safe_wrap(explanation_long, width=95):
            pdf.multi_cell(0, 5.5, seg)

        # Save + Download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            tmp.seek(0)
            st.download_button(
                label="⬇️ Download Detailed PDF",
                data=tmp.read(),
                file_name="AgriIntel_Advisory_Report.pdf",
                mime="application/pdf"
            )
