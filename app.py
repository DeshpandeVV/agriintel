
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

# =========================================================
# PAGE CONFIG + THEME
# =========================================================
st.set_page_config(page_title="AgriIntel Premium", layout="wide", page_icon="🌾")

PRIMARY = "#16a34a"   # Tailwind green-600
BG_CARD = "#ffffff"
BORDER = "#e5e7eb"

st.markdown(f"""
    <style>
        .agri-header {{
            background: linear-gradient(90deg, {PRIMARY} 0%, #22c55e 100%);
            padding: 18px 22px; border-radius: 16px; color: white; margin-bottom: 16px;
        }}
        .agri-card {{
            background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 16px;
            padding: 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.04);
        }}
        .metric {{
            border-radius: 14px; padding: 14px; border: 1px solid {BORDER};
            background: #fafafa;
        }}
        .btn-primary button {{
            background: {PRIMARY}; color: white; border-radius: 12px; height: 44px;
            border: none;
        }}
        .btn-primary button:hover {{ background: #15803d; }}
        .btn-outline button {{
            background: white; color: #111827; border-radius: 12px; height: 44px;
            border: 1px solid {BORDER};
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
        return fallback or prompt
    try:
        resp = gemini.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return fallback or f"(AI explanation unavailable: {e})"

# =========================================================
# LOGIN (demo)
# =========================================================
def login():
    st.markdown('<div class="agri-header"><h2 style="margin:0;">🌾 AgriIntel — Secure Login</h2></div>', unsafe_allow_html=True)
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
st.markdown('<div class="agri-header"><h2 style="margin:0;">🌾 AgriIntel — Premium Smart Agriculture Dashboard</h2><p style="margin:6px 0 0;opacity:.95">Unified crop, fertilizer, yield & soil insights with live weather and seasonal outlook.</p></div>', unsafe_allow_html=True)

top_left, top_right = st.columns([3,1])
with top_left:
    st.markdown('<div class="agri-card"><b>Tip:</b> Choose your language and region in the sidebar, then click <b>Analyze</b>.</div>', unsafe_allow_html=True)
with top_right:
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
# ANALYZE (Premium layout)
# =========================================================
analyze = st.container()
with analyze:
    cta_left, cta_right = st.columns([1,1])
    with cta_left:
        analyze_clicked = st.button("🔍 Analyze & Fetch Weather", use_container_width=True)
    with cta_right:
        st.caption("Models: Crop • Fertilizer • Yield • Soil  |  Weather: Now • 16 days • 3 months")

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
        "language": language, "region": region
    }

    # --- Metric Cards ---
    a,b,c,d = st.columns(4)
    a.markdown(f'<div class="metric"><b>🌿 Recommended Crop</b><h3 style="margin:6px 0">{crop}</h3></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="metric"><b>📈 Predicted Yield</b><h3 style="margin:6px 0">{y_pred} t/ha</h3></div>', unsafe_allow_html=True)
    c.markdown(f'<div class="metric"><b>🧪 Soil Health</b><h3 style="margin:6px 0">{soil_h}</h3></div>', unsafe_allow_html=True)
    d.markdown(f'<div class="metric"><b>📍 Location</b><h3 style="margin:6px 0">{loc_name} ({cc})</h3></div>', unsafe_allow_html=True)

    # --- Fertilizer Table ---
    st.markdown("#### 🧪 Fertilizer Recommendation (kg/ha)")
    st.markdown('<div class="agri-card">', unsafe_allow_html=True)
    st.table(pd.DataFrame([fert]))
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Weather Now ---
    cur = st.session_state._results["cur"]
    st.markdown("#### 🌤️ Real-time Weather")
    st.markdown('<div class="agri-card">', unsafe_allow_html=True)
    st.write({
        "Temperature (°C)": cur.get("temperature_2m"),
        "Humidity (%)": cur.get("relative_humidity_2m"),
        "Precipitation (mm)": cur.get("precipitation"),
    })
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 16-day Forecast ---
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

    # --- Seasonal (3 months) ---
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

    # --- Gemini Explanation ---
    prompt = f"""
    Explain this report in {language} for a farmer:
    Region: {loc_name}, {cc}
    Crop: {crop} | Soil health: {soil_h} | Yield: {y_pred} t/ha
    NPK (kg/ha): {fert}
    Now weather: {cur}
    16-day: {df_16.to_dict(orient='records')[:5]} ... (truncated)
    3-month: {df_month.to_dict(orient='records') if not df_month.empty else "NA"}
    Keep it brief, bullet points, actionable.
    """
    explanation = gemini_text(prompt, fallback="(Add GEMINI_API_KEY in Secrets to enable AI explanation.)")
    st.session_state._explanation = explanation

    st.markdown("#### 🗣️ AI Explanation")
    st.markdown('<div class="agri-card">', unsafe_allow_html=True)
    st.write(explanation)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PDF DOWNLOAD — OUTSIDE ANALYZE
# Unicode-safe using Matplotlib's DejaVu Sans
# =========================================================
if st.session_state.get("analysis_done", False):
    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    pdf_clicked = st.button("📄 Generate & Download PDF", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if pdf_clicked:
        r = st.session_state._results
        explanation = st.session_state._explanation

        crop = r["crop"]; fert = r["fert"]; soil_h = r["soil"]; y_pred = r["yield"]
        loc_name = r["loc_name"]; cc = r["cc"]; cur = r["cur"]
        language = r["language"]; region = r["region"]
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

        # Locate a built-in Unicode font (DejaVu Sans from Matplotlib)
        font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)

        pdf = FPDF()
        pdf.add_page()
        # Register Unicode font
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)

        pdf.cell(0, 10, "AgriIntel Smart Report", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("DejaVu", size=11)
        pdf.multi_cell(0, 8, f"Language: {language}")
        pdf.multi_cell(0, 8, f"Region: {region} ({loc_name}, {cc})")
        pdf.multi_cell(0, 8, f"Recommended Crop: {crop}")
        pdf.multi_cell(0, 8, f"Predicted Yield: {y_pred} t/ha")
        pdf.multi_cell(0, 8, f"Soil Health: {soil_h}")
        pdf.multi_cell(0, 8, f"Fertilizer (kg/ha): {fert}")

        pdf.ln(5)
        pdf.multi_cell(0, 8,
            f"Weather Now: {cur.get('temperature_2m')}°C | "
            f"{cur.get('relative_humidity_2m')}% humidity | "
            f"{cur.get('precipitation')} mm rainfall"
        )

        if not df_16.empty:
            pdf.ln(5); pdf.set_font("DejaVu", size=12); pdf.cell(0, 8, "16-Day Forecast:", ln=True)
            pdf.set_font("DejaVu", size=10)
            for _, r2 in df_16.iterrows():
                pdf.multi_cell(0, 6, f"{r2['date']} → Tmax {r2['t_max']}°C, Tmin {r2['t_min']}°C, Precip {r2['precip_mm']} mm")

        if not df_month.empty:
            pdf.ln(5); pdf.set_font("DejaVu", size=12); pdf.cell(0, 8, "3-Month Seasonal Outlook:", ln=True)
            pdf.set_font("DejaVu", size=10)
            for _, r3 in df_month.iterrows():
                pdf.multi_cell(0, 6, f"{r3['month']} → Temp Mean {r3['temp_mean']}°C, Rainfall {r3['precip_sum']} mm")

        pdf.ln(5); pdf.set_font("DejaVu", size=12); pdf.cell(0, 8, "AI Explanation:", ln=True)
        pdf.set_font("DejaVu", size=10)
        for line in explanation.split("
"):
            pdf.multi_cell(0, 6, line)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            tmp.seek(0)
            st.download_button(
                label="⬇️ Download PDF",
                data=tmp.read(),
                file_name="AgriIntel_Report.pdf",
                mime="application/pdf"
            )
