import os
import io
import requests
import datetime as dt
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from fpdf import FPDF
import tempfile
import google.generativeai as genai

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="AgriIntel", layout="wide")

LANGS = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn"
}

# =========================================================
# GEMINI SETUP
# =========================================================
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
    except:
        return fallback or "(AI explanation unavailable)"

# =========================================================
# LOGIN SYSTEM
# =========================================================
def login():
    st.title("🌾 AgriIntel — Login")
    email = st.text_input("Email")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
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
# MODEL LOADING — GOOGLE DRIVE LINKS
# =========================================================
MODEL_URLS = {
    "yield": "https://drive.google.com/uc?export=download&id=1EMwJ9wr_s5yMvRtpDTkP4Va2csniqfSv",
    "soil_encoder": "https://drive.google.com/uc?export=download&id=10fo75uk_uY6fYPcUZTXd-6AqolelWwDe",
    "soil": "https://drive.google.com/uc?export=download&id=1tQcpfJ3M8s3m5fuXVZ3ZrKuAWyfMrLhm",
    "fert": "https://drive.google.com/uc?export=download&id=16lWBeuxyKF1FjvIgka8fGEteadqEgrHc",
    "crop": "https://drive.google.com/uc?export=download&id=10y_phgu-8AV-gdH2K47TqOAw37L7vr-b"
}

@st.cache_resource(show_spinner=True)
def load_drive_model(url):
    r = requests.get(url)
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))

with st.spinner("Loading ML models…"):
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
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("🧾 Inputs")

language = st.sidebar.selectbox("Language", list(LANGS.keys()))
region = st.sidebar.text_input("Region / Place", "Pune, India")

N = st.sidebar.number_input("Nitrogen (N)", 0, 300, 90)
P = st.sidebar.number_input("Phosphorus (P)", 0, 300, 40)
K = st.sidebar.number_input("Potassium (K)", 0, 300, 40)
temperature = st.sidebar.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 75.0)
ph = st.sidebar.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 1000.0, 200.0)

colA, colB = st.columns([2, 1])
with colA:
    st.title("🌾 AgriIntel — Smart Agriculture")
with colB:
    if st.button("Logout"):
        st.session_state.auth = False
        st.rerun()

# =========================================================
# WEATHER UTILITIES — OPEN-METEO
# =========================================================
def geocode(place):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": place, "count": 1})
    r.raise_for_status()
    data = r.json()
    if data.get("results"):
        d = data["results"][0]
        return d["latitude"], d["longitude"], d["name"], d["country_code"]
    else:
        raise ValueError("Location not found")

def get_realtime_and_daily(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast",
                     params={
                         "latitude": lat,
                         "longitude": lon,
                         "current": "temperature_2m,relative_humidity_2m,precipitation",
                         "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                         "forecast_days": 16,
                         "timezone": "auto"
                     })
    r.raise_for_status()
    return r.json()

def get_seasonal_monthly(lat, lon):
    start = dt.date.today().replace(day=1)
    end = (pd.Timestamp(start) + pd.DateOffset(months=3)).date()

    r = requests.get("https://seasonal-api.open-meteo.com/v1/seasonal",
                     params={
                         "latitude": lat,
                         "longitude": lon,
                         "models": "ecmwf_seas5",
                         "monthly": "temperature_2m_mean,precipitation_sum",
                         "start_date": start,
                         "end_date": end,
                         "timezone": "auto"
                     })
    r.raise_for_status()
    return r.json()

# =========================================================
# MODEL FUNCTIONS
# =========================================================
def predict_crop():
    X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    return crop_model.predict(X)[0]

def predict_fertilizer(crop):
    feats = getattr(fert_model, "feature_names_in_", [])
    row = {f: 0 for f in feats}

    for k, v in {"N": N, "P": P, "K": K}.items():
        if k in row:
            row[k] = v

    for f in feats:
        if f == f"crop_{crop}":
            row[f] = 1

    X = pd.DataFrame([row], columns=feats)
    pred = fert_model.predict(X)[0]

    return {
        "delta_N": round(pred[0], 2),
        "delta_P": round(pred[1], 2),
        "delta_K": round(pred[2], 2)
    }

def predict_soil():
    pred = soil_model.predict([[N, P, K, ph]])[0]
    return soil_encoder.inverse_transform([pred])[0]

def predict_yield(crop):
    feats = getattr(yield_model, "feature_names_in_", [])
    row = {f: 0 for f in feats}

    for k, v in {
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }.items():
        if k in row:
            row[k] = v

    for f in feats:
        if f == f"crop_{crop}":
            row[f] = 1

    X = pd.DataFrame([row], columns=feats)
    return round(float(yield_model.predict(X)[0]), 2)

# =========================================================
# ANALYZE BUTTON
# =========================================================
analyze_clicked = st.button("🔍 Analyze & Fetch Weather")

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

    # Store in session for PDF
    st.session_state.analysis_done = True
    st.session_state._results = {
        "crop": crop,
        "fert": fert,
        "soil": soil_h,
        "yield": y_pred,
        "loc_name": loc_name,
        "cc": cc,
        "cur": realtime.get("current", {}),
        "daily": realtime.get("daily", {}),
        "seasonal": seasonal
    }

    # ---------- DISPLAY ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended Crop", crop)
    c2.metric("Predicted Yield", f"{y_pred} t/ha")
    c3.metric("Soil Health", soil_h)
    c4.metric("Location", f"{loc_name} ({cc})")

    st.markdown("### 🧪 Fertilizer (kg/ha)")
    st.table(pd.DataFrame([fert]))

    cur = st.session_state._results["cur"]
    st.markdown("### 🌤️ Real-time Weather")
    st.write({
        "Temperature (°C)": cur.get("temperature_2m"),
        "Humidity (%)": cur.get("relative_humidity_2m"),
        "Precipitation (mm)": cur.get("precipitation")
    })

    daily = st.session_state._results["daily"]
    df_16 = pd.DataFrame({
        "date": daily.get("time", []),
        "t_max": daily.get("temperature_2m_max", []),
        "t_min": daily.get("temperature_2m_min", []),
        "precip_mm": daily.get("precipitation_sum", [])
    })

    if not df_16.empty:
        st.markdown("### 📅 16-Day Forecast")
        st.dataframe(df_16)

    seasonal = st.session_state._results["seasonal"]
    df_month = pd.DataFrame()

    if seasonal and "monthly" in seasonal:
        try:
            m = seasonal["monthly"]
            df_month = pd.DataFrame({
                "month": m.get("time", []),
                "temp_mean": m.get("temperature_2m_mean", []),
                "precip_sum": m.get("precipitation_sum", [])
            })
            df_month = df_month.head(3)
            st.markdown("### 📈 3-Month Seasonal Averages")
            st.dataframe(df_month)
        except:
            st.info("Seasonal forecast not available.")

    prompt = f"""
    Explain this to a farmer in {language}:

    Recommended Crop: {crop}
    Soil Health: {soil_h}
    Fertilizer NPK: {fert}
    Predicted Yield: {y_pred}

    Real-time weather:
    {cur}

    16-day forecast summary:
    {df_16.to_dict(orient='records')}

    Seasonal (3-month) outlook:
    {df_month.to_dict(orient='records')}

    Keep it simple, clear, and helpful.
    """

    explanation = gemini_text(prompt)
    st.session_state._explanation = explanation

    st.markdown("### 🗣️ AI Explanation")
    st.write(explanation)

# =========================================================
# ✅ PDF DOWNLOAD — OUTSIDE ANALYZE BLOCK
# =========================================================
if st.session_state.get("analysis_done", False):

    if st.button("📄 Download PDF Report"):

        r = st.session_state._results
        explanation = st.session_state._explanation

        crop = r["crop"]
        fert = r["fert"]
        soil_h = r["soil"]
        y_pred = r["yield"]
        loc_name = r["loc_name"]
        cc = r["cc"]
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

        # ✅ GENERATE PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "AgriIntel Smart Report", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, f"Language: {language}")
        pdf.multi_cell(0, 8, f"Region: {region} ({loc_name}, {cc})")
        pdf.multi_cell(0, 8, f"Recommended Crop: {crop}")
        pdf.multi_cell(0, 8, f"Predicted Yield: {y_pred} t/ha")
        pdf.multi_cell(0, 8, f"Soil Health: {soil_h}")
        pdf.multi_cell(0, 8, f"Fertilizer (kg/ha): {fert}")

        pdf.ln(5)
        pdf.multi_cell(0, 8,
                       f"Real-time Weather: {cur.get('temperature_2m')}°C, "
                       f"{cur.get('relative_humidity_2m')}% humidity, "
                       f"{cur.get('precipitation')} mm precipitation")

        # 16-day forecast
        if not df_16.empty:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "16-Day Forecast:", ln=True)
            pdf.set_font("Arial", "", 10)
            for _, r2 in df_16.iterrows():
                pdf.multi_cell(0, 6,
                               f"{r2['date']} → Tmax {r2['t_max']}°C | "
                               f"Tmin {r2['t_min']}°C | Precip {r2['precip_mm']} mm")

        # Seasonal
        if not df_month.empty:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "3-Month Seasonal Outlook:", ln=True)
            pdf.set_font("Arial", "", 10)
            for _, r3 in df_month.iterrows():
                pdf.multi_cell(0, 6,
                               f"{r3['month']} → Temp Mean {r3['temp_mean']}°C | "
                               f"Precip Sum {r3['precip_sum']} mm")

        # AI explanation
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "AI Explanation:", ln=True)
        pdf.set_font("Arial", "", 11)
        for line in explanation.split("\n"):
            pdf.multi_cell(0, 6, line)

        # ✅ TEMP FILE FOR STREAMLIT
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            tmp.seek(0)
            st.download_button(
                label="⬇️ Download PDF",
                data=tmp.read(),
                file_name="AgriIntel_Report.pdf",
                mime="application/pdf"
            )
