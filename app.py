# app.py
import os
import io
import requests
import datetime as dt
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from fpdf import FPDF
import google.generativeai as genai

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AgriIntel", layout="wide")

LANGS = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn"
}

# =========================
# GEMINI SETUP
# =========================
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
USE_GEMINI = bool(GEMINI_KEY)
if USE_GEMINI:
    genai.configure(api_key=GEMINI_KEY)
    gemini = genai.GenerativeModel("gemini-pro")

def gemini_text(prompt: str, fallback: str = "") -> str:
    """Generate text via Gemini, fall back gracefully if not configured."""
    if not USE_GEMINI:
        return fallback or prompt
    try:
        resp = gemini.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return fallback or f"(AI explanation unavailable: {e})"

# =========================
# SIMPLE LOGIN (demo)
# =========================
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

# =========================
# MODEL LOADING (Google Drive)
# =========================
# Your confirmed mapping (direct download links):
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
    r = requests.get(url)
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))

with st.spinner("Loading models..."):
    try:
        crop_model = load_drive_model(MODEL_URLS["crop"])
        fert_model = load_drive_model(MODEL_URLS["fert"])
        soil_model = load_drive_model(MODEL_URLS["soil"])
        soil_encoder = load_drive_model(MODEL_URLS["soil_encoder"])
        yield_model = load_drive_model(MODEL_URLS["yield"])
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# =========================
# SIDEBAR INPUTS
# =========================
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

# =========================
# GEO + WEATHER (Open-Meteo)
# =========================
def geocode(place: str):
    """Return lat, lon, name, country_code using Open-Meteo geocoding."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": place, "count": 1})
    r.raise_for_status()
    data = r.json()
    if data.get("results"):
        hit = data["results"][0]
        return hit["latitude"], hit["longitude"], hit.get("name", place), hit.get("country_code", "")
    raise ValueError("Location not found")

def get_realtime_and_daily(lat, lon):
    """Current + 16-day daily forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 16, "timezone": "auto"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def get_seasonal_monthly(lat, lon):
    """3-month seasonal monthly averages using Open-Meteo Seasonal."""
    url = "https://seasonal-api.open-meteo.com/v1/seasonal"
    start = dt.date.today().replace(day=1)
    end = (pd.Timestamp(start) + pd.DateOffset(months=3)).date()
    params = {
        "latitude": lat, "longitude": lon,
        "models": "ecmwf_seas5",
        "monthly": "temperature_2m_mean,precipitation_sum",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "timezone": "auto"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

# =========================
# MODEL HELPERS
# =========================
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
    return {"delta_N": round(pred[0], 2), "delta_P": round(pred[1], 2), "delta_K": round(pred[2], 2)}

def predict_soil():
    pred = soil_model.predict([[N, P, K, ph]])[0]
    return soil_encoder.inverse_transform([pred])[0]

def predict_yield(crop):
    feats = getattr(yield_model, "feature_names_in_", [])
    row = {f: 0 for f in feats}
    vals = {"N": N, "P": P, "K": K, "temperature": temperature,
            "humidity": humidity, "ph": ph, "rainfall": rainfall}
    for k, v in vals.items():
        if k in row:
            row[k] = v
    for f in feats:
        if f == f"crop_{crop}":
            row[f] = 1
    X = pd.DataFrame([row], columns=feats)
    return round(float(yield_model.predict(X)[0]), 2)

# =========================
# RUN
# =========================
if st.button("🔍 Analyze & Fetch Weather"):
    # Weather
    try:
        lat, lon, loc_name, cc = geocode(region)
        w_now = get_realtime_and_daily(lat, lon)
        w_season = get_seasonal_monthly(lat, lon)
    except Exception as e:
        st.error(f"Weather lookup failed: {e}")
        st.stop()

    # Models
    crop = predict_crop()
    fert = predict_fertilizer(crop)
    soil_health = predict_soil()
    yield_pred = predict_yield(crop)

    # Results
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended Crop", crop)
    c2.metric("Predicted Yield", f"{yield_pred} t/ha")
    c3.metric("Soil Health", soil_health)
    c4.metric("Location", f"{loc_name} ({cc})")

    st.markdown("### 🧪 Fertilizer Recommendation (kg/ha)")
    st.table(pd.DataFrame([fert]))

    # Real-time weather
    cur = w_now.get("current", {})
    st.markdown("### 🌤️ Real-time Weather")
    st.write({
        "Temperature (°C)": cur.get("temperature_2m"),
        "Humidity (%)": cur.get("relative_humidity_2m"),
        "Precipitation (mm)": cur.get("precipitation")
    })

    # 16-day forecast
    daily = w_now.get("daily", {})
    df_16 = pd.DataFrame({
        "date": daily.get("time", []),
        "t_max": daily.get("temperature_2m_max", []),
        "t_min": daily.get("temperature_2m_min", []),
        "precip_mm": daily.get("precipitation_sum", [])
    })
    if not df_16.empty:
        st.markdown("### 📅 16-Day Forecast")
        st.dataframe(df_16)

    # 3-month seasonal monthly
    df_month = pd.DataFrame()
    try:
        monthly = w_season.get("monthly", {})
        df_month = pd.DataFrame({
            "month": monthly.get("time", []),
            "temp_mean": monthly.get("temperature_2m_mean", []),
            "precip_sum": monthly.get("precipitation_sum", [])
        })
        if len(df_month) > 3:
            df_month = df_month.head(3)
        st.markdown("### 📈 3-Month Seasonal Averages")
        st.dataframe(df_month)
    except Exception:
        st.info("Seasonal outlook not available for this location right now.")

    # Gemini explanation
    prompt = f"""
    Write a farmer-friendly advisory in {language}.
    Location: {loc_name}, {cc} (lat {lat}, lon {lon})
    Recommended crop: {crop}
    Soil health: {soil_health}
    Fertilizer (kg/ha): N={fert['delta_N']}, P={fert['delta_P']}, K={fert['delta_K']}
    Predicted yield: {yield_pred} t/ha
    Current weather: temp={cur.get('temperature_2m')}°C, humidity={cur.get('relative_humidity_2m')}%, precip={cur.get('precipitation')} mm
    16-day forecast summary: avg tmax={(df_16['t_max'].mean() if not df_16.empty else 'NA')}, total precip={(df_16['precip_mm'].sum() if not df_16.empty else 'NA')} mm
    Next 3 months (monthly averages): {df_month.to_dict(orient='records') if not df_month.empty else 'NA'}
    Keep it concise, clear bullet points, and write in {language}.
    """
    explanation = gemini_text(prompt, fallback="(Add GEMINI_API_KEY in Secrets to enable AI explanation.)")

    st.markdown("### 🗣️ AI Explanation")
    st.write(explanation)

    # PDF Download
    if st.button("📄 Download PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "AgriIntel Smart Report", ln=True, align="C")
        pdf.ln(3)
        pdf.multi_cell(0, 8, f"Language: {language}")
        pdf.multi_cell(0, 8, f"Region: {region} ({loc_name}, {cc})")
        pdf.multi_cell(0, 8, f"Recommended Crop: {crop}")
        pdf.multi_cell(0, 8, f"Predicted Yield: {yield_pred} t/ha")
        pdf.multi_cell(0, 8, f"Soil Health: {soil_health}")
        pdf.multi_cell(0, 8, f"Fertilizer (kg/ha): N={fert['delta_N']}  P={fert['delta_P']}  K={fert['delta_K']}")
        pdf.ln(4)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 7, f"Real-time Weather: T={cur.get('temperature_2m')}°C, RH={cur.get('relative_humidity_2m')}%, Precip={cur.get('precipitation')} mm")

        if not df_16.empty:
            pdf.ln(2)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "16-Day Forecast:", ln=True)
            pdf.set_font("Arial", "", 10)
            for _, r in df_16.iterrows():
                pdf.multi_cell(0, 6, f"{r['date']}: Tmax {r['t_max']}°C, Tmin {r['t_min']}°C, Precip {r['precip_mm']} mm")

        if not df_month.empty:
            pdf.ln(2)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "3-Month Seasonal Outlook:", ln=True)
            pdf.set_font("Arial", "", 10)
            for _, r in df_month.iterrows():
                pdf.multi_cell(0, 6, f"{r['month']}: Temp mean {r['temp_mean']}°C, Precip sum {r['precip_sum']} mm")

        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "AI Explanation:", ln=True)
        pdf.set_font("Arial", "", 11)
        for para in explanation.split("\n"):
            pdf.multi_cell(0, 6, para)

        out_path = "AgriIntel_Report.pdf"
        pdf.output(out_path)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name="AgriIntel_Report.pdf", mime="application/pdf")
