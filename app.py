# app.py — AgriIntel (Simple UI + Full Professional PDF + Gemini 1.5 Pro)
# - Minimal UI similar to your reference
# - Multilingual labels and Gemini translation
# - 4 local ML models (crop, fertilizer, soil, yield) loaded from Google Drive
# - A3 PDF with cover, TOC, summary, weather, charts (2 pages), detailed advisory, QR code
# - Unicode-safe PDF (DejaVu), safe margins (no overflow), wrapped text, page numbers

import os
import io
import textwrap
import tempfile
import datetime as dt
import requests
import pandas as pd
import numpy as np
import joblib
import streamlit as st

from fpdf import FPDF
import matplotlib.pyplot as plt
from matplotlib import font_manager
import qrcode

import google.generativeai as genai

# -------------------------------
# Basic Settings
# -------------------------------
APP_NAME = "AgriIntel Premium"
PAGE_ICON = "🌱"
st.set_page_config(page_title=APP_NAME, page_icon=PAGE_ICON, layout="wide")

# -------------------------------
# Multilingual UI dictionary
# -------------------------------
LANG = {
    "English": {
        "title": "Smart Agriculture Recommendation System",
        "subtitle": "Enter your farm's data to receive a comprehensive analysis powered by our AI and Google Gemini.",
        "sidebar_header": "Enter Sensor & Climate Data",
        "n_label": "Nitrogen (N) Content (kg/ha)",
        "p_label": "Phosphorus (P) Content (kg/ha)",
        "k_label": "Potassium (K) Content (kg/ha)",
        "temp_label": "Temperature (°C)",
        "humidity_label": "Humidity (%)",
        "ph_label": "Soil pH",
        "rainfall_label": "Rainfall (mm)",
        "region_label": "Region / Place",
        "button_text": "Generate Comprehensive Report",
        "report_header": "Your Comprehensive Agricultural Report",
        "spinner_text": "Analyzing data and generating your report with Google Gemini...",
        "info_text": "Please enter your farm's data in the sidebar and click 'Generate Comprehensive Report'.",
        "login_demo": "Demo login: admin@agriintel.app / agriintel123",
        "login_title": "Secure Login",
        "logout": "Logout",
    },
    "हिन्दी (Hindi)": {
        "title": "स्मार्ट कृषि सिफारिश प्रणाली",
        "subtitle": "हमारे AI और Google Gemini द्वारा संचालित व्यापक विश्लेषण के लिए अपने खेत का डेटा दर्ज करें।",
        "sidebar_header": "सेंसर और जलवायु डेटा दर्ज करें",
        "n_label": "नाइट्रोजन (N) (किग्रा/हेक्टेयर)",
        "p_label": "फॉस्फोरस (P) (किग्रा/हेक्टेयर)",
        "k_label": "पोटेशियम (K) (किग्रा/हेक्टेयर)",
        "temp_label": "तापमान (°C)",
        "humidity_label": "आर्द्रता (%)",
        "ph_label": "मिट्टी का pH",
        "rainfall_label": "वर्षा (मिमी)",
        "region_label": "क्षेत्र / स्थान",
        "button_text": "व्यापक रिपोर्ट तैयार करें",
        "report_header": "आपकी व्यापक कृषि रिपोर्ट",
        "spinner_text": "डेटा का विश्लेषण और Google Gemini से आपकी रिपोर्ट तैयार की जा रही है...",
        "info_text": "कृपया साइडबार में डेटा दर्ज करें और 'व्यापक रिपोर्ट तैयार करें' पर क्लिक करें।",
        "login_demo": "डेमो लॉगिन: admin@agriintel.app / agriintel123",
        "login_title": "सुरक्षित लॉगिन",
        "logout": "लॉगआउट",
    },
    "मराठी (Marathi)": {
        "title": "स्मार्ट कृषी शिफारस प्रणाली",
        "subtitle": "आमच्या AI आणि Google Gemini च्या सहाय्याने सर्वसमावेशक विश्लेषणासाठी तुमच्या शेताचा डेटा भरा.",
        "sidebar_header": "सेन्सर व हवामान माहिती",
        "n_label": "नायट्रोजन (N) (कि.ग्रा./हे)",
        "p_label": "फॉस्फरस (P) (कि.ग्रा./हे)",
        "k_label": "पोटॅशियम (K) (कि.ग्रा./हे)",
        "temp_label": "तापमान (°C)",
        "humidity_label": "आर्द्रता (%)",
        "ph_label": "मातीचा pH",
        "rainfall_label": "पर्जन्यमान (मिमी)",
        "region_label": "प्रदेश / ठिकाण",
        "button_text": "सर्वसमावेशक अहवाल तयार करा",
        "report_header": "तुमचा सर्वसमावेशक कृषी अहवाल",
        "spinner_text": "डेटाचे विश्लेषण चालू आहे व Google Gemini सह अहवाल तयार होत आहे...",
        "info_text": "कृपया साइडबारमध्ये डेटा भरा व 'सर्वसमावेशक अहवाल तयार करा' क्लिक करा.",
        "login_demo": "डेमो लॉगिन: admin@agriintel.app / agriintel123",
        "login_title": "सुरक्षित लॉगिन",
        "logout": "लॉगआऊट",
    },
    "தமிழ் (Tamil)": {
        "title": "ஸ்மார்ட் விவசாய பரிந்துரை அமைப்பு",
        "subtitle": "எங்கள் AI மற்றும் Google Gemini மூலம் விரிவான பகுப்பாய்விற்காக உங்களின் பண்ணை தரவை உள்ளிடவும்.",
        "sidebar_header": "சென்சார் மற்றும் காலநிலை தரவு",
        "n_label": "நைட்ரஜன் (N) (கி.கி/ஹெ)",
        "p_label": "பாஸ்பரஸ் (P) (கி.கி/ஹெ)",
        "k_label": "பொட்டாசியம் (K) (கி.கி/ஹெ)",
        "temp_label": "வெப்பநிலை (°C)",
        "humidity_label": "ஈரப்பதம் (%)",
        "ph_label": "மண் pH",
        "rainfall_label": "மழைப்பொழிவு (மிமீ)",
        "region_label": "பகுதி / இடம்",
        "button_text": "விரிவான அறிக்கையை உருவாக்கவும்",
        "report_header": "உங்கள் விரிவான விவசாய அறிக்கை",
        "spinner_text": "தரவை பகுப்பாய்வு செய்து Google Gemini மூலம் அறிக்கை உருவாக்கப்படுகிறது...",
        "info_text": "சைட்பாரில் தரவை உள்ளிட்டு 'விரிவான அறிக்கையை உருவாக்கவும்' என்பதைக் கிளிக் செய்யவும்.",
        "login_demo": "டெமோ லாகின்: admin@agriintel.app / agriintel123",
        "login_title": "பாதுகாப்பான உள்ளுகை",
        "logout": "வெளியேறு",
    },
    "తెలుగు (Telugu)": {
        "title": "స్మార్ట్ వ్యవసాయ సిఫార్సు వ్యవస్థ",
        "subtitle": "మా AI మరియు Google Gemini సహాయంతో సమగ్ర విశ్లేషణ కోసం మీ ఫార్మ్ డేటాను నమోదు చేయండి.",
        "sidebar_header": "సెన్సర్ మరియు వాతావరణ డేటా",
        "n_label": "నత్రజని (N) (కిలో/హెక్టారు)",
        "p_label": "భాస్వరం (P) (కిలో/హెక్టారు)",
        "k_label": "పొటాషియం (K) (కిలో/హెక్టారు)",
        "temp_label": "ఉష్ణోగ్రత (°C)",
        "humidity_label": "ఆర్ద్రత (%)",
        "ph_label": "నేల pH",
        "rainfall_label": "వర్షపాతం (మిమీ)",
        "region_label": "ప్రాంతం / స్థలం",
        "button_text": "సమగ్ర నివేదిక రూపొందించండి",
        "report_header": "మీ సమగ్ర వ్యవసాయ నివేదిక",
        "spinner_text": "డేటా విశ్లేషణ జరుగుతోంది మరియు Google Gemini ద్వారా నివేదిక సిద్ధమవుతోంది...",
        "info_text": "దయచేసి సైడ్బార్‌లో డేటా నమోదు చేసి 'సమగ్ర నివేదిక రూపొందించండి' క్లిక్ చేయండి.",
        "login_demo": "డెమో లాగిన్: admin@agriintel.app / agriintel123",
        "login_title": "సురక్షిత లాగిన్",
        "logout": "లాగ్ అవుట్",
    },
    "বাংলা (Bengali)": {
        "title": "স্মার্ট কৃষি সুপারিশ ব্যবস্থা",
        "subtitle": "আমাদের AI এবং Google Gemini দ্বারা চালিত একটি বিস্তৃত বিশ্লেষণের জন্য আপনার খামারের ডেটা দিন।",
        "sidebar_header": "সেন্সর এবং জলবায়ু ডেটা",
        "n_label": "নাইট্রোজেন (N) (কেজি/হেক্টর)",
        "p_label": "ফসফরাস (P) (কেজি/হেক্টর)",
        "k_label": "পটাশিয়াম (K) (কেজি/হেক্টর)",
        "temp_label": "তাপমাত্রা (°C)",
        "humidity_label": "আর্দ্রতা (%)",
        "ph_label": "মাটির pH",
        "rainfall_label": "বৃষ্টিপাত (মিমি)",
        "region_label": "অঞ্চল / স্থান",
        "button_text": "বিস্তারিত রিপোর্ট তৈরি করুন",
        "report_header": "আপনার বিস্তারিত কৃষি রিপোর্ট",
        "spinner_text": "ডেটা বিশ্লেষণ করা হচ্ছে এবং Google Gemini দিয়ে রিপোর্ট তৈরি করা হচ্ছে...",
        "info_text": "সাইডবারে ডেটা দিয়ে 'বিস্তারিত রিপোর্ট তৈরি করুন' ক্লিক করুন।",
        "login_demo": "ডেমো লগইন: admin@agriintel.app / agriintel123",
        "login_title": "নিরাপদ লগইন",
        "logout": "লগআউট",
    },
    "ಕನ್ನಡ (Kannada)": {
        "title": "ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಶಿಫಾರಸು ವ್ಯವಸ್ಥೆ",
        "subtitle": "ನಮ್ಮ AI ಮತ್ತು Google Gemini ಮೂಲಕ ಸಮಗ್ರ ವಿಶ್ಲೇಷಣೆಗೆ ನಿಮ್ಮ ಕೃಷಿ ಡೇಟಾವನ್ನು ನಮೂದಿಸಿ.",
        "sidebar_header": "ಸೆನ್ಸಾರ್ ಮತ್ತು ಹವಾಮಾನ ಡೇಟಾ",
        "n_label": "ನೈಟ್ರಜನ (N) (ಕೆಜಿ/ಹೆ)",
        "p_label": "ಫಾಸ್ಫರಸ್ (P) (ಕೆಜಿ/ಹೆ)",
        "k_label": "ಪೊಟ್ಯಾಸಿಯಂ (K) (ಕೆಜಿ/ಹೆ)",
        "temp_label": "ತಾಪಮಾನ (°C)",
        "humidity_label": "ಆರ್ದ್ರತೆ (%)",
        "ph_label": "ಮಣ್ಣಿನ pH",
        "rainfall_label": "ಮಳೆ (ಮಿ.ಮೀ)",
        "region_label": "ಪ್ರದೇಶ / ಸ್ಥಳ",
        "button_text": "ವಿಸ್ತೃತ ವರದಿ ರಚಿಸಿ",
        "report_header": "ನಿಮ್ಮ ವಿಸ್ತೃತ ಕೃಷಿ ವರದಿ",
        "spinner_text": "ಡೇಟಾ ವಿಶ್ಲೇಷಣೆ ನಡೆಯುತ್ತಿದೆ ಮತ್ತು Google Gemini ಮೂಲಕ ವರದಿ ರಚಿಸಲಾಗುತ್ತಿದೆ...",
        "info_text": "ಸೈಡ್‌ಬಾರ್‌ನಲ್ಲಿ ಡೇಟಾ ನಮೂದಿಸಿ 'ವಿಸ್ತೃತ ವರದಿ ರಚಿಸಿ' ಕ್ಲಿಕ್ ಮಾಡಿ.",
        "login_demo": "ಡೆಮೋ ಲಾಗಿನ್: admin@agriintel.app / agriintel123",
        "login_title": "ಸುರಕ್ಷಿತ ಲಾಗಿನ್",
        "logout": "ಲಾಗ್ ಔಟ್",
    },
}

# -------------------------------
# Gemini setup (model: gemini-1.5-pro)
# -------------------------------
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
USE_GEMINI = bool(GEMINI_KEY)
if USE_GEMINI:
    genai.configure(api_key=GEMINI_KEY)
    GEM_MODEL = genai.GenerativeModel("gemini-1.5-pro")
else:
    GEM_MODEL = None

def gemini_report(user_data, model_predictions, language_display):
    if not GEM_MODEL:
        return "(Gemini not configured. Add GEMINI_API_KEY in Streamlit secrets.)"
    # translate instruction
    translate = ""
    if language_display != "English":
        translate = f"\n\nTranslate the ENTIRE final report into **{language_display}** only. Do NOT mix English with {language_display}. Use correct local agricultural terms."

    prompt = f"""
You are a senior agricultural scientist and agronomist. Prepare a complete farmer-friendly advisory report.

FARM INPUTS:
- Nitrogen: {user_data['N']} kg/ha
- Phosphorus: {user_data['P']} kg/ha
- Potassium: {user_data['K']} kg/ha
- Soil pH: {user_data['ph']}
- Temperature: {user_data['temperature']} °C
- Humidity: {user_data['humidity']} %
- Rainfall: {user_data['rainfall']} mm
- Region: {user_data['region']}

LOCAL AI PREDICTIONS:
- Soil Health: {model_predictions['soil_health_status']}
- Recommended Crop: {model_predictions['recommended_crop']}
- Fertilizer Inputs (kg/ha): {model_predictions['fertilizer_products']}
- Expected Yield (t/ha): {model_predictions.get('yield_prediction','NA')}

Write a detailed report in clean Markdown with these sections:

### 1. Executive Summary
### 2. Detailed Soil Health Analysis
### 3. Crop Recommendation & Rationale
### 4. Actionable Fertilizer Plan (with schedule, split doses, method of application)
### 5. Irrigation & Pest Management Guidance (based on weather outlook)
### 6. Long-Term Soil Improvement Practices
### 7. Estimated Yield Potential & Risk Factors

Keep it concise, actionable, and easy to follow for farmers.
{translate}
"""
    try:
        resp = GEM_MODEL.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"(Gemini error: {e})"

# -------------------------------
# Authentication (simple demo)
# -------------------------------
def login_block(lang):
    st.subheader(lang["login_title"])
    st.caption(lang["login_demo"])
    with st.form("login_form"):
        email = st.text_input("Email", "")
        pwd = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")
        if ok:
            if email == "admin@agriintel.app" and pwd == "agriintel123":
                st.session_state.auth = True
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")

if "auth" not in st.session_state:
    st.session_state.auth = True  # set True if you don't want login for now

# -------------------------------
# Model loading (Google Drive URLs)
# -------------------------------
MODEL_URLS = {
    "yield": "https://drive.google.com/uc?export=download&id=1EMwJ9wr_s5yMvRtpDTkP4Va2csniqfSv",
    "soil_encoder": "https://drive.google.com/uc?export=download&id=10fo75uk_uY6fYPcUZTXd-6AqolelWwDe",
    "soil": "https://drive.google.com/uc?export=download&id=1tQcpfJ3M8s3m5fuXVZ3ZrKuAWyfMrLhm",
    "fert": "https://drive.google.com/uc?export=download&id=16lWBeuxyKF1FjvIgka8fGEteadqEgrHc",
    "crop": "https://drive.google.com/uc?export=download&id=10y_phgu-8AV-gdH2K47TqOAw37L7vr-b",
}

@st.cache_resource(show_spinner=True)
def load_drive_model(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))

with st.spinner("Loading ML models…"):
    try:
        CROP_MODEL = load_drive_model(MODEL_URLS["crop"])
        FERT_MODEL = load_drive_model(MODEL_URLS["fert"])
        SOIL_MODEL = load_drive_model(MODEL_URLS["soil"])
        SOIL_ENCODER = load_drive_model(MODEL_URLS["soil_encoder"])
        YIELD_MODEL = load_drive_model(MODEL_URLS["yield"])
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# -------------------------------
# Weather / geocoding (Open-Meteo)
# -------------------------------
@st.cache_data(show_spinner=False)
def geocode(place: str):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": place, "count": 1}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("results"):
        d = data["results"][0]
        return d["latitude"], d["longitude"], d["name"], d["country_code"]
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

# -------------------------------
# Local model helpers
# -------------------------------
def predict_crop(N,P,K,temperature,humidity,ph,rainfall):
    X = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    return CROP_MODEL.predict(X)[0]

def predict_fertilizer(crop, N,P,K):
    feats = getattr(FERT_MODEL, "feature_names_in_", [])
    row = {f:0 for f in feats}
    for k,v in {"N":N,"P":P,"K":K}.items():
        if k in row: row[k] = v
    for f in feats:
        if f == f"crop_{crop}":
            row[f] = 1
    X = pd.DataFrame([row], columns=feats)
    pred = FERT_MODEL.predict(X)[0]
    return {"delta_N": round(pred[0],2), "delta_P": round(pred[1],2), "delta_K": round(pred[2],2)}

def predict_soil(N,P,K,ph):
    pred = SOIL_MODEL.predict([[N,P,K,ph]])[0]
    return SOIL_ENCODER.inverse_transform([pred])[0]

def predict_yield(crop, N,P,K,temperature,humidity,ph,rainfall):
    feats = getattr(YIELD_MODEL, "feature_names_in_", [])
    row = {f:0 for f in feats}
    for k,v in {"N":N,"P":P,"K":K,"temperature":temperature,"humidity":humidity,"ph":ph,"rainfall":rainfall}.items():
        if k in row: row[k]=v
    for f in feats:
        if f == f"crop_{crop}":
            row[f]=1
    X = pd.DataFrame([row], columns=feats)
    return round(float(YIELD_MODEL.predict(X)[0]),2)

# -------------------------------
# Charts (matplotlib)
# -------------------------------
def save_fig(fig):
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return path

def chart_16day_temp(df):
    if df.empty: return ""
    fig, ax = plt.subplots(figsize=(9,3.4))
    ax.plot(pd.to_datetime(df["date"]), df["t_max"], label="Tmax (°C)")
    ax.plot(pd.to_datetime(df["date"]), df["t_min"], label="Tmin (°C)")
    ax.set_title("16-Day Temperature Forecast")
    ax.set_ylabel("°C")
    ax.legend()
    fig.autofmt_xdate()
    return save_fig(fig)

def chart_16day_rain(df):
    if df.empty: return ""
    fig, ax = plt.subplots(figsize=(9,3.4))
    ax.bar(pd.to_datetime(df["date"]), df["precip_mm"])
    ax.set_title("16-Day Rainfall (mm)")
    ax.set_ylabel("mm")
    fig.autofmt_xdate()
    return save_fig(fig)

def chart_seasonal_temp(dfm):
    if dfm.empty: return ""
    fig, ax = plt.subplots(figsize=(9,3.4))
    ax.plot(pd.to_datetime(dfm["month"]), dfm["temp_mean"], marker="o")
    ax.set_title("3-Month Seasonal Mean Temperature (°C)")
    ax.set_ylabel("°C")
    fig.autofmt_xdate()
    return save_fig(fig)

def chart_npk(n,p,k):
    fig, ax = plt.subplots(figsize=(6,3.4))
    ax.bar(["N","P","K"], [n,p,k])
    ax.set_title("NPK Recommendation (kg/ha)")
    return save_fig(fig)

# -------------------------------
# PDF helpers (A3, safe width, Unicode)
# -------------------------------
def wrap_text(text, width=120):
    lines=[]
    for para in (text or "").splitlines():
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width, break_long_words=True, break_on_hyphens=False))
    return lines

class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", size=10)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

def build_pdf(full, ui_lang):
    """
    full: dict with everything needed
    """
    # A3 portrait, hard-safe width
    pdf = PDF(format="A3")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Unicode DejaVu
    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    SAFE_WIDTH = 250  # 297 - 40 = 257; keep extra safety

    # -------- Cover Page --------
    pdf.set_font("DejaVu", size=28)
    pdf.cell(SAFE_WIDTH, 16, APP_NAME, ln=True, align="C")
    pdf.set_font("DejaVu", size=18)
    pdf.cell(SAFE_WIDTH, 12, "Advisory Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("DejaVu", size=12)
    today = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.multi_cell(SAFE_WIDTH, 8, f"Region: {full['region']}")
    pdf.multi_cell(SAFE_WIDTH, 8, f"Location: {full['loc_name']}, {full['cc']}")
    pdf.multi_cell(SAFE_WIDTH, 8, f"Coordinates: {full['lat']}, {full['lon']}")
    pdf.multi_cell(SAFE_WIDTH, 8, f"Generated: {today}")
    pdf.ln(6)
    for seg in wrap_text("Unified crop, fertilizer, yield & soil insights with live weather, seasonal outlook, and AI advisory.", 115):
        pdf.multi_cell(SAFE_WIDTH, 7, seg)

    # -------- Table of Contents --------
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "Table of Contents", ln=True)
    pdf.set_font("DejaVu", size=12)
    toc = [
        "1. Summary",
        "2. Weather (Now)",
        "3. Forecast Charts (16-Day Temp & Rain)",
        "4. Seasonal & Nutrition Charts",
        "5. Detailed AI Advisory",
        "6. QR Code"
    ]
    for idx, item in enumerate(toc, 1):
        pdf.multi_cell(SAFE_WIDTH, 7, f"{idx}. {item}")

    # -------- Summary --------
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "1. Summary", ln=True)
    pdf.set_font("DejaVu", size=12)
    fert_line = (
        f"N={full['fert']['delta_N']} kg/ha | "
        f"P={full['fert']['delta_P']} kg/ha ({'reduce' if full['fert']['delta_P']<0 else 'apply'}) | "
        f"K={full['fert']['delta_K']} kg/ha"
    )
    pdf.multi_cell(SAFE_WIDTH, 7, f"Recommended Crop: {full['crop']}")
    pdf.multi_cell(SAFE_WIDTH, 7, f"Predicted Yield: {full['yield']} t/ha")
    pdf.multi_cell(SAFE_WIDTH, 7, f"Soil Health: {full['soil']}")
    pdf.multi_cell(SAFE_WIDTH, 7, f"Fertilizer Plan: {fert_line}")

    # -------- Weather Now --------
    pdf.ln(4)
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "2. Weather (Now)", ln=True)
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(SAFE_WIDTH, 7, f"Temperature: {full['cur'].get('temperature_2m')} °C")
    pdf.multi_cell(SAFE_WIDTH, 7, f"Humidity: {full['cur'].get('relative_humidity_2m')} %")
    pdf.multi_cell(SAFE_WIDTH, 7, f"Precipitation: {full['cur'].get('precipitation')} mm")

    # -------- Charts Page 1: 16-day --------
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "3. Forecast Charts (16-Day)", ln=True)
    if full["chart_temp"] and os.path.exists(full["chart_temp"]):
        pdf.ln(2); pdf.image(full["chart_temp"], w=SAFE_WIDTH)
    if full["chart_rain"] and os.path.exists(full["chart_rain"]):
        pdf.ln(4); pdf.image(full["chart_rain"], w=SAFE_WIDTH)

    # -------- Charts Page 2: Seasonal & NPK --------
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "4. Seasonal & Nutrition Charts", ln=True)
    if full["chart_season"] and os.path.exists(full["chart_season"]):
        pdf.ln(2); pdf.image(full["chart_season"], w=SAFE_WIDTH)
    if full["chart_npk"] and os.path.exists(full["chart_npk"]):
        pdf.ln(4); pdf.image(full["chart_npk"], w=SAFE_WIDTH/2)

    # -------- Detailed AI Advisory --------
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "5. Detailed AI Advisory", ln=True)
    pdf.set_font("DejaVu", size=11)
    for seg in wrap_text(full["advisory"], 120):
        pdf.multi_cell(SAFE_WIDTH, 6.5, seg)

    # -------- QR Code --------
    pdf.add_page()
    pdf.set_font("DejaVu", size=16); pdf.cell(SAFE_WIDTH, 10, "6. QR Code", ln=True)
    pdf.set_font("DejaVu", size=11)
    app_url = os.getenv("APP_URL", "https://share.streamlit.io/")
    pdf.multi_cell(SAFE_WIDTH, 7, f"Scan to open the app: {app_url}")
    # make QR
    qr_img = qrcode.make(app_url)
    qr_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    qr_img.save(qr_path)
    pdf.ln(4); pdf.image(qr_path, w=80)

    # Output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp.seek(0)
        return tmp.read()

# -------------------------------
# UI (simple, like your reference)
# -------------------------------
selected_language = st.sidebar.selectbox("Choose Language / भाषा", options=list(LANG.keys()))
L = LANG[selected_language]

st.title(L["title"])
st.markdown(L["subtitle"])

st.sidebar.header(L["sidebar_header"])
region = st.sidebar.text_input(L["region_label"], "Pune, India")

user_inputs = {
    'N': st.sidebar.number_input(L['n_label'], 0, 300, 90),
    'P': st.sidebar.number_input(L['p_label'], 0, 300, 40),
    'K': st.sidebar.number_input(L['k_label'], 0, 300, 40),
    'temperature': st.sidebar.number_input(L['temp_label'], -10.0, 60.0, 25.0, format="%.2f"),
    'humidity': st.sidebar.number_input(L['humidity_label'], 0.0, 100.0, 75.0, format="%.2f"),
    'ph': st.sidebar.number_input(L['ph_label'], 0.0, 14.0, 6.5, format="%.2f"),
    'rainfall': st.sidebar.number_input(L['rainfall_label'], 0.0, 1000.0, 200.0, format="%.2f"),
    'region': region
}

if st.sidebar.button(L["button_text"]):
    # Compute local predictions
    crop = predict_crop(**user_inputs)
    fert = predict_fertilizer(crop, user_inputs['N'], user_inputs['P'], user_inputs['K'])
    soil = predict_soil(user_inputs['N'], user_inputs['P'], user_inputs['K'], user_inputs['ph'])
    ypred = predict_yield(crop, **user_inputs)

    # Weather
    try:
        lat, lon, loc_name, cc = geocode(region)
        rt = get_realtime_and_daily(lat, lon)
        seasonal = None
        try:
            seasonal = get_seasonal_monthly(lat, lon)
        except Exception:
            seasonal = None
    except Exception as e:
        st.error(f"Weather lookup failed: {e}")
        lat=lon=None; loc_name=region; cc=""

    # Forecast frames
    daily = (rt or {}).get("daily", {}) if 'rt' in locals() else {}
    df16 = pd.DataFrame({
        "date": daily.get("time", []),
        "t_max": daily.get("temperature_2m_max", []),
        "t_min": daily.get("temperature_2m_min", []),
        "precip_mm": daily.get("precipitation_sum", [])
    })
    dfm = pd.DataFrame()
    if seasonal and "monthly" in seasonal:
        m = seasonal["monthly"]
        dfm = pd.DataFrame({
            "month": m.get("time", []),
            "temp_mean": m.get("temperature_2m_mean", []),
            "precip_sum": m.get("precipitation_sum", [])
        }).head(3)

    # Gemini advisory (strict translation to chosen language)
    local_report = {
        "soil_health_status": soil,
        "recommended_crop": crop,
        "fertilizer_products": fert,
        "yield_prediction": ypred
    }
    with st.spinner(L["spinner_text"]):
        advisory = gemini_report(user_inputs, local_report, selected_language)

    # Show quick page report
    st.subheader(L["report_header"])
    st.write({
        "Recommended Crop": crop,
        "Soil Health": soil,
        "Fertilizer (kg/ha)": {
            "N": fert["delta_N"],
            "P": f"{fert['delta_P']} (reduce)" if fert["delta_P"] < 0 else fert["delta_P"],
            "K": fert["delta_K"]
        },
        "Predicted Yield (t/ha)": ypred
    })
    st.markdown("### AI Advisory")
    st.markdown(advisory)

    # Build charts
    path_temp = chart_16day_temp(df16)
    path_rain = chart_16day_rain(df16)
    path_season = chart_seasonal_temp(dfm)
    path_npk = chart_npk(fert["delta_N"], fert["delta_P"], fert["delta_K"])

    # Prepare full dict for PDF
    full = {
        "region": region,
        "loc_name": loc_name,
        "cc": cc,
        "lat": lat,
        "lon": lon,
        "crop": crop,
        "soil": soil,
        "fert": fert,
        "yield": ypred,
        "cur": (rt or {}).get("current", {}),
        "chart_temp": path_temp,
        "chart_rain": path_rain,
        "chart_season": path_season,
        "chart_npk": path_npk,
        "advisory": advisory
    }

    # Generate PDF bytes and offer download
    pdf_bytes = build_pdf(full, selected_language)
    st.download_button(
        "⬇️ Download Professional PDF (A3)",
        data=pdf_bytes,
        file_name=f"{APP_NAME.replace(' ','_')}_Report_A3.pdf",
        mime="application/pdf"
    )
else:
    st.info(L["info_text"])
