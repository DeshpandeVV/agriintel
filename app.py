# ============================
# AGRIINTEL – SIMPLE UI + FULL PRO REPORT (A3)
# ============================

import os, io, tempfile, requests, datetime as dt, textwrap
import joblib, qrcode, numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF
from matplotlib import font_manager
import google.generativeai as genai

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AgriIntel", layout="wide", page_icon="🌱")

# ---------------------------
# LANG MAP
# ---------------------------
LANG = {
    "English": {
        "title": "Smart Agriculture Recommendation System",
        "sidebar": "Farm Inputs",
        "region": "Region / Place",
        "n": "Nitrogen (N)",
        "p": "Phosphorus (P)",
        "k": "Potassium (K)",
        "temp": "Temperature (°C)",
        "hum": "Humidity (%)",
        "ph": "Soil pH",
        "rain": "Rainfall (mm)",
        "gen": "Generate Comprehensive Report",
        "rep": "Your Comprehensive Report",
        "wait": "Processing with AI...",
        "info": "Enter data and click Generate.",
    },
    "मराठी (Marathi)": {
        "title": "स्मार्ट कृषी शिफारस प्रणाली",
        "sidebar": "शेतीची माहिती",
        "region": "प्रदेश / ठिकाण",
        "n": "नायट्रोजन (N)",
        "p": "फॉस्फरस (P)",
        "k": "पोटॅशियम (K)",
        "temp": "तापमान (°C)",
        "hum": "आर्द्रता (%)",
        "ph": "मातीचा pH",
        "rain": "पर्जन्यमान (mm)",
        "gen": "अहवाल तयार करा",
        "rep": "तुमचा कृषी अहवाल",
        "wait": "AI सह विश्लेषण चालू...",
        "info": "माहिती भरा आणि Generate क्लिक करा.",
    }
}

# ---------------------------
# GEMINI SETUP
# ---------------------------
KEY = st.secrets.get("GEMINI_API_KEY", "")
if KEY:
    genai.configure(api_key=KEY)
    GEM = genai.GenerativeModel("gemini-1.5-pro")
else:
    GEM = None

def gemini_advisory(data, preds, lang):
    if not GEM:
        return "(Gemini API key missing)"
    trans = "" if lang == "English" else f"\nTranslate final report into {lang} only."

    prompt = f"""
You are an expert agronomist. Prepare a structured, highly actionable farming report.

FARM DATA:
{data}

AI PREDICTIONS:
{preds}

Sections required:
1. Executive Summary
2. Soil Health Analysis
3. Crop Recommendation Rationale
4. Fertilizer Plan (with quantities + schedule)
5. Irrigation & Weather-based Tips
6. Long-term Soil Management
7. Expected Yield & Risks

Keep it farmer-friendly.{trans}
"""
    try:
        r = GEM.generate_content(prompt)
        return r.text.strip()
    except Exception as e:
        return f"(Gemini error: {e})"

# ---------------------------
# MODEL LOADING
# ---------------------------
URLS = {
    "crop": "https://drive.google.com/uc?export=download&id=10y_phgu-8AV-gdH2K47TqOAw37L7vr-b",
    "fert": "https://drive.google.com/uc?export=download&id=16lWBeuxyKF1FjvIgka8fGEteadqEgrHc",
    "soil": "https://drive.google.com/uc?export=download&id=1tQcpfJ3M8s3m5fuXVZ3ZrKuAWyfMrLhm",
    "soil_enc": "https://drive.google.com/uc?export=download&id=10fo75uk_uY6fYPcUZTXd-6AqolelWwDe",
    "yield": "https://drive.google.com/uc?export=download&id=1EMwJ9wr_s5yMvRtpDTkP4Va2csniqfSv",
}

@st.cache_resource
def load(url):
    r = requests.get(url)
    return joblib.load(io.BytesIO(r.content))

CROP = load(URLS["crop"])
FERT = load(URLS["fert"])
SOIL = load(URLS["soil"])
SOIL_ENC = load(URLS["soil_enc"])
YIELD = load(URLS["yield"])

# ---------------------------
# WEATHER API
# ---------------------------
def geocode(q):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": q, "count": 1})
    d = r.json()
    if not d.get("results"):
        raise ValueError("Location not found")
    x = d["results"][0]
    return x["latitude"], x["longitude"], x["name"], x["country_code"]

def weather(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 16,
        "timezone": "auto",
    })
    return r.json()

def seasonal(lat, lon):
    start = dt.date.today().replace(day=1)
    end = (pd.Timestamp(start)+pd.DateOffset(months=3)).date()
    r = requests.get("https://seasonal-api.open-meteo.com/v1/seasonal", params={
        "latitude": lat, "longitude": lon,
        "models": "ecmwf_seas5",
        "monthly": "temperature_2m_mean,precipitation_sum",
        "start_date": start, "end_date": end,
        "timezone": "auto"
    })
    return r.json()

# ---------------------------
# CHART HELPERS
# ---------------------------
def save_fig(fig):
    p = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p

def chart_temp(df):
    if df.empty: return ""
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(pd.to_datetime(df["date"]), df["t_max"], label="Tmax")
    ax.plot(pd.to_datetime(df["date"]), df["t_min"], label="Tmin")
    ax.legend(); ax.set_title("16-Day Temperature")
    return save_fig(fig)

def chart_rain(df):
    if df.empty: return ""
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(pd.to_datetime(df["date"]), df["precip"])
    ax.set_title("Rainfall (mm)")
    return save_fig(fig)

def chart_season(df):
    if df.empty: return ""
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(pd.to_datetime(df["month"]), df["temp"], marker="o")
    ax.set_title("Seasonal Mean Temp")
    return save_fig(fig)

def chart_npk(n,p,k):
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(["N","P","K"], [n,p,k])
    ax.set_title("NPK Recommendation")
    return save_fig(fig)

# ---------------------------
# PDF GENERATION
# ---------------------------
def wrap(t, w=120):
    o=[]
    for l in t.split("\n"):
        o+=textwrap.wrap(l, w) if l.strip() else [""]
    return o

class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", size=10)
        self.cell(0,10,f"Page {self.page_no()}",0,0,"C")

def make_pdf(rep):
    pdf = PDF(format="A3")
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_page()

    path = font_manager.findfont("DejaVu Sans")
    pdf.add_font("DejaVu","",path,uni=True)
    pdf.set_font("DejaVu",size=14)
    W = 250

    # COVER
    pdf.set_font("DejaVu",size=26)
    pdf.cell(W,12,"AgriIntel Premium Report",ln=True,align="C")
    pdf.set_font("DejaVu",size=16)
    pdf.cell(W,10,rep["loc"],ln=True,align="C")
    pdf.ln(10)

    pdf.set_font("DejaVu",size=12)
    pdf.multi_cell(W,7,f"Region: {rep['region']}")
    pdf.multi_cell(W,7,f"Coordinates: {rep['lat']}, {rep['lon']}")
    pdf.multi_cell(W,7,f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # TOC
    pdf.add_page()
    pdf.set_font("DejaVu",size=18)
    pdf.cell(W,10,"Table of Contents",ln=True)
    toc=["1. Summary","2. Weather Now","3. Forecast Charts","4. Seasonal & NPK Charts","5. AI Advisory","6. QR Code"]
    pdf.set_font("DejaVu",size=12)
    for t in toc: pdf.multi_cell(W,7,t)

    # SUMMARY
    pdf.add_page()
    pdf.set_font("DejaVu",size=18)
    pdf.cell(W,10,"1. Summary",ln=True)
    pdf.set_font("DejaVu",size=12)
    pdf.multi_cell(W,7,f"Recommended Crop: {rep['crop']}")
    pdf.multi_cell(W,7,f"Soil Health: {rep['soil']}")
    pdf.multi_cell(W,7,f"Yield: {rep['yield']} t/ha")

    f=rep["fert"]
    P=f['P']
    ptxt=f"P={P} kg/ha (reduce)" if P<0 else f"P={P} kg/ha"
    pdf.multi_cell(W,7,f"N={f['N']} | {ptxt} | K={f['K']}")

    # WEATHER
    pdf.add_page()
    pdf.set_font("DejaVu",size=18); pdf.cell(W,10,"2. Weather Now",ln=True)
    c=rep["cur"]
    pdf.set_font("DejaVu",size=12)
    pdf.multi_cell(W,7,f"Temp: {c.get('temp')} °C")
    pdf.multi_cell(W,7,f"Humidity: {c.get('hum')} %")
    pdf.multi_cell(W,7,f"Rain: {c.get('rain')} mm")

    # CHART PAGE 1
    pdf.add_page()
    pdf.set_font("DejaVu",size=18); pdf.cell(W,10,"3. Forecast Charts",ln=True)
    if os.path.exists(rep["temp_chart"]):
        pdf.image(rep["temp_chart"],w=W)
    if os.path.exists(rep["rain_chart"]):
        pdf.ln(4); pdf.image(rep["rain_chart"],w=W)

    # CHART PAGE 2
    pdf.add_page()
    pdf.set_font("DejaVu",size=18); pdf.cell(W,10,"4. Seasonal & NPK Charts",ln=True)
    if os.path.exists(rep["season_chart"]):
        pdf.image(rep["season_chart"],w=W)
    if os.path.exists(rep["npk_chart"]):
        pdf.ln(4); pdf.image(rep["npk_chart"],w=W/2)

    # AI ADVISORY
    pdf.add_page()
    pdf.set_font("DejaVu",size=18); pdf.cell(W,10,"5. AI Advisory",ln=True)
    pdf.set_font("DejaVu",size=11)
    for line in wrap(rep["ai"]):
        pdf.multi_cell(W,6.5,line)

    # QR
    pdf.add_page()
    pdf.set_font("DejaVu",size=18); pdf.cell(W,10,"6. QR Code",ln=True)
    pdf.set_font("DejaVu",size=12)
    url="https://share.streamlit.io/"
    pdf.multi_cell(W,7,url)

    img=qrcode.make(url)
    qtmp=tempfile.NamedTemporaryFile(delete=False,suffix=".png").name
    img.save(qtmp)
    pdf.ln(4); pdf.image(qtmp,w=80)

    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as f:
        pdf.output(f.name); f.seek(0)
        return f.read()

# ---------------------------
# UI
# ---------------------------
sel = st.sidebar.selectbox("Language", list(LANG.keys()))
L=LANG[sel]

st.title(L["title"])

st.sidebar.header(L["sidebar"])
region = st.sidebar.text_input(L["region"], "Pune, India")
N = st.sidebar.number_input(L["n"], 0,300,90)
P = st.sidebar.number_input(L["p"], 0,300,40)
K = st.sidebar.number_input(L["k"], 0,300,40)
T = st.sidebar.number_input(L["temp"], -10.0,60.0,25.0)
H = st.sidebar.number_input(L["hum"], 0.0,100.0,75.0)
PH = st.sidebar.number_input(L["ph"], 0.0,14.0,6.5)
R = st.sidebar.number_input(L["rain"], 0.0,1000.0,200.0)

if st.sidebar.button(L["gen"]):

    # MODEL INPUTS FIX
    fields={"N":N,"P":P,"K":K,"temperature":T,"humidity":H,"ph":PH,"rainfall":R}

    # local preds
    crop = CROP.predict([[N,P,K,T,H,PH,R]])[0]
    fert_raw = FERT.predict(pd.DataFrame([{
        f:fields.get(f,0) if f in ["N","P","K"] else (1 if f==f"crop_{crop}" else 0)
        for f in FERT.feature_names_in_
    }]))[0]
    fert = {"N":round(fert_raw[0],2),"P":round(fert_raw[1],2),"K":round(fert_raw[2],2)}

    soil = SOIL_ENC.inverse_transform([SOIL.predict([[N,P,K,PH]])[0]])[0]
    ypred = round(float(YIELD.predict(pd.DataFrame([{
        f:fields.get(f,0) if f in fields else (1 if f==f"crop_{crop}" else 0)
        for f in YIELD.feature_names_in_
    }]))[0]),2)

    # Weather
    try:
        lat,lon,name,cc = geocode(region)
        rt = weather(lat,lon)
        seas = seasonal(lat,lon)
    except:
        lat=lon=None; name=region; cc=""; rt={"current":{}}

    c = rt.get("current",{})
    current = {
        "temp": c.get("temperature_2m"),
        "hum": c.get("relative_humidity_2m"),
        "rain": c.get("precipitation")
    }

    # forecast frames
    d = rt.get("daily",{})
    df16 = pd.DataFrame({
        "date": d.get("time",[]),
        "t_max": d.get("temperature_2m_max",[]),
        "t_min": d.get("temperature_2m_min",[]),
        "precip": d.get("precipitation_sum",[])
    })
    dfm=pd.DataFrame()
    if seas and "monthly" in seas:
        m=seas["monthly"]
        dfm=pd.DataFrame({
            "month":m.get("time",[]),
            "temp":m.get("temperature_2m_mean",[])
        }).head(3)

    # charts
    temp_chart = chart_temp(df16)
    rain_chart = chart_rain(df16)
    season_chart = chart_season(dfm)
    npk_chart = chart_npk(fert["N"], fert["P"], fert["K"])

    # AI
    data_txt=f"N={N}, P={P}, K={K}, Temp={T}, Hum={H}, pH={PH}, Rain={R}, Region={region}"
    pred_txt=f"Soil={soil}, Crop={crop}, Fert={fert}, Yield={ypred}"
    with st.spinner(L["wait"]):
        ai = gemini_advisory(data_txt, pred_txt, sel)

    st.subheader(L["rep"])
    st.write({"Crop":crop,"Soil":soil,"Fertilizer":fert,"Yield":ypred})
    st.markdown(ai)

    rep = {
        "region":region,
        "loc":f"{name} ({cc})",
        "lat":lat,"lon":lon,
        "crop":crop,"soil":soil,"yield":ypred,
        "fert":fert,"cur":current,
        "temp_chart":temp_chart,
        "rain_chart":rain_chart,
        "season_chart":season_chart,
        "npk_chart":npk_chart,
        "ai":ai,
    }

    pdf_bytes = make_pdf(rep)

    st.download_button(
        "📄 Download A3 Professional PDF",
        pdf_bytes,
        "AgriIntel_Report.pdf",
        mime="application/pdf"
    )

else:
    st.info(L["info"])
