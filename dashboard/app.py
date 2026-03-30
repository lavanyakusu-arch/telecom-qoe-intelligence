"""
app.py  ─  Telecom QoE Intelligence Dashboard
==============================================
Loads:  qoe_model.pkl  (saved by telecom_qoe_model_Reg.ipynb)
        artifact keys: "model", "scaler", "features"

NO model training code lives here.
Run:    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import truncnorm

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QoE Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
CAT_ORDER   = ["Poor", "Fair", "Good", "Excellent"]
CAT_COLORS  = {"Poor": "#ef4444", "Fair": "#f59e0b",
               "Good": "#3b82f6", "Excellent": "#10b981"}
CAT_PALETTE = [CAT_COLORS[c] for c in CAT_ORDER]
APP_COLORS  = {"video": "#4C72B0", "volte": "#DD8452", "gaming": "#55A868"}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e2e8f0"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor="#1e2d45",
                borderwidth=1, font_size=11),
)

# ── Embedded RobustScaler statistics ──────────────────────────────────
#   The feature engineering notebook saved telecom_qoe_features_ml_ready.csv
#   AFTER applying RobustScaler on raw features.
#   The model notebook loaded that pre-scaled CSV and fitted a second scaler
#   on it — which produced near-identity stats (centre≈0, scale≈1).
#   The pkl therefore contains a near-identity SCALER.
#
#   app must apply the ORIGINAL feature engineering scaler transform
#        (embedded below as FE_CENTRES / FE_SCALES) BEFORE calling
#        the pkl's SCALER.transform(). Only then does the model receive
#        the same distribution it was trained on.
#
# These values are the median (centre) and IQR (scale) of each raw
# engineered feature computed on the full 5000-sample training set
# (seed=42, identical to the notebook).
FE_CENTRES = {
    "r_factor_proxy":    79.7107,
    "goodput":           10.2754,
    "goodput_lat_ratio":  0.3467,
    "mos_from_r":         4.0130,
    "packet_loss_sq":     1.9864,
    "video_stall_risk":   0.0000,
    "loss_x_latency":    36.8823,
    "tput_lat_ratio":     0.3523,
    "health_score":       0.5232,
    "log_throughput":     2.4359,
    "log_packet_loss":    0.8794,
    "packet_loss":        1.4094,
    "volte_loss_cliff":   0.0000,
    "video_loss_flag":    0.0000,
    "sinr_x_tput":       95.5955,
    "game_loss_cliff":    0.0000,
    "sinr_per_prb":       0.2093,
    "sqrt_throughput":    3.2289,
    "throughput":        10.4255,
    "tput_tier":          2.0000,
}
FE_SCALES = {
    "r_factor_proxy":     5.0306,
    "goodput":           12.4574,
    "goodput_lat_ratio":  0.6409,
    "mos_from_r":         0.1918,
    "packet_loss_sq":     3.5392,
    "video_stall_risk":   1.0000,
    "loss_x_latency":    65.3577,
    "tput_lat_ratio":     0.6424,
    "health_score":       0.2113,
    "log_throughput":     1.1979,
    "log_packet_loss":    0.5156,
    "packet_loss":        1.2302,
    "volte_loss_cliff":   1.0000,
    "video_loss_flag":    1.0000,
    "sinr_x_tput":      226.6974,
    "game_loss_cliff":    1.0000,
    "sinr_per_prb":       0.3338,
    "sqrt_throughput":    2.0145,
    "throughput":        12.5230,
    "tput_tier":          2.0000,
}

# Fixed global bounds for health_score normalisation
# (using per-batch min/max caused every UE to look average — another bug fix)
RAW_KPI_BOUNDS = {
    "throughput":       (0.5,   50.0),
    "latency":          (5.0,  200.0),
    "packet_loss":      (0.0,    5.0),
    "sinr":            (-5.0,   30.0),
    "rsrp":           (-120.0, -70.0),
    "cell_load":       (10.0,  100.0),
    "prb_utilization": (10.0,  100.0),
    "mobility":         (0.0,  120.0),
}

# ══════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
  :root{--bg:#0a0e1a;--surface:#111827;--surface2:#1a2235;--border:#1e2d45;
        --accent:#00d4ff;--accent2:#7c3aed;--text:#e2e8f0;--muted:#64748b;}
  html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);}
  .stApp{background-color:var(--bg);}
  [data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border);}
  [data-testid="stSidebar"] *{color:var(--text)!important;}
  .metric-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:20px 24px;position:relative;overflow:hidden;}
  .metric-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,var(--accent),var(--accent2));}
  .metric-label{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:2px;
    text-transform:uppercase;color:var(--muted);margin-bottom:8px;}
  .metric-value{font-family:'Space Mono',monospace;font-size:30px;
    font-weight:700;color:var(--accent);line-height:1;}
  .metric-sub{font-size:12px;color:var(--muted);margin-top:6px;}
  .alarm-critical{background:linear-gradient(135deg,rgba(239,68,68,.13),rgba(239,68,68,.04));
    border:1px solid rgba(239,68,68,.45);border-radius:10px;padding:14px 18px;margin:5px 0;}
  .alarm-warning{background:linear-gradient(135deg,rgba(245,158,11,.13),rgba(245,158,11,.04));
    border:1px solid rgba(245,158,11,.45);border-radius:10px;padding:14px 18px;margin:5px 0;}
  .alarm-ok{background:linear-gradient(135deg,rgba(16,185,129,.10),rgba(16,185,129,.03));
    border:1px solid rgba(16,185,129,.35);border-radius:10px;padding:14px 18px;margin:5px 0;}
  .section-title{font-family:'Space Mono',monospace;font-size:11px;letter-spacing:3px;
    text-transform:uppercase;color:var(--accent);border-bottom:1px solid var(--border);
    padding-bottom:8px;margin:24px 0 16px 0;}
  .insight-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:18px 20px;position:relative;overflow:hidden;height:100%;}
  .insight-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;}
  .insight-card.video::before{background:#4C72B0;}
  .insight-card.volte::before{background:#DD8452;}
  .insight-card.gaming::before{background:#55A868;}
  .stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2));
    color:#fff;border:none;border-radius:8px;font-family:'Space Mono',monospace;
    font-size:12px;letter-spacing:1px;padding:10px 28px;font-weight:700;}
  .stButton>button:hover{opacity:.88;}
  .stTabs [data-baseweb="tab-list"]{background:var(--surface);border-radius:10px;
    padding:4px;border:1px solid var(--border);gap:4px;}
  .stTabs [data-baseweb="tab"]{background:transparent;color:var(--muted);
    border-radius:7px;font-family:'Space Mono',monospace;font-size:11px;letter-spacing:1px;}
  .stTabs [aria-selected="true"]{
    background:linear-gradient(135deg,var(--accent),var(--accent2))!important;
    color:white!important;}
  #MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading qoe_model.pkl ...")
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


artifact = load_model("qoe_model.pkl")

if artifact is None:
    st.error("""
    ### 🚫 `qoe_model.pkl` not found
    Run `telecom_qoe_model_Reg.ipynb` to completion, then place
    `qoe_model.pkl` in the same folder as this `app.py`.
    """)
    st.stop()

MODEL    = artifact["model"]
SCALER   = artifact["scaler"]
FEATURES = artifact["features"]


# ══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING + SCALING
# ══════════════════════════════════════════════════════════════════════
def engineer_and_scale(df_raw: pd.DataFrame) -> np.ndarray:
    """
    1. Compute all 20 raw engineered features from raw KPI DataFrame.
    2. Apply FE_CENTRES / FE_SCALES (the original feature engineering scaler).
    3. Apply pkl SCALER.transform() (near-identity, kept for correctness).
    Returns numpy array ready for MODEL.predict().
    """
    d = df_raw.copy().reset_index(drop=True)

    # ── Raw feature computation ────────────────────────────────────────
    d["tput_tier"] = pd.cut(
        d["throughput"], bins=[0, 1, 5, 15, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(float)

    d["goodput"]           = d["throughput"] * (1 - d["packet_loss"] / 100)
    d["goodput_lat_ratio"] = d["goodput"]    / (d["latency"] + 1)
    d["tput_lat_ratio"]    = d["throughput"] / (d["latency"] + 1)
    d["loss_x_latency"]    = d["packet_loss"] * d["latency"]
    d["sinr_x_tput"]       = d["sinr"]        * d["throughput"]
    d["sinr_per_prb"]      = d["sinr"]        / (d["prb_utilization"] + 1)
    d["log_throughput"]    = np.log1p(d["throughput"])
    d["log_packet_loss"]   = np.log1p(d["packet_loss"])
    d["sqrt_throughput"]   = np.sqrt(np.clip(d["throughput"], 0, None))
    d["packet_loss_sq"]    = d["packet_loss"] ** 2

    Id = (0.024 * d["latency"]
          + 0.11 * (d["latency"] - 177.3)
          * (d["latency"] > 177.3).astype(float))
    Ie = 7 + 30 * np.log(1 + 15 * d["packet_loss"] / 100)
    d["r_factor_proxy"] = np.clip(93.2 - Id - Ie, 0, 100)
    d["mos_from_r"]     = np.clip(
        1 + 0.035 * d["r_factor_proxy"]
        + d["r_factor_proxy"] * (d["r_factor_proxy"] - 60)
        * (100 - d["r_factor_proxy"]) * 7e-6,
        1, 5)

    # Health score with FIXED global bounds (not batch min/max)
    RAW_KPI = ["throughput","latency","packet_loss","sinr","rsrp",
               "cell_load","prb_utilization","mobility"]
    km = pd.DataFrame(index=d.index)
    for col in RAW_KPI:
        lo, hi = RAW_KPI_BOUNDS[col]
        km[col] = ((d[col] - lo) / (hi - lo)).clip(0, 1)
    imp = ["latency","packet_loss","cell_load","prb_utilization","mobility"]
    ben = ["throughput","sinr","rsrp"]
    d["health_score"] = (
        0.5 * (1 - km[imp].mean(axis=1))
        + 0.5 *      km[ben].mean(axis=1)
    )

    d["video_stall_risk"] = ((d["app"]=="video") & (d["throughput"]<2.0)).astype(float)
    d["video_loss_flag"]  = ((d["app"]=="video") & (d["packet_loss"]>1.0)).astype(float)
    d["volte_loss_cliff"] = ((d["app"]=="volte") & (d["packet_loss"]>0.5)).astype(float)
    d["game_loss_cliff"]  = ((d["app"]=="gaming")& (d["packet_loss"]>1.5)).astype(float)

    # ── Apply FE RobustScaler (embedded constants) ────────────────────
    X_raw    = d[FEATURES].astype(float)
    centres  = np.array([FE_CENTRES[f] for f in FEATURES])
    scales   = np.array([FE_SCALES[f]  for f in FEATURES])
    X_fe     = (X_raw.values - centres) / scales

    # ── Apply pkl near-identity scaler ────────────────────────────────
    return SCALER.transform(X_fe)


def predict_scores(df_raw: pd.DataFrame) -> np.ndarray:
    return np.clip(MODEL.predict(engineer_and_scale(df_raw)), 1.0, 5.0)


def score_to_category(s: float) -> str:
    if s < 2:   return "Poor"
    elif s < 3: return "Fair"
    elif s < 4: return "Good"
    else:       return "Excellent"


# ══════════════════════════════════════════════════════════════════════
# FLEET GENERATION
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=30)
def generate_fleet(n: int = 1000, seed: int = None) -> pd.DataFrame:
    if seed is None:
        seed = int(time.time()) // 30
    np.random.seed(seed)

    apps = ["video","volte","gaming"]
    app  = np.random.choice(apps, n, p=[0.38, 0.32, 0.30])

    _a, _b = (0-25)/30, (120-25)/30
    mob    = truncnorm.rvs(_a, _b, loc=25, scale=30, size=n)
    mob_n  = mob / 120

    ll   = np.random.random(n) < 0.60
    cl   = np.clip(
        np.where(ll, np.random.normal(35,12,n), np.random.normal(75,10,n)),
        10, 100)
    prb  = np.clip(0.88*cl + np.random.normal(0,5,n), 10, 100)
    sinr = np.clip(np.random.normal(12,7,n) - 0.12*(cl-50) - 8*mob_n, -5, 30)
    rsrp = np.clip(-95 + 0.4*(sinr-12) + np.random.normal(0,6,n) - 10*mob_n, -120,-70)
    sl   = 10**(sinr/10)
    tput = np.clip(np.log2(1+sl)*3.5 - 0.12*(cl-40) + np.random.normal(0,1.5,n), 0.5, 50)
    lat  = np.clip(10/np.clip(1-cl/100,0.05,1) + 20*mob_n + np.random.normal(0,8,n), 5, 200)
    loss = np.clip(
        np.clip(1.5-0.08*sinr,0,3) + 0.025*np.clip(cl-50,0,50)
        + 1.5*mob_n + np.random.normal(0,0.3,n),
        0, 5)

    imsi = [f"310150{np.random.randint(100_000_000,999_999_999)}" for _ in range(n)]
    return pd.DataFrame({
        "imsi":imsi, "app":app,
        "throughput":      np.round(tput,2),
        "latency":         np.round(lat,2),
        "packet_loss":     np.round(loss,2),
        "sinr":            np.round(sinr,2),
        "rsrp":            np.round(rsrp,2),
        "cell_load":       np.round(cl,2),
        "prb_utilization": np.round(prb,2),
        "mobility":        np.round(mob,2),
    })


# ══════════════════════════════════════════════════════════════════════
# ALARM ENGINE
# ══════════════════════════════════════════════════════════════════════
def get_alarms(fleet: pd.DataFrame) -> list:
    alarms = []

    hl = (fleet["cell_load"] > 80).mean() * 100
    if hl > 40:
        alarms.append(("critical","🔴 CELL CONGESTION",
            f"{hl:.0f}% of UEs on overloaded cells (load > 80%)",
            "Trigger load balancing / SON handovers"))
    elif hl > 20:
        alarms.append(("warning","🟡 HIGH CELL LOAD",
            f"{hl:.0f}% of UEs on heavily loaded cells",
            "Monitor — approaching congestion threshold"))

    hi_lat = (fleet["latency"] > 100).mean() * 100
    if hi_lat > 30:
        alarms.append(("critical","🔴 LATENCY DEGRADATION",
            f"{hi_lat:.0f}% of UEs with latency > 100 ms",
            "Check scheduler queue depth and HO parameters"))
    elif hi_lat > 15:
        alarms.append(("warning","🟡 ELEVATED LATENCY",
            f"{hi_lat:.0f}% of UEs with latency > 100 ms",
            "Investigate peak-hour traffic shaping"))

    poor_sinr = (fleet["sinr"] < 0).mean() * 100
    if poor_sinr > 25:
        alarms.append(("critical","🔴 POOR SIGNAL QUALITY",
            f"{poor_sinr:.0f}% of UEs with SINR < 0 dB",
            "Audit antenna tilt, power config & interference"))
    elif poor_sinr > 10:
        alarms.append(("warning","🟡 SIGNAL DEGRADATION",
            f"{poor_sinr:.0f}% of UEs with SINR < 0 dB",
            "Consider coverage optimisation"))

    hi_loss = (fleet["packet_loss"] > 2).mean() * 100
    if hi_loss > 20:
        alarms.append(("critical","🔴 PACKET LOSS SPIKE",
            f"{hi_loss:.0f}% of UEs with packet loss > 2%",
            "Inspect backhaul and retransmission buffers"))
    elif hi_loss > 10:
        alarms.append(("warning","🟡 ELEVATED PACKET LOSS",
            f"{hi_loss:.0f}% of UEs with packet loss > 2%",
            "Review QoS markings and PDCP config"))

    poor_qoe = fleet["qoe_category"].isin(["Poor","Fair"]).mean() * 100
    if poor_qoe > 35:
        alarms.append(("critical","🔴 WIDESPREAD POOR QoE",
            f"{poor_qoe:.0f}% of subscribers with Poor/Fair experience",
            "Escalate — immediate capacity / optimisation action needed"))
    elif poor_qoe > 20:
        alarms.append(("warning","🟡 DEGRADED USER EXPERIENCE",
            f"{poor_qoe:.0f}% of subscribers with Poor/Fair experience",
            "Review per-app thresholds and PRB allocation"))

    v = fleet[fleet["app"]=="volte"]
    if len(v) > 0:
        vc = ((v["latency"]>150)|(v["packet_loss"]>2.5)).mean()*100
        if vc > 15:
            alarms.append(("critical","🔴 VoLTE E-MODEL CLIFF",
                f"{vc:.0f}% of VoLTE calls in unacceptable quality zone",
                "Prioritise VoLTE QCI bearer — verify IMS config"))

    g = fleet[fleet["app"]=="gaming"]
    if len(g) > 0:
        gc = (g["latency"]>120).mean()*100
        if gc > 25:
            alarms.append(("warning","🟡 GAMING LATENCY CLIFF",
                f"{gc:.0f}% of gaming UEs exceeding 120 ms latency",
                "Check scheduling priority for gaming bearers"))

    if not alarms:
        alarms.append(("ok","✅ ALL SYSTEMS NOMINAL",
            "All network KPIs within acceptable thresholds",
            "Continue monitoring"))
    return alarms


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px;">
      <div style="font-family:'Space Mono',monospace;font-size:22px;font-weight:700;
           background:linear-gradient(135deg,#00d4ff,#7c3aed);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        📡 QoE Intelligence
      </div>
      <div style="font-size:10px;letter-spacing:3px;color:#64748b;margin-top:4px;">
        TELECOM ANALYTICS PLATFORM
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation",
                    ["📊 QoE Analytics","🔍 UE Insights"],
                    label_visibility="collapsed")
    st.markdown("---")

    n_features = len(FEATURES)
    model_type = type(MODEL).__name__
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;
                padding:14px 16px;font-size:12px;color:#94a3b8;line-height:1.9;">
      <div style="font-family:'Space Mono',monospace;font-size:10px;
                  letter-spacing:2px;color:#64748b;margin-bottom:8px;">MODEL OVERVIEW</div>
      <b style="color:#e2e8f0;">File</b>&nbsp; qoe_model.pkl<br>
      <b style="color:#e2e8f0;">Type</b>&nbsp; {model_type}<br>
      <b style="color:#e2e8f0;">Features</b>&nbsp; {n_features} selected<br>
      <b style="color:#e2e8f0;">R²</b>&nbsp; 0.9630&nbsp;<b style="color:#10b981;">✓</b><br>
      <b style="color:#e2e8f0;">RMSE</b>&nbsp; 0.1878
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — FLEET ANALYTICS
# ══════════════════════════════════════════════════════════════════════
if page == "📊 QoE Analytics":

    st.markdown("""
    <div style="margin-bottom:4px;">
      <span style="font-family:'Space Mono',monospace;font-size:22px;
                   font-weight:700;color:#00d4ff;">Network Performance Overview</span>
      <span style="font-size:13px;color:#64748b;margin-left:12px;">
        1,000 UE live snapshot — QoE predicted via saved regression model
      </span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        refresh = st.button("🔄  Refresh Fleet", use_container_width=True)
    with c2:
        app_filter = st.multiselect("App",["video","volte","gaming"],
            default=["video","volte","gaming"], label_visibility="collapsed")
    with c3:
        st.markdown(f"""
        <div style="padding:8px 14px;background:#111827;border:1px solid #1e2d45;
                    border-radius:8px;font-size:12px;color:#64748b;line-height:1.6;">
          Auto-refresh 30s &nbsp;·&nbsp; {model_type} &nbsp;·&nbsp;
          {n_features} features &nbsp;·&nbsp; R²=0.9630
        </div>
        """, unsafe_allow_html=True)

    seed      = None if refresh else int(time.time()) // 30
    raw_fleet = generate_fleet(1000, seed=seed)

    with st.spinner("Running model inference on 1,000 UEs ..."):
        scores = predict_scores(raw_fleet)
        raw_fleet["qoe_score"]    = np.round(scores, 2)
        raw_fleet["qoe_category"] = [score_to_category(s) for s in scores]

    fleet = (raw_fleet[raw_fleet["app"].isin(app_filter)]
             if app_filter else raw_fleet).copy()

    # ── KPI Cards ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Fleet KPI Overview</div>',
                unsafe_allow_html=True)
    m1,m2,m3,m4,m5 = st.columns(5)
    for col, label, value, sub in [
        (m1,"AVG QoE SCORE",   f"{fleet['qoe_score'].mean():.2f}",                             "out of 5.00"),
        (m2,"POOR / FAIR UEs", f"{fleet['qoe_category'].isin(['Poor','Fair']).mean()*100:.0f}%","below Good"),
        (m3,"AVG THROUGHPUT",  f"{fleet['throughput'].mean():.1f}",                             "Mbps"),
        (m4,"AVG LATENCY",     f"{fleet['latency'].mean():.0f}",                                "ms"),
        (m5,"CONGESTED CELLS", f"{(fleet['cell_load']>80).mean()*100:.0f}%",                   "load > 80%"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Alarms ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚡ Network Condition Alarms</div>',
                unsafe_allow_html=True)
    for severity, title, detail, action in get_alarms(fleet):
        css = {"critical":"alarm-critical","warning":"alarm-warning","ok":"alarm-ok"}[severity]
        st.markdown(f"""
        <div class="{css}">
          <div style="display:flex;justify-content:space-between;align-items:start;">
            <div>
              <div style="font-family:'Space Mono',monospace;font-size:12px;
                          font-weight:700;margin-bottom:3px;">{title}</div>
              <div style="font-size:13px;color:#cbd5e1;">{detail}</div>
            </div>
            <div style="font-size:11px;color:#94a3b8;text-align:right;
                        max-width:240px;padding-left:16px;flex-shrink:0;">
              💡 {action}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Distribution ───────────────────────────────────────────
    st.markdown('<div class="section-title">QoE Distribution & Application Breakdown</div>',
                unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        fig = go.Figure()
        for app_name, color in APP_COLORS.items():
            if app_name not in app_filter: continue
            sub = fleet[fleet["app"]==app_name]["qoe_score"]
            fig.add_trace(go.Histogram(x=sub, name=app_name.title(),
                marker_color=color, opacity=0.75, xbins=dict(size=0.12)))
        fig.update_layout(**PLOT_LAYOUT, height=280, barmode="overlay",
            title="QoE Score Distribution by App",
            xaxis=dict(title="QOE Score",color="#64748b",range=[1,5]),
            yaxis=dict(title="Count",color="#64748b",gridcolor="#1a2235"))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        cat_counts = fleet["qoe_category"].value_counts().reindex(CAT_ORDER,fill_value=0)
        fig = go.Figure(go.Pie(labels=cat_counts.index, values=cat_counts.values,
            hole=0.62, marker_colors=CAT_PALETTE, textinfo="label+percent",
            textfont=dict(size=11,family="DM Sans"),
            hovertemplate="%{label}: %{value} UEs (%{percent})<extra></extra>"))
        fig.add_annotation(text=f"<b>{len(fleet)}</b><br>UEs",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16,color="#e2e8f0",family="Space Mono"))
        fig.update_layout(**PLOT_LAYOUT, height=280, title="QoE Category Split")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: KPI Deep-Dive ─────────────────────────────────────────
    st.markdown('<div class="section-title">KPI Deep-Dive by Application</div>',
                unsafe_allow_html=True)
    ch3, ch4 = st.columns(2)

    with ch3:
        ct     = fleet.groupby(["app","qoe_category"]).size().unstack(fill_value=0)
        ct_pct = ct.div(ct.sum(1),axis=0)*100
        fig    = go.Figure()
        for cat in CAT_ORDER:
            if cat not in ct_pct.columns: continue
            fig.add_trace(go.Bar(name=cat, x=ct_pct.index, y=ct_pct[cat],
                marker_color=CAT_COLORS[cat], marker_line_width=0))
        fig.update_layout(**PLOT_LAYOUT, height=280, barmode="stack",
            title="QoE Category % by App", xaxis=dict(color="#64748b"),
            yaxis=dict(title="%",color="#64748b",gridcolor="#1a2235",range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)

    with ch4:
        fig = go.Figure()
        for app_name, color in APP_COLORS.items():
            if app_name not in app_filter: continue
            sub = fleet[fleet["app"]==app_name]["qoe_score"]
            fig.add_trace(go.Box(y=sub, name=app_name.title(),
                marker_color=color, line_color=color,
                fillcolor = f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.2)",
                boxpoints="outliers", jitter=0.3, marker_size=3))
        fig.update_layout(**PLOT_LAYOUT, height=320,
            title="QoE Score Distribution by App",
            yaxis=dict(title="QOE Score",color="#64748b",
                       gridcolor="#1a2235",range=[1,5]),
            xaxis=dict(color="#64748b"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Network Analysis ───────────────────────────────────────
    st.markdown('<div class="section-title">Network Condition Analysis</div>',
                unsafe_allow_html=True)
    ch6, ch7 = st.columns(2)

    with ch6:
        fig = go.Figure()
        for app_name, color in APP_COLORS.items():
            if app_name not in app_filter: continue
            sub = fleet[fleet["app"]==app_name]
            fig.add_trace(go.Scatter(x=sub["latency"], y=sub["qoe_score"],
                mode="markers", name=app_name.title(),
                marker=dict(color=color,size=5,opacity=0.45,line=dict(width=0))))
        fig.update_layout(**PLOT_LAYOUT, height=300, title="Latency vs QoE Score",
            xaxis=dict(title="Latency (ms)",color="#64748b",gridcolor="#1a2235"),
            yaxis=dict(title="QoE Score",color="#64748b",
                       gridcolor="#1a2235",range=[1,5]))
        st.plotly_chart(fig, use_container_width=True)

    with ch7:
        ft = fleet.copy()
        ft["load_bin"] = pd.cut(ft["cell_load"],
            bins=[10,30,50,70,90,100],
            labels=["10–30","30–50","50–70","70–90","90–100"])
        pivot = ft.pivot_table(values="qoe_score",index="app",
                               columns="load_bin",aggfunc="mean")
        pivot = pivot.reindex(
            index=[a for a in ["video","volte","gaming"] if a in app_filter])
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
            colorscale=[[0,"#ef4444"],[0.33,"#f59e0b"],
                        [0.67,"#3b82f6"],[1,"#10b981"]],
            zmin=1, zmax=5, text=np.round(pivot.values,2),
            texttemplate="%{text}", textfont=dict(size=12,family="Space Mono"),
            colorbar=dict(title="QOE",tickfont=dict(color="#94a3b8")
                          )))
        fig.update_layout(**PLOT_LAYOUT, height=300,
            title="QoE Impact by Application and Cell Load",
            xaxis=dict(title="Cell Load (%)",color="#64748b"),
            yaxis=dict(color="#64748b"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-App Insights ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Per-Application Insights</div>',
                unsafe_allow_html=True)
    app_icons  = {"video":"📹","volte":"📞","gaming":"🎮"}
    apps_shown = app_filter or ["video","volte","gaming"]
    icols      = st.columns(len(apps_shown))
    for col, app_name in zip(icols, apps_shown):
        sub    = fleet[fleet["app"]==app_name]
        avg_q  = sub["qoe_score"].mean()
        poor_f = sub["qoe_category"].isin(["Poor","Fair"]).mean()*100
        if   sub["latency"].mean()     > 80: concern = "⚠️ High latency"
        elif sub["packet_loss"].mean() > 1:  concern = "⚠️ Elevated packet loss"
        elif sub["sinr"].mean()        < 5:  concern = "⚠️ Weak signal"
        elif sub["cell_load"].mean()   > 70: concern = "⚠️ Congestion risk"
        else:                                concern = "✅ KPIs nominal"
        with col:
            st.markdown(f"""
            <div class="insight-card {app_name}">
              <div class="metric-label">
                {app_icons.get(app_name,'')} {app_name.upper()}
              </div>
              <div style="font-family:'Space Mono',monospace;font-size:26px;
                          font-weight:700;color:{APP_COLORS[app_name]};line-height:1;">
                {avg_q:.2f}
              </div>
              <div style="font-size:12px;color:#64748b;margin-top:4px;">
                Avg QOE · {len(sub)} UEs
              </div>
              <div style="margin-top:12px;font-size:12px;color:#94a3b8;line-height:1.8;">
                Poor/Fair: <b style="color:#f59e0b;">{poor_f:.0f}%</b><br>
                Avg Latency: {sub['latency'].mean():.0f} ms<br>
                Avg Tput: {sub['throughput'].mean():.1f} Mbps<br>
                Avg Pkt Loss: {sub['packet_loss'].mean():.2f}%<br>
                <span style="font-size:11px;">{concern}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Worst 50 Table ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Subscriber Detail — Worst 50 UEs by QoE</div>',
                unsafe_allow_html=True)
    worst50 = fleet.nsmallest(50,"qoe_score")[[
        "imsi","app","qoe_score","qoe_category",
        "throughput","latency","packet_loss","sinr","cell_load","mobility"
    ]].reset_index(drop=True)

    def _color_cat(val):
        return {"Poor":"background-color:#7f1d1d;color:#fca5a5",
                "Fair":"background-color:#78350f;color:#fde68a",
                "Good":"background-color:#1e3a5f;color:#93c5fd",
                "Excellent":"background-color:#064e3b;color:#6ee7b7"}.get(val,"")

    styled = (worst50.style
        .applymap(_color_cat, subset=["qoe_category"])
        .format({"qoe_score":"{:.2f}","throughput":"{:.1f}","latency":"{:.0f}",
                 "packet_loss":"{:.2f}","sinr":"{:.1f}",
                 "cell_load":"{:.0f}","mobility":"{:.0f}"})
        .set_properties(**{"background-color":"#111827","color":"#e2e8f0",
                           "border-color":"#1e2d45"}))
    st.dataframe(styled, use_container_width=True, height=380)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — UE INSPECTOR
# ══════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div style="margin-bottom:4px;">
      <span style="font-family:'Space Mono',monospace;font-size:22px;
                   font-weight:700;color:#00d4ff;">UE Inspector</span>
      <span style="font-size:13px;color:#64748b;margin-left:12px;">
        Examine individual UE KPIs
      </span>
    </div>
    """, unsafe_allow_html=True)

    raw_fleet = generate_fleet(1000, seed=int(time.time())//30)
    scores    = predict_scores(raw_fleet)
    raw_fleet["qoe_score"]    = np.round(scores, 2)
    raw_fleet["qoe_category"] = [score_to_category(s) for s in scores]

    col_pick, col_detail = st.columns([1, 2], gap="large")

    with col_pick:
        st.markdown('<div class="section-title">Select UE</div>',
                    unsafe_allow_html=True)
        app_sel = st.selectbox("App",["All","video","volte","gaming"],
                               label_visibility="collapsed")
        cat_sel = st.selectbox("Category",["All","Poor","Fair","Good","Excellent"],
                               label_visibility="collapsed")
        subset  = raw_fleet.copy()
        if app_sel != "All": subset = subset[subset["app"]==app_sel]
        if cat_sel != "All": subset = subset[subset["qoe_category"]==cat_sel]
        subset  = subset.sort_values("qoe_score").reset_index(drop=True)

        if len(subset) == 0:
            st.warning("No UEs match filters.")
            st.stop()

        ue_idx = st.slider("UE rank (worst → best)", 0, len(subset)-1, 0)
        ue_row = subset.iloc[ue_idx]
        sc_col = CAT_COLORS[ue_row["qoe_category"]]

        st.markdown(f"""
        <div class="metric-card" style="margin-top:12px;">
          <div class="metric-label">SELECTED UE</div>
          <div style="font-family:'Space Mono',monospace;font-size:10px;
                      color:#64748b;margin-bottom:8px;">{ue_row['imsi']}</div>
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="font-family:'Space Mono',monospace;font-size:36px;
                        font-weight:700;color:{sc_col};">
              {ue_row['qoe_score']:.2f}
            </div>
            <div style="background:{sc_col};color:#fff;border-radius:20px;
                        padding:4px 14px;font-size:13px;font-weight:600;">
              {ue_row['qoe_category']}
            </div>
          </div>
          <div style="margin-top:10px;font-size:12px;color:#94a3b8;line-height:1.8;">
            App: <b style="color:#e2e8f0;">{ue_row['app'].upper()}</b><br>
            Throughput: {ue_row['throughput']:.1f} Mbps<br>
            Latency: {ue_row['latency']:.0f} ms<br>
            Packet Loss: {ue_row['packet_loss']:.2f}%<br>
            SINR: {ue_row['sinr']:.1f} dB<br>
            Cell Load: {ue_row['cell_load']:.0f}%<br>
            Mobility: {ue_row['mobility']:.0f} km/h
          </div>
        </div>
        """, unsafe_allow_html=True)
