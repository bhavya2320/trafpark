import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from distance_engine import get_route_distances, DistanceInputError
import streamlit as st

st.title("CrowdSense AI 🚀")
st.write("App is running successfully!")
# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CrowdSense AI | Traffic & Parking Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS  — explicit colors everywhere, zero wildcards
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* === BASE === */
html, body { background-color: #F0F2F6; font-family: 'Inter', sans-serif; }
.stApp     { background-color: #F0F2F6; }

/* === MAIN AREA TEXT — always dark === */
.stApp p,
.stApp li,
.stApp span,
.stApp label,
.stApp div { color: #1a1a1a; font-family: 'Inter', sans-serif; }

/* === HEADINGS === */
.stApp h1 { color: #111827 !important; font-weight: 800; font-size: 2rem; }
.stApp h2 { color: #111827 !important; font-weight: 700; }
.stApp h3 { color: #111827 !important; font-weight: 700; }
.stApp h4 { color: #111827 !important; font-weight: 600; }

/* === SIDEBAR BACKGROUND === */
[data-testid="stSidebar"] > div:first-child {
    background-color: #1E3A8A;
    padding: 20px 16px;
}

/* === SIDEBAR LABELS (white on dark blue) === */
[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* === SIDEBAR MARKDOWN TEXT === */
[data-testid="stSidebar"] .stMarkdown p {
    color: #BFDBFE !important;
    font-size: 13px;
}

/* === SIDEBAR TEXT INPUTS (white bg, dark text) === */
[data-testid="stSidebar"] input[type="text"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1.5px solid #93C5FD !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
}
[data-testid="stSidebar"] input[type="text"]::placeholder {
    color: #9CA3AF !important;
}

/* === SIDEBAR SELECTBOX (white bg, dark text) === */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #93C5FD !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #111827 !important;
    font-size: 14px !important;
}

/* === DROPDOWN OPTIONS (always visible) === */
[data-baseweb="popover"],
[data-baseweb="popover"] * {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}
[data-baseweb="menu"] li {
    color: #111827 !important;
    background-color: #FFFFFF !important;
}
[data-baseweb="menu"] li:hover {
    background-color: #DBEAFE !important;
    color: #1E3A8A !important;
}

/* === SIDEBAR BUTTON (amber, dark text) === */
[data-testid="stSidebar"] .stButton > button {
    background-color: #F59E0B !important;
    color: #1a1a1a !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 4px 14px rgba(245,158,11,0.4) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #D97706 !important;
}

/* === KPI METRIC CARDS === */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
[data-testid="stMetricLabel"] > div {
    color: #6B7280 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] > div {
    color: #111827 !important;
    font-size: 1.45rem !important;
    font-weight: 800 !important;
}

/* === ALERT BOXES (warning/info/error/success) === */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}
[data-testid="stAlert"] p {
    color: #1a1a1a !important;
    font-weight: 500 !important;
}

/* === DATAFRAME TABLE === */
[data-testid="stDataFrameContainer"] {
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    overflow: hidden;
}

/* === PROGRESS BAR === */
[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, #1E3A8A, #3B82F6) !important;
}

/* === DIVIDER === */
hr { border-color: #E5E7EB !important; margin: 16px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
AREA_WEIGHTS = {
    "Commercial":  0.90,
    "Residential": 0.40,
    "Mall":        0.85,
    "College":     0.60,
    "Hospital":    0.70,
}
TIME_WEIGHTS = {
    "6–9 AM":       0.95,
    "9–12 PM":      0.75,
    "12–3 PM":      0.50,
    "3–6 PM":       0.70,
    "6–9 PM":       0.90,
    "9 PM–12 AM":   0.30,
}
# dist_km values below are defaults used ONLY when both geocoding and OSRM fail.
# They are replaced at runtime by get_route_distances() for every real analysis.
ROUTE_PROFILES = [
    {"name": "Route A — Main Highway",    "density": 0.80, "dist_km": 12.5},
    {"name": "Route B — Sub-Arterial Rd", "density": 0.50, "dist_km": 14.2},
    {"name": "Route C — Neighbourhood",   "density": 0.30, "dist_km": 15.8},
]

# ─────────────────────────────────────────────
#  SCORING
# ─────────────────────────────────────────────
def compute_scores(area, time_slot, density):
    aw = AREA_WEIGHTS.get(area, 0.55)
    tw = TIME_WEIGHTS.get(time_slot, 0.55)
    congestion = float(np.clip((aw*0.4 + tw*0.5 + density*0.1)*100, 5, 98))
    parking    = float(np.clip(100 - (aw*0.6 + tw*0.4)*100 + np.random.randint(-4,5), 5, 95))
    return round(congestion,1), round(parking,1)

def c_label(s):
    if s < 40:  return "🟢 Low",       "low",    "#16A34A", "#DCFCE7", "#166534"
    if s < 72:  return "🟡 Medium",    "medium", "#F59E0B", "#FEF3C7", "#92400E"
    return          "🔴 High",         "high",   "#DC2626", "#FEE2E2", "#991B1B"

def p_label(s):
    if s > 60:  return "🟢 Available", "low",    "#16A34A", "#DCFCE7", "#166534"
    if s > 30:  return "🟡 Limited",   "medium", "#F59E0B", "#FEF3C7", "#92400E"
    return          "🔴 Full",         "high",   "#DC2626", "#FEE2E2", "#991B1B"

def badge(text, bg, border, fg):
    return (f"<span style='background:{bg};color:{fg};border:1px solid {border};"
            f"border-radius:20px;padding:4px 14px;font-weight:700;font-size:0.88rem;'>{text}</span>")

def card(title, value, sub=""):
    return f"""
<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:14px;
            padding:18px 22px;box-shadow:0 2px 8px rgba(0,0,0,0.07);height:100%;'>
  <div style='color:#6B7280;font-size:12px;font-weight:600;
              text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;'>{title}</div>
  <div style='color:#111827;font-size:1.4rem;font-weight:800;line-height:1.2;'>{value}</div>
  <div style='color:#6B7280;font-size:12px;margin-top:4px;'>{sub}</div>
</div>"""

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:12px;'>
      <div style='font-size:1.5rem;font-weight:800;color:#FFFFFF;'>🚦 CrowdSense AI</div>
      <div style='font-size:0.8rem;color:#93C5FD;margin-top:2px;'>Traffic & Parking Predictor</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#3B5998;margin:10px 0 18px;'>", unsafe_allow_html=True)

    src       = st.text_input("📍 Source Location",      placeholder="e.g. MG Road")
    dest      = st.text_input("🏁 Destination Location", placeholder="e.g. Whitefield")
    area      = st.selectbox("🏙️ Area Type",  list(AREA_WEIGHTS.keys()))
    time_slot = st.selectbox("🕒 Time Slot",  list(TIME_WEIGHTS.keys()))

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze Traffic", use_container_width=True)

    st.markdown("<hr style='border-color:#3B5998;margin:18px 0 10px;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#93C5FD;font-size:0.73rem;text-align:center;'>Powered by CrowdSense AI Engine v2.0</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:24px;'>
  <h1 style='color:#111827;font-weight:800;font-size:2rem;margin-bottom:4px;'>
    🚀 CrowdSense AI Dashboard
  </h1>
  <p style='color:#6B7280;font-size:1rem;margin:0;'>
    Intelligent Traffic &amp; Parking Prediction — Rule-Based Simulated Intelligence
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────────
if analyze_btn:
    src_c  = src.strip()
    dest_c = dest.strip()

    if not src_c or not dest_c:
        st.warning("⚠️ Please enter **both** Source and Destination locations.")
        st.stop()
    if src_c.lower() == dest_c.lower():
        st.error("❌ Source and Destination cannot be the same. Please enter a valid route.")
        st.stop()

    # ── Dynamic distance computation ─────────────────────────
    live_distances = None
    dist_source    = "default"
    try:
        with st.spinner("📡 Computing real-world route distances…"):
            live_distances, dist_source = get_route_distances(src_c, dest_c)
    except DistanceInputError as _err:
        st.warning(f"⚠️ Location input issue: {_err}. Showing profile-based estimates.")

    if live_distances is None:
        # Geocoding or validation failed — fall back to route profile defaults
        st.info(
            "⚠️ Could not resolve one or both locations to coordinates. "
            "Showing estimated distances based on standard route profiles."
        )
        live_distances = [r["dist_km"] for r in ROUTE_PROFILES]
        dist_source    = "default"

    # Build route profiles with real distances
    live_profiles = [
        {**r, "dist_km": d}
        for r, d in zip(ROUTE_PROFILES, live_distances)
    ]

    # Distance source badge (subtle info line — no UI layout change)
    _source_labels = {
        "google":    "🗺️ Verified distance (Google Distance Matrix)",
        "mapbox":    "🗺️ Verified distance (Mapbox Directions)",
        "osrm":      "🛰️ Road-network distance (OSRM)",
        "haversine": "📐 Estimated distance (Haversine fallback)",
        "cache":     "⚡ Cached verified distance",
        "default":   "📋 Profile-based distance estimates",
    }
    # Strip any 'invalid_input:' prefix from dist_source for label lookup
    _src_key = dist_source.split(":")[0].strip()
    st.caption(_source_labels.get(_src_key, ""))

    # Compute
    rows = []
    for r in live_profiles:
        cs, ps   = compute_scores(area, time_slot, r["density"])
        delay    = round(cs / 4.0, 1)
        ttime    = round((r["dist_km"]/40.0)*60 + delay, 1)
        rows.append({
            "Route": r["name"],
            "Distance (km)": r["dist_km"],
            "Congestion (%)": cs,
            "Parking Avail. (%)": ps,
            "Delay (min)": delay,
            "Total Time (min)": ttime,
        })

    df   = pd.DataFrame(rows)
    best = df.loc[df["Total Time (min)"].idxmin()]

    ct, _, cc, cbg, cfg = c_label(best["Congestion (%)"])
    pt, _, pc, pbg, pfg = p_label(best["Parking Avail. (%)"])
    risk = round(best["Congestion (%)"]*0.7 + (100-best["Parking Avail. (%)"])*0.3, 1)
    rt, _, rc, rbg, rfg = c_label(risk)

    # ── Peak alert ───────────────────────────
    if all(df["Congestion (%)"] >= 72):
        st.markdown("""
<div style='background:#FFFBEB;border-left:5px solid #F59E0B;border-radius:0 10px 10px 0;
            padding:14px 20px;margin-bottom:20px;'>
  <span style='color:#92400E;font-weight:700;'>⚠️ Peak Congestion Alert:</span>
  <span style='color:#78350F;'> All routes are experiencing HIGH congestion.
  Consider travelling during <strong style='color:#92400E;'>9 PM – 12 AM</strong> for best results.</span>
</div>""", unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────
    st.markdown("<p style='color:#374151;font-weight:700;font-size:1rem;margin-bottom:10px;'>📊 Key Performance Indicators</p>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(card("🚗 Traffic Level", ct, f"Score: {best['Congestion (%)']}%"), unsafe_allow_html=True)
    with k2:
        st.markdown(card("🅿️ Parking Status", pt, f"Vacancy: {best['Parking Avail. (%)']}%"), unsafe_allow_html=True)
    with k3:
        st.markdown(card("⏱️ Est. Delay", f"{best['Delay (min)']} min", f"Travel: {best['Total Time (min)']} min total"), unsafe_allow_html=True)
    with k4:
        st.markdown(card("⚠️ Risk Score", f"{risk} / 100", rt.split(" ",1)[1] + " risk"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Route Table ──────────────────────────
    st.markdown("<p style='color:#111827;font-weight:700;font-size:1.1rem;margin-bottom:8px;'>🗺️ Route Comparison Analysis</p>", unsafe_allow_html=True)

    # Build HTML table — white text everywhere, color-coded congestion badge
    def cong_cell(score):
        if score < 40:
            return f"<td style='padding:12px 16px;color:#FFFFFF;font-weight:700;background:#16A34A;text-align:center;'>{score}%</td>"
        elif score < 72:
            return f"<td style='padding:12px 16px;color:#FFFFFF;font-weight:700;background:#D97706;text-align:center;'>{score}%</td>"
        else:
            return f"<td style='padding:12px 16px;color:#FFFFFF;font-weight:700;background:#DC2626;text-align:center;'>{score}%</td>"

    table_rows = ""
    for i, row in df.iterrows():
        row_bg = "#FFFFFF" if i % 2 == 0 else "#F8FAFF"
        table_rows += f"""
        <tr style='background:{row_bg};'>
          <td style='padding:12px 16px;color:#111827;font-weight:600;border-right:1px solid #E5E7EB;'>{row['Route']}</td>
          <td style='padding:12px 16px;color:#1E3A8A;font-weight:600;text-align:center;border-right:1px solid #E5E7EB;'>{row['Distance (km)']:.1f} km</td>
          {cong_cell(row['Congestion (%)'])}
          <td style='padding:12px 16px;color:#374151;text-align:center;border-right:1px solid #E5E7EB;'>{row['Delay (min)']:.1f} min</td>
          <td style='padding:12px 16px;color:#1E3A8A;font-weight:700;text-align:center;'>{row['Total Time (min)']:.1f} min</td>
        </tr>"""

    st.markdown(f"""
<div style='border:1px solid #E5E7EB;border-radius:12px;overflow:hidden;'>
  <table style='width:100%;border-collapse:collapse;font-family:Inter,sans-serif;font-size:14px;'>
    <thead>
      <tr style='background:#1E3A8A;'>
        <th style='padding:13px 16px;color:#FFFFFF;font-weight:700;text-align:left;border-right:1px solid #3B5998;'>Route</th>
        <th style='padding:13px 16px;color:#FFFFFF;font-weight:700;text-align:center;border-right:1px solid #3B5998;'>Distance</th>
        <th style='padding:13px 16px;color:#FFFFFF;font-weight:700;text-align:center;border-right:1px solid #3B5998;'>Congestion</th>
        <th style='padding:13px 16px;color:#FFFFFF;font-weight:700;text-align:center;border-right:1px solid #3B5998;'>Est. Delay</th>
        <th style='padding:13px 16px;color:#FFFFFF;font-weight:700;text-align:center;'>Total Time</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────
    v1, v2 = st.columns([3, 2])

    with v1:
        st.markdown("<p style='color:#111827;font-weight:700;font-size:1rem;margin-bottom:4px;'>📈 Congestion Score by Route</p>", unsafe_allow_html=True)
        bar_colors = ["#16A34A" if s<40 else "#F59E0B" if s<72 else "#DC2626"
                      for s in df["Congestion (%)"]]
        fig = go.Figure(go.Bar(
            x=df["Route"],
            y=df["Congestion (%)"],
            marker_color=bar_colors,
            text=[f"{v}%" for v in df["Congestion (%)"]],
            textposition="outside",
            textfont=dict(color="#111827", size=13, family="Inter"),
        ))
        fig.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F9FAFB",
            font=dict(family="Inter", color="#111827", size=13),
            xaxis=dict(tickfont=dict(color="#111827"), title=dict(text="Route", font=dict(color="#374151")), gridcolor="#E5E7EB"),
            yaxis=dict(tickfont=dict(color="#111827"), title=dict(text="Congestion (%)", font=dict(color="#374151")), range=[0, 115], gridcolor="#E5E7EB"),
            margin=dict(t=30, b=20, l=10, r=10), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with v2:
        st.markdown("<p style='color:#111827;font-weight:700;font-size:1rem;margin-bottom:4px;'>🅿️ Parking Vacancy</p>", unsafe_allow_html=True)
        avail = best["Parking Avail. (%)"]
        fig2 = go.Figure(go.Pie(
            labels=["Available","Occupied"],
            values=[avail, 100-avail],
            hole=0.52,
            marker_colors=["#16A34A","#E5E7EB"],
            textfont=dict(color="#111827", size=13, family="Inter"),
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        ))
        fig2.update_layout(
            paper_bgcolor="#FFFFFF",
            font=dict(family="Inter", color="#111827"),
            legend=dict(font=dict(color="#111827", size=12), bgcolor="#FFFFFF"),
            annotations=[dict(text=f"<b>{avail}%</b><br>Free",
                              x=0.5, y=0.5, font_size=15, font_color="#111827", showarrow=False)],
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"<p style='color:#374151;font-weight:600;font-size:0.9rem;margin-bottom:6px;'>Overall Travel Risk</p>", unsafe_allow_html=True)
        st.progress(min(risk/100, 1.0))
        st.markdown(badge(f"Risk: {risk}/100 — {rt.split(' ',1)[1]}", rbg, rc, rfg), unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Recommendations ──────────────────────
    st.markdown("<p style='color:#111827;font-weight:700;font-size:1.1rem;margin-bottom:10px;'>🤖 Smart Travel Recommendation</p>", unsafe_allow_html=True)

    best_time = min(TIME_WEIGHTS, key=TIME_WEIGHTS.get)
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
<div style='background:#F0FDF4;border:1px solid #86EFAC;border-radius:12px;padding:16px 18px;'>
  <p style='color:#166534;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 6px;'>🛣️ Optimal Route</p>
  <p style='color:#14532D;font-size:1rem;font-weight:700;margin:0;'>{best['Route']}</p>
</div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
<div style='background:#EFF6FF;border:1px solid #93C5FD;border-radius:12px;padding:16px 18px;'>
  <p style='color:#1E40AF;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 6px;'>🕐 Best Time Window</p>
  <p style='color:#1E3A8A;font-size:1rem;font-weight:700;margin:0;'>{best_time}</p>
</div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
<div style='background:#FFFBEB;border:1px solid #FCD34D;border-radius:12px;padding:16px 18px;'>
  <p style='color:#92400E;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 6px;'>⏳ Total Journey Time</p>
  <p style='color:#78350F;font-size:1rem;font-weight:700;margin:0;'>{best['Total Time (min)']} minutes</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    p_word = pt.split(" ",1)[1] if " " in pt else pt
    c_word = ct.split(" ",1)[1] if " " in ct else ct

    st.markdown(f"""
<div style='background:#EFF6FF;border-left:5px solid #1E3A8A;border-radius:0 12px 12px 0;padding:22px 26px;'>
  <p style='color:#1E3A8A;font-weight:800;font-size:1rem;margin:0 0 12px;'>🧠 CrowdSense AI Insight</p>
  <p style='color:#1e293b;font-size:0.94rem;line-height:1.8;margin:0;'>
    Your journey from <strong style='color:#1E3A8A;'>{src_c}</strong>
    to <strong style='color:#1E3A8A;'>{dest_c}</strong>
    has been analysed for the <strong style='color:#1E3A8A;'>{time_slot}</strong>
    slot in a <strong style='color:#1E3A8A;'>{area}</strong> zone.<br><br>
    ▸ <strong style='color:#111827;'>Recommended Route:</strong>
      <span style='color:#1E3A8A;font-weight:600;'>{best['Route']}</span><br>
    ▸ <strong style='color:#111827;'>Traffic Density:</strong>
      <span style='color:{cc};font-weight:600;'>{best['Congestion (%)']}% — {c_word}</span><br>
    ▸ <strong style='color:#111827;'>Parking at Destination:</strong>
      <span style='color:{pc};font-weight:600;'>{p_word}</span><br>
    ▸ <strong style='color:#111827;'>Estimated Delay:</strong>
      <span style='color:#374151;'>{best['Delay (min)']} minutes</span><br>
    ▸ <strong style='color:#111827;'>Best Departure Time:</strong>
      <span style='color:#1E3A8A;font-weight:600;'>{best_time}</span><br><br>
    <em style='color:#475569;'>💡 We recommend departing
    <strong style='color:#111827;'>10 minutes early</strong> to buffer for
    real-world variability. Switching to <strong style='color:#111827;'>{best_time}</strong>
    can reduce delays by up to 40%.</em>
  </p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  WELCOME STATE
# ─────────────────────────────────────────────
else:
    st.markdown("""
<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;
            padding:40px;text-align:center;margin-bottom:24px;'>
  <h2 style='color:#1E3A8A;font-weight:800;margin-bottom:10px;'>👋 Welcome to CrowdSense AI</h2>
  <p style='color:#374151;font-size:1rem;max-width:580px;margin:0 auto;'>
    Enter your journey details in the sidebar — Source, Destination, Area Type, and Time Slot —
    then click <strong style='color:#1E3A8A;'>"Analyze Traffic"</strong> to receive AI-powered predictions.
  </p>
</div>""", unsafe_allow_html=True)

    ca, cb, cc_col = st.columns(3)
    cards = [
        ("🗺️", "Intelligent Routing",   "#1E3A8A", "Compares 3 route alternatives and ranks them by congestion risk and travel time."),
        ("🅿️", "Parking Prediction",    "#15803D", "Estimates parking vacancy at your destination based on area type and time of day."),
        ("📊", "Risk Analytics",         "#9333EA", "Generates a composite risk score combining congestion and parking stress signals."),
    ]
    for col, (icon, title, color, desc) in zip([ca, cb, cc_col], cards):
        with col:
            st.markdown(f"""
<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:14px;
            padding:28px 22px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
  <div style='font-size:2.5rem;margin-bottom:12px;'>{icon}</div>
  <h3 style='color:{color};font-weight:700;font-size:1.05rem;margin:0 0 8px;'>{title}</h3>
  <p style='color:#6B7280;font-size:0.88rem;margin:0;line-height:1.6;'>{desc}</p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:24px 0 10px;margin-top:30px;border-top:1px solid #E5E7EB;'>
  <span style='color:#9CA3AF;font-size:0.83rem;'>
    CrowdSense AI System &nbsp;|&nbsp; Rule-Based Simulated Intelligence Engine &nbsp;|&nbsp; © 2026 Virtual Urban Lab
  </span>
</div>""", unsafe_allow_html=True)
