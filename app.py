"""
VisionAI Pro v4
Run: streamlit run app.py
"""

import io, cv2, time, json, csv, tempfile, os
import numpy as np
import pandas as pd
import streamlit as st
from collections import defaultdict
from datetime import datetime

st.set_page_config(
    page_title="VisionAI Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  CSS  — aggressive selectors so Streamlit's light theme cant win
# ══════════════════════════════════════════════════════════════════
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── NUCLEAR BACKGROUND OVERRIDE ─── */
html                                    { background:#0d0d1f !important; }
body                                    { background:#0d0d1f !important; }
.stApp                                  { background:#0d0d1f !important; }
[data-testid="stAppViewContainer"]      { background:#0d0d1f !important; }
[data-testid="stAppViewBlockContainer"] { background:#0d0d1f !important; }
[data-testid="stVerticalBlock"]         { background:transparent !important; }
[data-testid="stHorizontalBlock"]       { background:transparent !important; }
.main .block-container                  { background:transparent !important; padding-top:0 !important; }
section.main                            { background:transparent !important; }
[data-testid="stHeader"]               { background:transparent !important; display:none !important; }
section[data-testid="stSidebar"]        { display:none !important; }
#MainMenu, footer                       { display:none !important; }
.stDeployButton,[data-testid="stDecoration"],[data-testid="collapsedControl"] { display:none !important; }

/* ─── GLOBAL TYPOGRAPHY ─── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], p, span, div, label, h1, h2, h3, h4, h5 {
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    color: #e0e0f4 !important;
}

/* ─── TOP NAV ─── */
.vnav {
    position: sticky; top: 0; z-index: 200;
    display: flex; align-items: center;
    padding: 0 28px; height: 58px;
    background: #16162a;
    border-bottom: 1px solid #2a2a48;
    box-shadow: 0 2px 24px rgba(0,0,0,.6);
}
.vnav-logo { display:flex; align-items:center; gap:10px; }
.vnav-mark {
    width:30px; height:30px; border-radius:9px;
    background: linear-gradient(135deg,#00f5d4,#7c3aed);
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.vnav-mark svg { width:15px; height:15px; }
.vnav-name { font-size:.98rem; font-weight:700; letter-spacing:-.01em; color:#e0e0f4 !important; }
.vnav-name span { color:#00f5d4 !important; }
.vnav-name sub { color:#7c3aed !important; font-size:.72rem; font-weight:500; vertical-align:middle; margin-left:2px; }
.vnav-badge {
    font-size:.6rem; font-family:'JetBrains Mono',monospace !important;
    padding:2px 8px; border-radius:999px; margin-left:6px;
    background:rgba(124,58,237,.18); border:1px solid rgba(124,58,237,.4);
    color:#a78bfa !important;
}
.vnav-stats { display:flex; gap:0; margin-left:auto; }
.vnav-stat {
    display:flex; flex-direction:column; align-items:center;
    padding:0 20px; border-left:1px solid #1e1e38;
}
.vnav-stat-n { font-size:.88rem; font-weight:700; font-family:'JetBrains Mono',monospace !important; line-height:1.1; color:#ffffff !important; }
.vnav-stat-l { font-size:.6rem; text-transform:uppercase; letter-spacing:.08em; color:#8888aa !important; margin-top:2px; font-weight:500; }

/* ─── MAIN CONTENT WRAPPER ─── */
.vcontent {
    background: #07071010;
    min-height: calc(100vh - 54px);
    padding: 24px 28px 40px;
}

/* ─── SETTINGS BAR ─── */
.vcontrols {
    background: #16162a;
    border: 1px solid #2a2a48;
    border-radius: 14px;
    padding: 18px 20px 14px;
    margin-bottom: 18px;
}

/* ─── SLIDERS ─── */
[data-testid="stSlider"] { padding: 0 !important; }
[data-testid="stSlider"] > div { background:transparent !important; }
[data-testid="stSlider"] > div > div > div > div {
    background: #1e1e38 !important; height:3px !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #00f5d4 !important;
    border: 2px solid #07071010 !important;
    width:14px !important; height:14px !important;
    box-shadow: 0 0 10px rgba(0,245,212,.6) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    background: #00f5d4 !important; color: #07071010 !important;
    font-family: 'JetBrains Mono',monospace !important;
    font-size:.68rem !important; font-weight:600 !important;
    border-radius:4px !important; padding:1px 5px !important;
}
[data-testid="stSlider"] label p { color:#7070a0 !important; font-size:.72rem !important; }

/* ─── TOGGLE ─── */
[data-testid="stToggle"] label p { color:#7070a0 !important; font-size:.72rem !important; }
[data-testid="stToggle"] span[data-baseweb="toggle"] > div {
    background:#1e1e38 !important; border:1px solid #2e2e50 !important;
}
[data-testid="stToggle"] input:checked ~ span[data-baseweb="toggle"] > div {
    background:rgba(0,245,212,.25) !important; border-color:rgba(0,245,212,.5) !important;
}
[data-testid="stToggle"] span[data-baseweb="toggle"] > div > div { background:#4a4a70 !important; }
[data-testid="stToggle"] input:checked ~ span[data-baseweb="toggle"] > div > div { background:#00f5d4 !important; }

/* ─── TEXT INPUT ─── */
[data-testid="stTextInput"] label p { color:#7070a0 !important; font-size:.72rem !important; }
[data-testid="stTextInput"] input {
    background:#0a0a1a !important; border:1px solid #2e2e50 !important;
    border-radius:9px !important; color:#e0e0f4 !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.76rem !important;
    padding:8px 12px !important;
}
[data-testid="stTextInput"] input:focus { border-color:#00f5d4 !important; outline:none !important; box-shadow:0 0 0 2px rgba(0,245,212,.15) !important; }
[data-testid="stTextInput"] input::placeholder { color:#3a3a60 !important; }

/* ─── NUMBER INPUT ─── */
[data-testid="stNumberInput"] label p { color:#7070a0 !important; font-size:.72rem !important; }
[data-testid="stNumberInput"] input {
    background:#0a0a1a !important; border:1px solid #2e2e50 !important;
    border-radius:9px !important; color:#e0e0f4 !important;
    font-family:'JetBrains Mono',monospace !important;
}
[data-testid="stNumberInput"] button {
    background:#0e0e20 !important; border-color:#2e2e50 !important; color:#7070a0 !important;
}
[data-testid="stNumberInput"] button:hover { background:#1e1e38 !important; color:#00f5d4 !important; }

/* ─── RADIO (mode selector) ─── */
[data-testid="stRadio"] > label { display:none !important; }
[data-testid="stRadio"] div[role="radiogroup"] { display:flex !important; gap:8px !important; flex-wrap:wrap !important; }
[data-testid="stRadio"] label[data-baseweb="radio"] {
    background:#0e0e20 !important; border:1px solid #2e2e50 !important;
    border-radius:10px !important; padding:9px 20px !important;
    color:#6060a0 !important; font-size:.8rem !important; font-weight:500 !important;
    cursor:pointer !important; transition:all .15s !important; margin:0 !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:hover { border-color:#00f5d4 !important; color:#00f5d4 !important; background:rgba(0,245,212,.05) !important; }
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background:rgba(0,245,212,.1) !important; border-color:#00f5d4 !important;
    color:#00f5d4 !important; box-shadow:0 0 14px rgba(0,245,212,.15) !important;
}
[data-testid="stRadio"] [data-baseweb="radio"] div:first-child { display:none !important; }

/* ─── FILE UPLOADER ─── */
[data-testid="stFileUploader"] > section {
    background:#0a0a1a !important;
    border:2px dashed #2e2e50 !important;
    border-radius:12px !important;
    transition:border-color .2s !important;
}
[data-testid="stFileUploader"] > section:hover { border-color:#00f5d4 !important; }
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span { color:#4a4a70 !important; }
[data-testid="stFileUploader"] button {
    background:#0e0e20 !important; border:1px solid #2e2e50 !important;
    border-radius:8px !important; color:#9090c0 !important; font-size:.76rem !important;
}
[data-testid="stFileUploader"] button:hover { border-color:#00f5d4 !important; color:#00f5d4 !important; }

/* ─── BUTTONS ─── */
.stButton > button {
    background:#0e0e20 !important; border:1px solid #2e2e50 !important;
    border-radius:10px !important; color:#9090c0 !important;
    font-family:'Space Grotesk',sans-serif !important;
    font-weight:500 !important; font-size:.8rem !important;
    padding:8px 20px !important; transition:all .15s !important;
}
.stButton > button:hover {
    background:rgba(0,245,212,.08) !important; border-color:#00f5d4 !important;
    color:#00f5d4 !important; transform:translateY(-1px) !important;
    box-shadow:0 4px 16px rgba(0,245,212,.12) !important;
}
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,rgba(0,245,212,.18),rgba(124,58,237,.18)) !important;
    border:1px solid #00f5d4 !important; color:#00f5d4 !important;
    box-shadow:0 0 20px rgba(0,245,212,.1) !important;
}
.stButton > button[kind="primary"]:hover {
    background:linear-gradient(135deg,rgba(0,245,212,.28),rgba(124,58,237,.28)) !important;
    box-shadow:0 4px 24px rgba(0,245,212,.2) !important;
}

/* ─── DOWNLOAD BUTTON ─── */
[data-testid="stDownloadButton"] > button {
    background:rgba(124,58,237,.1) !important; border:1px solid rgba(124,58,237,.4) !important;
    color:#a78bfa !important; border-radius:10px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background:rgba(124,58,237,.2) !important; box-shadow:0 4px 16px rgba(124,58,237,.2) !important;
}

/* ─── PROGRESS ─── */
[data-testid="stProgress"] > div { background:#1e1e38 !important; border-radius:4px !important; }
[data-testid="stProgress"] > div > div {
    background:linear-gradient(90deg,#00f5d4,#7c3aed) !important;
    border-radius:4px !important; transition:width .15s !important;
}

/* ─── TABS ─── */
[data-testid="stTabs"] [role="tablist"] {
    background:transparent !important; border-bottom:1px solid #1e1e38 !important; gap:2px !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background:transparent !important; border:none !important;
    color:#4a4a70 !important; font-family:'Space Grotesk',sans-serif !important;
    font-size:.8rem !important; font-weight:500 !important;
    padding:10px 20px !important; border-radius:8px 8px 0 0 !important;
    transition:all .15s !important;
}
[data-testid="stTabs"] button[role="tab"]:hover { color:#9090c0 !important; background:rgba(255,255,255,.03) !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color:#00f5d4 !important; background:rgba(0,245,212,.07) !important;
    border-bottom:2px solid #00f5d4 !important;
}
[data-testid="stTabs"] [role="tabpanel"] { padding-top:20px !important; background:transparent !important; }

/* ─── IMAGE ─── */
[data-testid="stImage"] img {
    border-radius:14px !important; border:1px solid #1e1e38 !important;
    box-shadow:0 8px 40px rgba(0,0,0,.5) !important;
}

/* ─── DATAFRAME ─── */
[data-testid="stDataFrame"] { border:1px solid #1e1e38 !important; border-radius:12px !important; overflow:hidden !important; }
[data-testid="stDataFrame"] * { background:#0a0a1a !important; color:#c0c0e0 !important; }
iframe { background:#0a0a1a !important; }

/* ─── EXPANDER ─── */
[data-testid="stExpander"] {
    background:#0e0e20 !important; border:1px solid #1e1e38 !important;
    border-radius:12px !important;
}
[data-testid="stExpander"] summary { color:#6060a0 !important; }
[data-testid="stExpander"] summary:hover { color:#9090c0 !important; }

/* ─── ALERTS ─── */
[data-testid="stAlert"] {
    background:rgba(58,134,255,.08) !important; border:1px solid rgba(58,134,255,.25) !important;
    border-radius:10px !important; color:#6ba3ff !important;
}

/* ─── SPINNER ─── */
[data-testid="stSpinner"] > div { border-top-color:#00f5d4 !important; }

/* ─── DIVIDER ─── */
hr { border:none !important; border-top:1px solid #1e1e38 !important; margin:16px 0 !important; }

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#07071010; }
::-webkit-scrollbar-thumb { background:#2e2e50; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#00f5d4; }

/* ─── METRIC CHIPS ─── */
.chips { display:flex; gap:10px; margin:14px 0; flex-wrap:wrap; }
.chip {
    flex:1; min-width:110px;
    background:#16162a; border-radius:13px;
    border:1px solid #2a2a48; padding:14px 16px;
    position:relative; overflow:hidden;
    transition:transform .15s, box-shadow .15s;
}
.chip:hover { transform:translateY(-2px); box-shadow:0 6px 24px rgba(0,0,0,.4); }
.chip::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; border-radius:13px 13px 0 0; }
.chip.cy { border-color:rgba(0,245,212,.2); }
.chip.cy::before { background:linear-gradient(90deg,transparent,#00f5d4,transparent); }
.chip.pu { border-color:rgba(124,58,237,.25); }
.chip.pu::before { background:linear-gradient(90deg,transparent,#7c3aed,transparent); }
.chip.am { border-color:rgba(255,190,11,.2); }
.chip.am::before { background:linear-gradient(90deg,transparent,#ffbe0b,transparent); }
.chip.gr { border-color:rgba(6,214,160,.2); }
.chip.gr::before { background:linear-gradient(90deg,transparent,#06d6a0,transparent); }
.chip.bl { border-color:rgba(58,134,255,.2); }
.chip.bl::before { background:linear-gradient(90deg,transparent,#3a86ff,transparent); }
.chip-v { font-size:1.5rem; font-weight:700; letter-spacing:-.03em; line-height:1; margin-bottom:5px; font-family:'JetBrains Mono',monospace !important; }
.chip-l { font-size:.6rem; text-transform:uppercase; letter-spacing:.1em; color:#6868a0 !important; }
.cy .chip-v { color:#00f5d4 !important; } .pu .chip-v { color:#a78bfa !important; }
.am .chip-v { color:#ffbe0b !important; } .gr .chip-v { color:#06d6a0 !important; }
.bl .chip-v { color:#60a5fa !important; }

/* ─── SECTION LABELS ─── */
.slbl {
    font-size:.62rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.12em; color:#7070a0 !important;
    display:flex; align-items:center; gap:10px; margin:18px 0 10px;
}
.slbl::after { content:''; flex:1; height:1px; background:#2a2a48; }

/* ─── BADGES ─── */
.badges { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
.bdg {
    display:inline-flex; align-items:center; gap:6px;
    padding:5px 12px; border-radius:999px;
    background:#0e0e20; border:1px solid #2e2e50;
    font-size:.7rem; font-family:'JetBrains Mono',monospace !important; color:#9090c0 !important;
    transition:border-color .15s;
}
.bdg:hover { border-color:#00f5d4; }
.bdg-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }

/* ─── GLASS CARD ─── */
.gcard {
    background:#16162a; border:1px solid #2a2a48;
    border-radius:14px; padding:20px;
    transition:border-color .15s, box-shadow .15s;
}
.gcard:hover { border-color:#2e2e50; box-shadow:0 4px 20px rgba(0,0,0,.3); }

/* ─── VIDEO INFO BAR ─── */
.vinfo {
    font-size:.7rem; font-family:'JetBrains Mono',monospace !important;
    color:#4a4a70 !important; background:#0a0a1a;
    border:1px solid #1e1e38; border-radius:8px;
    padding:7px 12px; margin:8px 0;
}

/* ─── EMPTY STATE ─── */
.empty-state {
    text-align:center; padding:60px 20px;
    border:2px dashed #1e1e38; border-radius:14px;
    background:#0a0a1a;
}
.empty-icon { font-size:3rem; margin-bottom:12px; filter:grayscale(.3); }
.empty-title { font-size:.88rem; color:#5a5a80 !important; margin-bottom:5px; }
.empty-sub { font-size:.7rem; color:#2e2e50 !important; }

/* ─── SPLASH ─── */
.splash {
    position:fixed; inset:0; z-index:9999;
    background:#07071010;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
}
.splash-logo {
    font-size:3.6rem; font-weight:700; letter-spacing:-.04em;
    background:linear-gradient(135deg,#00f5d4 0%,#7c3aed 55%,#3a86ff 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    animation:sup .9s cubic-bezier(.16,1,.3,1) both;
}
.splash-tagline {
    font-size:.78rem; color:#3a3a60 !important; letter-spacing:.16em;
    text-transform:uppercase; margin:6px 0 48px;
    animation:sup .9s .15s cubic-bezier(.16,1,.3,1) both;
}
.splash-bar { width:200px; height:2px; background:#1e1e38; border-radius:2px; overflow:hidden; animation:sup .5s .3s both; }
.splash-fill { height:100%; background:linear-gradient(90deg,#00f5d4,#7c3aed,#3a86ff); animation:fill 2s ease forwards; }
@keyframes fill { from{width:0} to{width:100%} }
.splash-dots { display:flex; gap:8px; margin-top:20px; animation:sup .5s .45s both; }
.sdot { width:6px; height:6px; border-radius:50%; animation:dglow 1.5s ease infinite; }
.sdot:nth-child(2){animation-delay:.25s} .sdot:nth-child(3){animation-delay:.5s}
@keyframes dglow{0%,100%{background:#1e1e38;transform:scale(1)}50%{background:#00f5d4;transform:scale(1.6);box-shadow:0 0 10px #00f5d4}}
@keyframes sup{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}

/* ─── LINE CHART overrides ─── */
[data-testid="stVegaLiteChart"] canvas, [data-testid="stArrowVegaLiteChart"] canvas { background:transparent !important; }
</style>
""", unsafe_allow_html=True)

# Also inject via components for extra certainty
st.components.v1.html("""
<script>
  function forceDark() {
    document.body.style.setProperty('background', '#0d0d1f', 'important');
    document.documentElement.style.setProperty('background', '#0d0d1f', 'important');
    var els = document.querySelectorAll('[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],section.main,.main');
    els.forEach(function(el){ el.style.setProperty('background','#0d0d1f','important'); });
  }
  forceDark();
  setTimeout(forceDark, 100);
  setTimeout(forceDark, 500);
  var obs = new MutationObserver(forceDark);
  obs.observe(document.body, {childList:true, subtree:true});
</script>
""", height=0)

# ══════════════════════════════════════════════════════════════════
#  SPLASH
# ══════════════════════════════════════════════════════════════════
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if not st.session_state.splash_done:
    spl = st.empty()
    spl.markdown("""
    <div class="splash">
      <div class="splash-logo">VisionAI Pro</div>
      <div class="splash-tagline">Real-time Object Detection · YOLOv8s</div>
      <div class="splash-bar"><div class="splash-fill"></div></div>
      <div class="splash-dots">
        <div class="sdot"></div><div class="sdot"></div><div class="sdot"></div>
      </div>
    </div>""", unsafe_allow_html=True)
    time.sleep(2.2)
    spl.empty()
    st.session_state.splash_done = True

# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "history":       [],
        "total_dets":    0,
        "class_counts":  defaultdict(int),
        "conf_dist":     defaultdict(int),
        "session_start": datetime.now(),
        "inf_times":     [],
        "last_ann":      None,
        "last_dets":     [],
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ══════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8s.pt")

PALETTE = [
    "#00f5d4","#f72585","#a855f7","#3a86ff","#fb5607","#ffbe0b",
    "#06d6a0","#ef233c","#4cc9f0","#ff006e","#80b0ff","#8338ec",
    "#ff9f1c","#2ec4b6","#e71d36","#f4d35e",
]

def hex_bgr(h):
    h=h.lstrip('#'); return int(h[4:6],16),int(h[2:4],16),int(h[0:2],16)

# ══════════════════════════════════════════════════════════════════
#  DETECTION CORE  (resize→detect→scale back for 4K)
# ══════════════════════════════════════════════════════════════════
DETECT_W = 1280

def detect_frame(frame_bgr, conf=.40, iou=.45, face_blur=False, cls_filter=None):
    model = load_model()
    oh, ow = frame_bgr.shape[:2]
    scale  = DETECT_W / ow
    small  = cv2.resize(frame_bgr, (DETECT_W, int(oh*scale)))

    t0  = time.perf_counter()
    kw  = dict(conf=conf, iou=iou, verbose=False)
    if cls_filter: kw["classes"] = cls_filter
    res = model(small, **kw)[0]
    inf_ms = round((time.perf_counter()-t0)*1000, 1)

    inv   = 1.0/scale
    thick = max(2, int(ow/700))
    fs    = max(.45, ow/2800)
    ft    = max(1, int(ow/1400))
    pad   = max(4, int(ow/500))

    dets=[];  cls_cnt=defaultdict(int)

    if res.boxes is not None:
        for box in res.boxes:
            cid   = int(box.cls[0]); cv_ = float(box.conf[0])
            label = model.names[cid]; color = PALETTE[cid % len(PALETTE)]
            bgr   = hex_bgr(color)
            sx1,sy1,sx2,sy2 = box.xyxy[0].tolist()
            x1=int(sx1*inv); y1=int(sy1*inv); x2=int(sx2*inv); y2=int(sy2*inv)

            if face_blur and label=="person":
                fy1=max(0,y1); fx1=max(0,x1); hh=max(20,(y2-y1)//3)
                roi=frame_bgr[fy1:fy1+hh,fx1:min(ow,x2)]
                if roi.size>0:
                    k=max(21,(hh//4)*2+1)
                    frame_bgr[fy1:fy1+hh,fx1:min(ow,x2)]=cv2.GaussianBlur(roi,(k,k),0)

            ov=frame_bgr.copy()
            cv2.rectangle(ov,(x1,y1),(x2,y2),bgr,-1)
            cv2.addWeighted(ov,.12,frame_bgr,.88,0,frame_bgr)
            cv2.rectangle(frame_bgr,(x1,y1),(x2,y2),bgr,thick,cv2.LINE_AA)

            cl=max(10,min(int((x2-x1)*.12),int((y2-y1)*.12),int(ow/80)))
            for cx_,cy_,dx,dy in[(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame_bgr,(cx_,cy_),(cx_+dx*cl,cy_),bgr,thick+1,cv2.LINE_AA)
                cv2.line(frame_bgr,(cx_,cy_),(cx_,cy_+dy*cl),bgr,thick+1,cv2.LINE_AA)

            txt=f"{label}  {cv_*100:.0f}%"
            (tw,lh),bl=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,fs,ft)
            ly=y1-6 if y1>lh+14 else y2+lh+8
            cv2.rectangle(frame_bgr,(x1-1,ly-lh-pad),(x1+tw+pad*2,ly+bl),bgr,-1)
            cv2.putText(frame_bgr,txt,(x1+pad,ly),cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,255),ft,cv2.LINE_AA)

            dets.append({"label":label,"conf":round(cv_,3),"bbox":[x1,y1,x2-x1,y2-y1],"color":color,"cls_id":cid})
            cls_cnt[label]+=1

    if dets:
        wms=max(.6,ow/2000); wmt=max(2,int(ow/700))
        cv2.putText(frame_bgr,f"Objects: {len(dets)}",(12,int(oh*.018)+18),
                    cv2.FONT_HERSHEY_SIMPLEX,wms,(0,245,212),wmt,cv2.LINE_AA)

    st.session_state.total_dets+=len(dets)
    st.session_state.inf_times.append(inf_ms)
    for lbl,cnt in cls_cnt.items(): st.session_state.class_counts[lbl]+=cnt
    for d in dets:
        b=min(int(d["conf"]*100)//10*10,90); st.session_state.conf_dist[str(b)]+=1
    st.session_state.history.append({"ts":time.time(),"count":len(dets),"inf":inf_ms,"classes":dict(cls_cnt)})
    st.session_state.last_dets=dets
    return frame_bgr,dets,inf_ms

def parse_cls(text):
    if not text.strip(): return None
    m=load_model(); inv={v.lower():k for k,v in m.names.items()}
    ids=[inv[s.strip().lower()] for s in text.split(",") if s.strip().lower() in inv]
    return ids or None

# ══════════════════════════════════════════════════════════════════
#  TOP NAV
# ══════════════════════════════════════════════════════════════════
its=st.session_state.inf_times
avg_ms=round(sum(its)/max(len(its),1),1) if its else 0
top_cls=max(st.session_state.class_counts,key=st.session_state.class_counts.get,default="—") if st.session_state.class_counts else "—"

st.markdown(f"""
<div class="vnav">
  <div class="vnav-logo">
    <div class="vnav-mark">
      <svg viewBox="0 0 15 15" fill="none" stroke="white" stroke-width="1.8">
        <rect x="1" y="1" width="5.5" height="5.5" rx="1.2"/>
        <rect x="8.5" y="1" width="5.5" height="5.5" rx="1.2"/>
        <rect x="1" y="8.5" width="5.5" height="5.5" rx="1.2"/>
        <rect x="8.5" y="8.5" width="5.5" height="5.5" rx="1.2"/>
      </svg>
    </div>
    <span class="vnav-name">Vision<span>AI</span><sub>Pro</sub></span>
    <span class="vnav-badge">YOLOv8s</span>
  </div>
  <div class="vnav-stats">
    <div class="vnav-stat">
      <span class="vnav-stat-n" style="color:#00f5d4">{st.session_state.total_dets:,}</span>
      <span class="vnav-stat-l">Detections</span>
    </div>
    <div class="vnav-stat">
      <span class="vnav-stat-n" style="color:#a78bfa">{top_cls}</span>
      <span class="vnav-stat-l">Top class</span>
    </div>
    <div class="vnav-stat">
      <span class="vnav-stat-n" style="color:#ffbe0b">{avg_ms}ms</span>
      <span class="vnav-stat-l">Avg inference</span>
    </div>
    <div class="vnav-stat">
      <span class="vnav-stat-n">{len(st.session_state.history)}</span>
      <span class="vnav-stat-l">Frames</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# padding after nav
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════
tab_d, tab_a, tab_e = st.tabs(["  🔍  Detect  ","  📊  Analytics  ","  💾  Export  "])

# ─────────────────────────── DETECT ────────────────────────────
with tab_d:

    # Settings card
    with st.container():
        st.markdown('<div class="vcontrols">', unsafe_allow_html=True)
        sc1,sc2,sc3,sc4 = st.columns([1.5,1.5,0.8,1.8])
        with sc1: conf_thr=st.slider("Confidence threshold",.10,.95,.35,.05,format="%.0f%%")
        with sc2: iou_thr =st.slider("IoU threshold",.10,.90,.45,.05,format="%.0f%%")
        with sc3: do_blur  =st.toggle("Face blur",False)
        with sc4: cls_text =st.text_input("Filter classes","",placeholder="person, car, dog, bird…")
        st.markdown('</div>', unsafe_allow_html=True)
    cls_filter=parse_cls(cls_text)

    # Mode
    st.markdown('<div class="slbl">Input mode</div>', unsafe_allow_html=True)
    mode=st.radio("_mode",["📷  Image","🎥  Video"],horizontal=True,label_visibility="collapsed")
    st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)

    # ── IMAGE ──────────────────────────────────────────────────
    if "Image" in mode:
        c1,c2=st.columns([4,1])
        with c1: uploaded=st.file_uploader("",type=["jpg","jpeg","png","bmp","webp"],label_visibility="collapsed")
        with c2:
            st.markdown("<div style='height:28px'></div>",unsafe_allow_html=True)
            clicked=st.button("🎯  Detect",type="primary",use_container_width=True)

        if uploaded:
            raw=np.frombuffer(uploaded.read(),np.uint8)
            frame=cv2.imdecode(raw,cv2.IMREAD_COLOR)
            if clicked:
                with st.spinner("Running YOLOv8s…"):
                    ann,dets,inf_ms=detect_frame(frame.copy(),conf_thr,iou_thr,do_blur,cls_filter)
                ann_rgb=cv2.cvtColor(ann,cv2.COLOR_BGR2RGB)
                _,buf=cv2.imencode(".jpg",ann,[cv2.IMWRITE_JPEG_QUALITY,93])
                st.session_state.last_ann=buf.tobytes()

                top=dets[0]["label"] if dets else "—"
                mx=f'{max(d["conf"] for d in dets)*100:.0f}%' if dets else "—"
                st.markdown(f"""
                <div class="chips">
                  <div class="chip cy"><div class="chip-v">{len(dets)}</div><div class="chip-l">Objects found</div></div>
                  <div class="chip am"><div class="chip-v">{inf_ms}ms</div><div class="chip-l">Inference</div></div>
                  <div class="chip pu"><div class="chip-v">{top}</div><div class="chip-l">Top class</div></div>
                  <div class="chip gr"><div class="chip-v">{mx}</div><div class="chip-l">Max confidence</div></div>
                </div>""",unsafe_allow_html=True)

                st.image(ann_rgb,use_container_width=True)

                if dets:
                    bdgs="".join([f'<span class="bdg"><span class="bdg-dot" style="background:{d["color"]}"></span>{d["label"]} {d["conf"]*100:.0f}%</span>'
                                   for d in sorted(dets,key=lambda x:-x["conf"])])
                    st.markdown(f'<div class="badges">{bdgs}</div>',unsafe_allow_html=True)
                    with st.expander("Detection table",expanded=False):
                        df=pd.DataFrame([{"Class":d["label"],"Conf":f'{d["conf"]*100:.1f}%',
                                           "X":d["bbox"][0],"Y":d["bbox"][1],"W":d["bbox"][2],"H":d["bbox"][3]}
                                          for d in dets])
                        st.dataframe(df,use_container_width=True,hide_index=True)
            else:
                st.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),use_container_width=True,caption="Image ready — click Detect")
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">📁</div>
              <div class="empty-title">Drag & drop an image, or click Browse</div>
              <div class="empty-sub">JPG · PNG · BMP · WEBP · up to 200 MB</div>
            </div>""",unsafe_allow_html=True)

    # ── VIDEO ──────────────────────────────────────────────────
    else:
        vid_file=st.file_uploader("",type=["mp4","avi","mov","mkv","mpeg4"],label_visibility="collapsed")
        v1,v2=st.columns(2)
        every_n   =v1.number_input("Process every N frames",1,10,2,help="2 = every other frame (faster)")
        max_frames=v2.number_input("Max frames",50,600,300)

        if vid_file:
            st.markdown('<div class="vinfo">⚡ 4K/HD auto-resized to 1280px for detection → boxes scaled back to original resolution</div>',unsafe_allow_html=True)

        if vid_file and st.button("▶  Process video",type="primary"):
            tmp_path=None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4",delete=False) as tmp:
                    tmp.write(vid_file.read()); tmp_path=tmp.name

                cap=cv2.VideoCapture(tmp_path)
                total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_v=cap.get(cv2.CAP_PROP_FPS) or 25
                ow=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); oh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                st.markdown(f'<div class="vinfo">Video: {ow}×{oh}px · {total} frames · {fps_v:.0f}fps</div>',unsafe_allow_html=True)

                prog=st.progress(0,text="Starting…")
                preview=st.empty()
                fi=0; processed=0; all_dets=[]; inf_list=[]; last_ann_bgr=None

                while cap.isOpened() and fi<max_frames:
                    ret,fr=cap.read()
                    if not ret: break
                    fi+=1
                    if fi%every_n!=0: continue
                    ann,dets,inf_ms=detect_frame(fr.copy(),conf_thr,iou_thr,do_blur,cls_filter)
                    processed+=1; last_ann_bgr=ann
                    dw=min(960,ow); dh=int(oh*dw/ow)
                    disp=cv2.resize(ann,(dw,dh))
                    preview.image(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB),use_container_width=True)
                    prog.progress(min(fi/max_frames,1.0),text=f"Frame {fi}/{min(max_frames,total)} · {len(dets)} objects · {inf_ms}ms")
                    all_dets.extend([{**d,"frame":fi} for d in dets]); inf_list.append(inf_ms)

                cap.release(); prog.empty()

                if last_ann_bgr is not None:
                    _,buf=cv2.imencode(".jpg",last_ann_bgr,[cv2.IMWRITE_JPEG_QUALITY,92])
                    st.session_state.last_ann=buf.tobytes()

                avg_i=round(sum(inf_list)/max(len(inf_list),1),1)
                st.markdown(f"""
                <div class="chips">
                  <div class="chip cy"><div class="chip-v">{len(all_dets):,}</div><div class="chip-l">Total detections</div></div>
                  <div class="chip pu"><div class="chip-v">{processed}</div><div class="chip-l">Frames processed</div></div>
                  <div class="chip am"><div class="chip-v">{avg_i}ms</div><div class="chip-l">Avg inference</div></div>
                </div>""",unsafe_allow_html=True)

                if all_dets:
                    cf=defaultdict(int)
                    for d in all_dets: cf[d["label"]]+=1
                    bdgs="".join([f'<span class="bdg"><span class="bdg-dot" style="background:{PALETTE[i%len(PALETTE)]}"></span>{l} ×{c}</span>'
                                   for i,(l,c) in enumerate(sorted(cf.items(),key=lambda x:-x[1]))])
                    st.markdown(f'<div class="badges">{bdgs}</div>',unsafe_allow_html=True)
                    with st.expander(f"Detection table ({len(all_dets):,} rows)",expanded=False):
                        df=pd.DataFrame(all_dets)[["frame","label","conf","bbox"]]
                        df["conf"]=df["conf"].map(lambda x:f"{x*100:.1f}%")
                        df.columns=["Frame","Class","Confidence","Bbox"]
                        st.dataframe(df,use_container_width=True,hide_index=True)
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except: pass
        elif not vid_file:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">🎬</div>
              <div class="empty-title">Upload a video file to begin</div>
              <div class="empty-sub">MP4 · AVI · MOV · MKV · 4K supported</div>
            </div>""",unsafe_allow_html=True)

# ─────────────────────────── ANALYTICS ─────────────────────────
with tab_a:
    hist=st.session_state.history; cc=st.session_state.class_counts
    its=st.session_state.inf_times; avg_inf=round(sum(its)/max(len(its),1),1) if its else 0
    top_c=max(cc,key=cc.get,default="—") if cc else "—"

    st.markdown(f"""
    <div class="chips">
      <div class="chip cy"><div class="chip-v">{st.session_state.total_dets:,}</div><div class="chip-l">Total detections</div></div>
      <div class="chip pu"><div class="chip-v">{len(hist):,}</div><div class="chip-l">Frames</div></div>
      <div class="chip am"><div class="chip-v">{top_c}</div><div class="chip-l">Top class</div></div>
      <div class="chip gr"><div class="chip-v">{avg_inf}ms</div><div class="chip-l">Avg inference</div></div>
      <div class="chip bl"><div class="chip-v">{len(cc)}</div><div class="chip-l">Unique classes</div></div>
    </div>""",unsafe_allow_html=True)

    if hist:
        df_h=pd.DataFrame(hist); df_h["frame"]=range(len(df_h))
        a1,a2=st.columns(2)
        with a1:
            st.markdown('<div class="slbl">Objects per frame</div>',unsafe_allow_html=True)
            st.line_chart(df_h.set_index("frame")[["count"]],color=["#00f5d4"],height=180,use_container_width=True)
        with a2:
            st.markdown('<div class="slbl">Inference time (ms)</div>',unsafe_allow_html=True)
            st.line_chart(df_h.set_index("frame")[["inf"]],color=["#a78bfa"],height=180,use_container_width=True)
        a3,a4=st.columns(2)
        with a3:
            st.markdown('<div class="slbl">Class frequency</div>',unsafe_allow_html=True)
            if cc:
                df_cls=pd.DataFrame(list(cc.items()),columns=["Class","Count"]).sort_values("Count",ascending=False).head(12)
                st.bar_chart(df_cls.set_index("Class"),color="#a78bfa",height=200,use_container_width=True)
        with a4:
            st.markdown('<div class="slbl">Confidence distribution</div>',unsafe_allow_html=True)
            cd=st.session_state.conf_dist
            cdf=pd.DataFrame({"Range":[f"{k}–{k+9}%" for k in range(0,100,10)],
                               "Count":[cd.get(str(k),0) for k in range(0,100,10)]}).set_index("Range")
            st.bar_chart(cdf,color="#ffbe0b",height=200,use_container_width=True)
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("🔄  Reset analytics"):
            for k in ["history","inf_times","last_dets"]: st.session_state[k]=[]
            st.session_state.total_dets=0; st.session_state.class_counts=defaultdict(int)
            st.session_state.conf_dist=defaultdict(int); st.session_state.session_start=datetime.now()
            st.rerun()
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📊</div>
          <div class="empty-title">No data yet</div>
          <div class="empty-sub">Run detections in the Detect tab first</div>
        </div>""",unsafe_allow_html=True)

# ─────────────────────────── EXPORT ────────────────────────────
with tab_e:
    st.markdown('<div class="slbl">Download options</div>',unsafe_allow_html=True)
    e1,e2,e3=st.columns(3)

    with e1:
        st.markdown('<div class="gcard"><div style="font-size:1.8rem;margin-bottom:8px">📄</div><div style="font-size:.9rem;font-weight:600;color:#e0e0f4;margin-bottom:5px">CSV Export</div><div style="font-size:.72rem;color:#4a4a70">Detection history with timestamps, classes, counts & inference times. Open in Excel or Sheets.</div></div>',unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
        h2=st.session_state.history
        if h2:
            rows=[]
            for snap in h2:
                for lbl,cnt in snap.get("classes",{}).items():
                    rows.append({"timestamp":datetime.fromtimestamp(snap["ts"]).isoformat(),"class":lbl,"count":cnt,"inference_ms":snap["inf"]})
            buf=io.StringIO()
            if rows:
                w=csv.DictWriter(buf,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
            st.download_button("⬇  Download CSV",buf.getvalue(),"visionai_detections.csv","text/csv",use_container_width=True)
        else:
            st.button("⬇  Download CSV",disabled=True,use_container_width=True)

    with e2:
        st.markdown('<div class="gcard"><div style="font-size:1.8rem;margin-bottom:8px">📦</div><div style="font-size:.9rem;font-weight:600;color:#e0e0f4;margin-bottom:5px">JSON Export</div><div style="font-size:.72rem;color:#4a4a70">Full analytics dump with class counts, confidence distribution & frame-level history.</div></div>',unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
        payload={"exported_at":datetime.now().isoformat(),"session_start":st.session_state.session_start.isoformat(),
                  "total_detections":st.session_state.total_dets,"class_counts":dict(st.session_state.class_counts),
                  "conf_distribution":dict(st.session_state.conf_dist),"history":st.session_state.history[-200:]}
        st.download_button("⬇  Download JSON",json.dumps(payload,default=str,indent=2),"visionai_analytics.json","application/json",use_container_width=True)

    with e3:
        st.markdown('<div class="gcard"><div style="font-size:1.8rem;margin-bottom:8px">🖼</div><div style="font-size:.9rem;font-weight:600;color:#e0e0f4;margin-bottom:5px">Annotated Image</div><div style="font-size:.72rem;color:#4a4a70">Last detected frame with bounding boxes and confidence scores drawn on it. Ready for reports.</div></div>',unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
        if st.session_state.last_ann:
            st.download_button("⬇  Save image",st.session_state.last_ann,f"visionai_{datetime.now().strftime('%H%M%S')}.jpg","image/jpeg",use_container_width=True)
        else:
            st.button("⬇  Save image",disabled=True,use_container_width=True)
            st.caption("Run detection first")

    st.markdown('<div class="slbl">Session summary</div>',unsafe_allow_html=True)
    its_e=st.session_state.inf_times
    sdata={"Session start":st.session_state.session_start.strftime("%Y-%m-%d %H:%M:%S"),
            "Total detections":st.session_state.total_dets,"Frames processed":len(st.session_state.history),
            "Avg inference":f"{round(sum(its_e)/max(len(its_e),1),1)}ms" if its_e else "—",
            "Min inference":f"{min(its_e,default=0)}ms","Max inference":f"{max(its_e,default=0)}ms",
            "Unique classes":len(st.session_state.class_counts),
            "Top class":max(st.session_state.class_counts,key=st.session_state.class_counts.get,default="—") if st.session_state.class_counts else "—"}
    st.dataframe(pd.DataFrame(list(sdata.items()),columns=["Metric","Value"]),use_container_width=True,hide_index=True)
