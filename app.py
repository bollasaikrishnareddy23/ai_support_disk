# ============================================================
# AI Customer Support Recovery System
# Streamlit app: Guided chatbot + Manager Command Center + Intelligence Center
# ============================================================

import os
import re
import json
import hmac
import html
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# Optional transformer support. The app still works without these packages.
try:
    from transformers import pipeline
except Exception:
    pipeline = None

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AI Customer Support Recovery System",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Constants
# ============================================================

DATA_FILE = "support_cases.csv"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_MANAGER_PASSWORD = ""
HIGH_RISK_THRESHOLD = 70
MEDIUM_RISK_THRESHOLD = 40

EXPECTED_COLUMNS = [
    "case_id",
    "customer_id",
    "timestamp",
    "last_updated",
    "message",
    "conversation",
    "conversation_summary",
    "clean_text",
    "analysis_source",
    "emotion",
    "emotion_confidence",
    "tone",
    "sarcasm_detected",
    "complaint_topic",
    "topic_confidence",
    "customer_intent",
    "business_risk",
    "risk_score",
    "risk_level",
    "risk_confidence",
    "recovery_score",
    "escalation_trigger",
    "next_best_action",
    "risk_reason",
    "recommended_action",
    "customer_reply",
    "status",
    "assigned_to",
    "coupon_offer",
    "coupon_code",
    "coupon_status",
    "coupon_reason",
    "emotion_journey",
    "topic_journey",
    "risk_journey",
    "tone_journey",
]

ISSUE_TOPICS = [
    "Auto Detect",
    "Delivery Issue",
    "Refund Issue",
    "Billing Issue",
    "Product Issue",
    "Technical Issue",
    "Customer Service Issue",
    "General Complaint",
]

BOT_GREETING = (
    "Hi! I’m your support assistant. Tell me what happened and I’ll help create "
    "a support case for our team."
)

# ============================================================
# Secrets / configuration
# ============================================================

def get_secret_value(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    if value == "":
        value = os.getenv(name, default)
    return str(value)


def bool_secret(name: str, default: bool = False) -> bool:
    value = get_secret_value(name, str(default)).strip().lower()
    return value in ["1", "true", "yes", "on", "enabled"]


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    api_key = get_secret_value("GEMINI_API_KEY", "")
    if genai is None or api_key == "":
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_emotion_transformer():
    if pipeline is None:
        return None
    if not bool_secret("ENABLE_TRANSFORMERS", False):
        return None
    model_name = get_secret_value("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
    try:
        return pipeline("text-classification", model=model_name, top_k=None)
    except Exception:
        return None

# ============================================================
# Utility helpers
# ============================================================

def safe_text(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in ["nan", "none", "null"]:
        return ""
    return html.escape(text)


def display_value(value, fallback: str = "—") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if text == "" or text.lower() in ["nan", "none", "null"]:
        return fallback
    return text


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_case_id() -> str:
    return "CASE-" + datetime.now().strftime("%Y%m%d%H%M%S%f")[-14:]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def append_journey(existing: str, value: str) -> str:
    value = display_value(value, "")
    if value == "":
        return existing or ""
    if not existing or str(existing).lower() in ["nan", "none"]:
        return value
    return f"{existing} → {value}"


def parse_risk_journey(values: str) -> List[int]:
    if not values:
        return []
    nums = []
    for part in str(values).split("→"):
        try:
            nums.append(int(float(part.strip().replace("%", ""))))
        except Exception:
            pass
    return nums

# ============================================================
# CSS / responsive themes
# ============================================================

def apply_theme(theme: str):
    dark = theme == "Dark"
    bg = "#07111f" if dark else "#f4f7fb"
    surface = "rgba(15, 23, 42, 0.78)" if dark else "rgba(255,255,255,0.86)"
    surface2 = "#0f172a" if dark else "#ffffff"
    text = "#f8fafc" if dark else "#0f172a"
    muted = "#94a3b8" if dark else "#64748b"
    border = "rgba(148,163,184,.28)" if dark else "rgba(226,232,240,.95)"
    sidebar = "#020617" if dark else "#0f172a"
    hero_a = "#111827" if dark else "#0f172a"
    hero_b = "#312e81" if dark else "#1d4ed8"
    hero_c = "#0891b2" if dark else "#06b6d4"

    st.markdown(
        f"""
        <style>
            :root {{
                --app-bg: {bg};
                --surface: {surface};
                --surface2: {surface2};
                --text: {text};
                --muted: {muted};
                --border: {border};
                --sidebar: {sidebar};
                --hero-a: {hero_a};
                --hero-b: {hero_b};
                --hero-c: {hero_c};
            }}

            .stApp {{
                background:
                    radial-gradient(circle at 15% 8%, rgba(14,165,233,.18), transparent 28%),
                    radial-gradient(circle at 85% 12%, rgba(168,85,247,.16), transparent 30%),
                    linear-gradient(180deg, var(--app-bg) 0%, var(--app-bg) 100%);
                color: var(--text);
            }}

            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1320px;
            }}

            section[data-testid="stSidebar"] {{
                background: var(--sidebar);
                border-right: 1px solid rgba(148,163,184,.2);
            }}
            section[data-testid="stSidebar"] * {{ color: #f8fafc !important; }}

            .topbar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                margin-bottom: .8rem;
            }}
            .brand-pill {{
                display:inline-flex;
                align-items:center;
                gap:.55rem;
                padding:.6rem .9rem;
                border-radius: 999px;
                background: var(--surface);
                border: 1px solid var(--border);
                color: var(--text);
                box-shadow: 0 12px 32px rgba(15,23,42,.08);
                font-weight: 800;
                letter-spacing: -.01em;
            }}

            .hero {{
                background:
                    linear-gradient(135deg, var(--hero-a), var(--hero-b) 58%, var(--hero-c)),
                    radial-gradient(circle at top left, rgba(255,255,255,.18), transparent 40%);
                color: white;
                padding: clamp(22px, 4vw, 38px);
                border-radius: 30px;
                margin-bottom: 24px;
                box-shadow: 0 24px 70px rgba(15, 23, 42, 0.22);
                position: relative;
                overflow: hidden;
            }}
            .hero:after {{
                content:"";
                position:absolute;
                right:-80px; top:-80px;
                width:260px; height:260px;
                border-radius:50%;
                background: rgba(255,255,255,.12);
            }}
            .hero h1 {{
                font-size: clamp(28px, 5vw, 48px);
                line-height: 1.05;
                margin: 0 0 10px 0;
                font-weight: 900;
                letter-spacing: -0.04em;
                position: relative;
                z-index:1;
            }}
            .hero p {{
                color: rgba(255,255,255,.82);
                font-size: clamp(14px, 2vw, 17px);
                max-width: 880px;
                margin: 0;
                position: relative;
                z-index:1;
            }}

            .glass-card, .panel {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 26px;
                padding: clamp(16px, 2vw, 24px);
                box-shadow: 0 18px 44px rgba(15, 23, 42, 0.09);
                backdrop-filter: blur(14px);
                margin-bottom: 18px;
                color: var(--text);
            }}
            .mini-card {{
                background: var(--surface2);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 18px;
                box-shadow: 0 14px 32px rgba(15,23,42,.08);
                color: var(--text);
            }}
            .section-title {{
                font-size: clamp(18px, 2vw, 23px);
                font-weight: 900;
                color: var(--text);
                margin-bottom: 8px;
                letter-spacing: -.03em;
            }}
            .muted {{
                color: var(--muted);
                font-size: 14px;
                line-height: 1.55;
            }}
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 14px;
                margin-bottom: 18px;
            }}
            .kpi-card {{
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 20px;
                box-shadow: 0 16px 38px rgba(15, 23, 42, 0.08);
                min-height: 112px;
            }}
            .kpi-value {{
                font-size: clamp(24px, 4vw, 35px);
                font-weight: 900;
                color: var(--text);
                letter-spacing: -0.04em;
            }}
            .kpi-label {{
                color: var(--muted);
                font-size: 13px;
                margin-top: 4px;
            }}
            .badge {{
                display: inline-block;
                padding: 6px 11px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 800;
                margin: 3px 4px 3px 0;
            }}
            .badge-high {{ background:#fee2e2; color:#991b1b; }}
            .badge-medium {{ background:#fef3c7; color:#92400e; }}
            .badge-low {{ background:#dcfce7; color:#166534; }}
            .badge-critical {{ background:#7f1d1d; color:white; }}
            .badge-blue {{ background:#dbeafe; color:#1e40af; }}
            .badge-gray {{ background:#f1f5f9; color:#334155; }}
            .badge-purple {{ background:#ede9fe; color:#5b21b6; }}
            .badge-teal {{ background:#ccfbf1; color:#115e59; }}
            .badge-orange {{ background:#ffedd5; color:#9a3412; }}
            .emotion-path {{
                display:flex;
                flex-wrap: wrap;
                align-items:center;
                gap:10px;
                padding: 16px;
                border-radius: 24px;
                background: var(--surface);
                border:1px solid var(--border);
            }}
            .emotion-node {{
                padding: 10px 14px;
                border-radius: 16px;
                font-weight: 850;
                background: linear-gradient(135deg, #dbeafe, #ede9fe);
                color:#1e1b4b;
                box-shadow: 0 10px 24px rgba(15,23,42,.08);
            }}
            .arrow {{ color: var(--muted); font-weight:900; }}
            .chat-shell {{
                border-radius: 30px;
                border:1px solid var(--border);
                background: var(--surface);
                padding: 18px;
                box-shadow: 0 18px 44px rgba(15, 23, 42, 0.09);
            }}
            .suggestion-chip {{
                display:inline-block;
                padding: 8px 12px;
                border-radius:999px;
                background: var(--surface2);
                border:1px solid var(--border);
                color: var(--text);
                margin:4px;
                font-size:13px;
            }}
            div.stButton > button,
            div.stFormSubmitButton > button {{
                border-radius: 16px;
                border: none;
                background: linear-gradient(135deg, #2563eb, #7c3aed);
                color: white;
                font-weight: 850;
                padding: 0.65rem 1.1rem;
                box-shadow: 0 12px 22px rgba(37,99,235,.2);
            }}
            div.stButton > button:hover,
            div.stFormSubmitButton > button:hover {{
                filter: brightness(1.04);
                color: white;
            }}
            .stTextInput input,
            .stTextArea textarea,
            .stSelectbox div,
            .stMultiSelect div {{
                border-radius: 15px !important;
            }}
            #MainMenu {{ visibility: hidden; }}
            footer {{ visibility: hidden; }}

            @media (max-width: 1050px) {{
                .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            }}
            @media (max-width: 700px) {{
                .block-container {{ padding-left: .85rem; padding-right: .85rem; }}
                .hero {{ border-radius: 22px; }}
                .glass-card, .panel, .mini-card {{ border-radius: 20px; }}
                .kpi-grid {{ grid-template-columns: 1fr; }}
                .topbar {{ flex-direction: column; align-items: stretch; }}
                .brand-pill {{ justify-content: center; }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_topbar():
    """Stable top bar with no raw HTML, so it renders cleanly on all pages/devices."""
    left, right = st.columns([0.65, 0.35], vertical_alignment="center")
    with left:
        st.caption("AI Support Recovery System")
    with right:
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.get("theme", "Light") == "Light" else 1,
            key="theme_select",
            label_visibility="collapsed",
        )
        st.session_state.theme = theme


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{safe_text(title)}</h1>
            <p>{safe_text(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_grid(items: List[Tuple[str, str]]):
    """Render KPI cards using native Streamlit containers to avoid raw HTML showing in the UI."""
    if not items:
        return

    max_cols = min(len(items), 5)
    rows = [items[i:i + max_cols] for i in range(0, len(items), max_cols)]

    for row in rows:
        cols = st.columns(len(row), gap="medium")
        for col, (label, value) in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {display_value(value)}")
                    st.caption(display_value(label, ""))


def badge(text: str, kind: str = "gray") -> str:
    return f'<span class="badge badge-{kind}">{safe_text(text)}</span>'


def risk_kind(level: str) -> str:
    level = str(level)
    if level == "High":
        return "high"
    if level == "Medium":
        return "medium"
    return "low"


def priority_kind(priority: str) -> str:
    if str(priority) == "Critical":
        return "critical"
    return risk_kind(priority)

# ============================================================
# Data storage
# ============================================================

def load_records() -> List[Dict]:
    if not os.path.exists(DATA_FILE):
        return []
    try:
        df = pd.read_csv(DATA_FILE, keep_default_na=False)
        if df.empty:
            return []
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[EXPECTED_COLUMNS].to_dict("records")
    except Exception:
        return []


def save_records(records: List[Dict]):
    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[EXPECTED_COLUMNS]
    df = df.fillna("")
    df.to_csv(DATA_FILE, index=False)


def prepare_dataframe(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[EXPECTED_COLUMNS].fillna("")
    if len(df) > 0:
        for col in ["risk_score", "recovery_score"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def clear_records_file():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

# ============================================================
# Classification and intelligence logic
# ============================================================

def is_greeting(text: str) -> bool:
    cleaned = clean_text(text)
    greetings = ["hi", "hai", "hello", "hey", "good morning", "good evening", "good afternoon"]
    return cleaned in greetings or cleaned in ["hii", "helo"]


def is_thanks(text: str) -> bool:
    cleaned = clean_text(text)
    return cleaned in ["thanks", "thank you", "ok", "okay", "cool", "fine", "alright"]


def is_order_reference(text: str) -> bool:
    cleaned = str(text).strip()
    if len(cleaned) > 30:
        return False
    patterns = [
        r"^(ord|order|id|ref|case)[-\s]?[a-zA-Z0-9]{2,20}$",
        r"^[A-Z]{2,5}\d{2,12}$",
        r"^\d{4,14}$",
    ]
    return any(re.match(p, cleaned, flags=re.IGNORECASE) for p in patterns)


def is_manager_escalation_request(text: str) -> bool:
    t = clean_text(text)
    phrases = [
        "manager", "supervisor", "human agent", "real person", "talk to someone",
        "speak to someone", "talk with your manager", "talk to your manager",
        "connect me", "escalate", "higher authority", "senior support",
        "representative", "live agent", "human support",
    ]
    return any(p in t for p in phrases)


def detect_rude_tone(text: str) -> bool:
    t = clean_text(text)
    rude_words = [
        "fuck", "shit", "asshole", "idiot", "stupid", "dumb", "useless", "bullshit",
        "sucks", "shut up", "nonsense", "trash",
    ]
    return any(w in t for w in rude_words)


def detect_sarcasm(text: str) -> bool:
    t = clean_text(text)
    positive_markers = ["great", "amazing", "wonderful", "fantastic", "excellent", "perfect", "lovely", "nice"]
    negative_context = [
        "late", "delayed", "broken", "damaged", "charged", "ignored", "nobody", "no reply",
        "wrong", "error", "crash", "refund", "cancelled", "missing", "not received", "still",
    ]
    return any(p in t for p in positive_markers) and any(n in t for n in negative_context)


def classify_topic(text: str, selected_hint: str = "Auto Detect", previous_topic: str = "") -> Tuple[str, float]:
    t = clean_text(text)

    if selected_hint and selected_hint != "Auto Detect" and selected_hint in ISSUE_TOPICS:
        # Use as gentle context unless the text strongly says otherwise.
        hint_topic = selected_hint
    else:
        hint_topic = ""

    delivery = ["delivery", "shipping", "shipment", "late", "delay", "delayed", "package", "order", "where is", "tracking", "arrive", "arrived", "eta", "possible date", "when", "missing", "not received", "delivered"]
    refund = ["refund", "money back", "return", "chargeback", "refund status"]
    billing = ["charged", "payment", "billing", "bill", "invoice", "paid", "card", "transaction", "charged twice", "double charge"]
    product = ["broken", "damaged", "defective", "faulty", "quality", "wrong product", "wrong item", "replacement", "item broke"]
    service = ["support", "customer service", "agent", "representative", "no reply", "nobody helped", "ignored", "rude agent"]
    technical = ["app", "website", "login", "password", "error", "crash", "bug", "not working", "technical", "reset"]

    scores = {
        "Delivery Issue": sum(1 for w in delivery if w in t),
        "Refund Issue": sum(1 for w in refund if w in t),
        "Billing Issue": sum(1 for w in billing if w in t),
        "Product Issue": sum(1 for w in product if w in t),
        "Customer Service Issue": sum(1 for w in service if w in t),
        "Technical Issue": sum(1 for w in technical if w in t),
    }

    # Common correction: "where is my product" is delivery/tracking, not product quality.
    if "where is" in t or "possible date" in t or "eta" in t:
        scores["Delivery Issue"] += 3

    # If customer sends only a reference, keep previous topic if known.
    if is_order_reference(text) and previous_topic:
        return previous_topic, 0.70

    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]

    if best_score == 0 and hint_topic:
        return hint_topic, 0.58
    if best_score == 0 and previous_topic:
        return previous_topic, 0.52
    if best_score == 0:
        return "General Complaint", 0.45

    confidence = min(0.95, 0.58 + best_score * 0.12)
    return best_topic, confidence


def detect_emotion_rules(text: str, tone: str, topic: str) -> Tuple[str, float]:
    t = clean_text(text)
    if is_greeting(text) or is_thanks(text):
        return "Neutral", 0.70
    if tone in ["Rude", "Sarcastic"]:
        return "Angry", 0.85

    angry_words = ["angry", "furious", "mad", "worst", "terrible", "awful", "hate", "unacceptable", "ridiculous", "never buy"]
    disappointed_words = ["disappointed", "upset", "frustrated", "annoyed", "unhappy", "poor", "bad experience", "not happy"]
    confused_words = ["confused", "unclear", "do not understand", "dont understand", "why", "how", "what happened"]
    positive_words = ["thank", "thanks", "appreciate", "great help", "excellent support", "resolved", "solved", "happy"]

    if any(w in t for w in angry_words):
        return "Angry", 0.86
    if any(w in t for w in disappointed_words):
        return "Disappointed", 0.78
    if any(w in t for w in confused_words):
        return "Confused", 0.70
    if any(w in t for w in positive_words):
        return "Satisfied", 0.82

    # Complaint topics are rarely neutral.
    if topic in ["Refund Issue", "Billing Issue", "Delivery Issue", "Customer Service Issue", "Product Issue", "Technical Issue"]:
        return "Disappointed", 0.66
    return "Neutral", 0.55


def detect_emotion_transformer(text: str) -> Optional[Tuple[str, float, str]]:
    model = load_emotion_transformer()
    if model is None:
        return None
    try:
        result = model(text)
        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]
        top = max(result, key=lambda x: x.get("score", 0))
        label = str(top.get("label", "neutral")).lower()
        score = float(top.get("score", 0))
        mapping = {
            "anger": "Angry",
            "disgust": "Angry",
            "fear": "Confused",
            "joy": "Satisfied",
            "neutral": "Neutral",
            "sadness": "Disappointed",
            "surprise": "Confused",
        }
        return mapping.get(label, "Neutral"), score, label
    except Exception:
        return None


def detect_tone(text: str) -> str:
    if is_greeting(text):
        return "Polite"
    if is_thanks(text):
        return "Positive"
    if detect_rude_tone(text):
        return "Rude"
    if detect_sarcasm(text):
        return "Sarcastic"
    t = clean_text(text)
    if any(w in t for w in ["angry", "furious", "mad", "unacceptable", "worst"]):
        return "Angry"
    if any(w in t for w in ["confused", "why", "how", "what happened"]):
        return "Confused"
    if any(w in t for w in ["thank", "thanks", "appreciate", "happy", "solved"]):
        return "Positive"
    return "Polite"


def get_risk_level(score: int) -> str:
    if score >= HIGH_RISK_THRESHOLD:
        return "High"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def calculate_risk(
    emotion: str,
    tone: str,
    topic: str,
    text: str,
    sarcasm: bool,
    manager_request: bool,
) -> Tuple[int, str, str]:
    t = clean_text(text)
    base = {"Satisfied": 5, "Neutral": 15, "Confused": 30, "Disappointed": 45, "Angry": 72}.get(emotion, 20)
    reasons = [f"Emotion detected as {emotion}."]

    if topic in ["Refund Issue", "Billing Issue"]:
        base += 10
        reasons.append("Money-related complaint.")
    if topic == "Customer Service Issue":
        base += 8
        reasons.append("Support experience problem.")
    if topic == "Delivery Issue" and any(w in t for w in ["late", "delayed", "missing", "not received", "where is"]):
        base += 8
        reasons.append("Delivery delay or missing package signal.")
    if any(w in t for w in ["refund", "money back", "cancel", "cancellation", "never buy"]):
        base += 12
        reasons.append("Refund, cancellation, or churn language detected.")
    if any(w in t for w in ["again", "still", "no reply", "nobody", "ignored", "waiting", "three times"]):
        base += 10
        reasons.append("Repeated or unresolved issue detected.")
    if sarcasm:
        base += 10
        reasons.append("Sarcastic dissatisfaction detected.")
    if tone == "Rude":
        base += 12
        reasons.append("Rude or abusive tone detected.")
    if manager_request:
        base += 15
        reasons.append("Customer requested manager escalation.")

    score = max(0, min(int(base), 100))
    confidence = "High" if len(reasons) >= 3 else "Medium" if len(reasons) == 2 else "Low"
    return score, " ".join(reasons), confidence


def calculate_recovery_score(risk_score: int, tone: str, manager_request: bool, status: str = "") -> int:
    score = 82
    if risk_score >= 85:
        score -= 22
    elif risk_score >= 70:
        score -= 15
    elif risk_score >= 40:
        score -= 6
    if tone == "Rude":
        score -= 12
    if tone == "Sarcastic":
        score -= 8
    if manager_request:
        score -= 10
    if status == "Resolved":
        score += 15
    return max(5, min(98, score))


def next_best_action(topic: str, risk_level: str, tone: str, manager_request: bool, has_ref: bool) -> str:
    if manager_request:
        return "Manager follow-up required"
    if risk_level == "High" and topic in ["Refund Issue", "Billing Issue"]:
        return "Assign to Billing Team and review refund/payment details"
    if risk_level == "High" and topic == "Delivery Issue":
        return "Assign to Delivery Team and check tracking status"
    if risk_level == "High" and tone in ["Rude", "Sarcastic", "Angry"]:
        return "Escalate carefully with calm service recovery response"
    if topic == "Delivery Issue" and not has_ref:
        return "Request order ID or tracking reference"
    if topic == "Technical Issue":
        return "Assign to Technical Team"
    if topic == "Product Issue":
        return "Review replacement or product support option"
    return "Support team should review and follow up"


def summarize_conversation(chat_messages: List[Dict], latest_topic: str, risk_level: str) -> str:
    user_msgs = [m.get("content", "") for m in chat_messages if m.get("role") == "user"]
    if not user_msgs:
        return "No customer issue submitted yet."
    first = user_msgs[0]
    latest = user_msgs[-1]
    if len(user_msgs) == 1:
        return f"Customer reported a {latest_topic.lower()} issue. Latest risk level is {risk_level}."
    return f"Customer conversation is about {latest_topic.lower()}. Latest message: {latest[:140]}. Current risk level is {risk_level}."


def format_conversation(chat_messages: List[Dict]) -> str:
    lines = []
    for msg in chat_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"Customer: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n\n".join(lines)


def get_previous_case_context() -> Dict[str, str]:
    case_id = st.session_state.get("active_case_id", "")
    if not case_id:
        return {}
    for record in load_records():
        if record.get("case_id") == case_id:
            return record
    return {}


def generate_customer_reply(
    message: str,
    topic: str,
    tone: str,
    manager_request: bool,
    is_reference: bool,
    previous_topic: str = "",
) -> str:
    t = clean_text(message)

    if is_greeting(message):
        return "Hi! How can I help you today? You can tell me about a delivery, refund, billing, product, technical, or support issue."
    if is_thanks(message):
        return "You’re welcome. Is there anything else you want me to add to this support case?"
    if manager_request:
        return "I understand. I’ve marked your case for manager review so a manager or senior support member can review it."
    if is_reference:
        if previous_topic == "Delivery Issue":
            return f"Thank you. I’ve noted the reference {message.strip()} so the delivery team can check the tracking status."
        return f"Thank you. I’ve added the reference {message.strip()} to your support case."
    if tone == "Rude":
        return "I understand you’re upset. I’m here to help, but please describe the issue clearly so I can record it properly for our team."
    if tone == "Sarcastic":
        return "I’m sorry this has been frustrating. I’ve recorded the issue and the team will review what went wrong."

    # Context-aware follow-ups.
    if any(w in t for w in ["possible date", "eta", "when will", "when", "date"]):
        if previous_topic == "Delivery Issue" or topic == "Delivery Issue":
            return "I understand you want the expected delivery date. I’ve added that to your case so the delivery team can check the latest status."

    if topic == "Delivery Issue":
        if "where is" in t or "missing" in t or "not received" in t:
            return "I’m sorry you haven’t received your package. I’ve recorded this as a delivery issue so the team can check the tracking status."
        return "I’m sorry about the delivery problem. I’ve recorded the details so the delivery team can review the delay or shipment status."
    if topic == "Refund Issue":
        return "I understand your concern about the refund. I’ve recorded your refund request so the team can review the order and follow up with the next steps."
    if topic == "Billing Issue":
        return "Thank you for reporting the billing issue. I’ve recorded it so the billing team can review the payment details carefully."
    if topic == "Product Issue":
        return "I’m sorry about the product issue. I’ve recorded the details so the team can review replacement, return, or support options."
    if topic == "Technical Issue":
        return "Thank you for reporting the technical issue. I’ve recorded the problem so the technical team can review it."
    if topic == "Customer Service Issue":
        return "I’m sorry about the support experience. I’ve recorded this so the team can review the service issue."

    if len(t.split()) <= 3:
        return "Can you share a little more detail about the issue so I can record it correctly for our team?"
    return "Thank you for sharing this. I’ve added the details to your support case so our team can review it."


def analyze_message(message: str, selected_hint: str = "Auto Detect") -> Dict:
    previous = get_previous_case_context()
    previous_topic = display_value(previous.get("complaint_topic", ""), "")
    tone = detect_tone(message)
    sarcasm = detect_sarcasm(message)
    manager_request = is_manager_escalation_request(message)
    is_reference = is_order_reference(message)

    topic, topic_confidence = classify_topic(message, selected_hint, previous_topic)

    transformer_emotion = detect_emotion_transformer(message)
    if transformer_emotion and transformer_emotion[1] >= 0.55:
        emotion, emotion_confidence, raw_label = transformer_emotion
        analysis_source = f"Transformer + Rules ({raw_label})"
    else:
        emotion, emotion_confidence = detect_emotion_rules(message, tone, topic)
        analysis_source = "Rules"

    # Strong corrections.
    if manager_request or tone in ["Rude", "Sarcastic"]:
        emotion = "Angry"
        emotion_confidence = max(emotion_confidence, 0.82)
    if topic != "General Complaint" and emotion == "Neutral" and not is_greeting(message) and not is_thanks(message):
        emotion = "Disappointed"
        emotion_confidence = max(emotion_confidence, 0.65)

    risk_score, risk_reason, risk_confidence = calculate_risk(
        emotion, tone, topic, message, sarcasm, manager_request
    )
    risk_level = get_risk_level(risk_score)
    recovery = calculate_recovery_score(risk_score, tone, manager_request)

    escalation_trigger = ""
    if manager_request:
        escalation_trigger = "Customer requested manager escalation"
    elif risk_level == "High":
        escalation_trigger = risk_reason

    action = next_best_action(topic, risk_level, tone, manager_request, is_reference)
    customer_reply = generate_customer_reply(message, topic, tone, manager_request, is_reference, previous_topic)

    if manager_request:
        status = "Escalated"
        assigned = "Manager"
    else:
        status = previous.get("status", "New") or "New"
        assigned = previous.get("assigned_to", "Unassigned") or "Unassigned"

    return {
        "clean_text": clean_text(message),
        "analysis_source": analysis_source,
        "emotion": emotion,
        "emotion_confidence": round(float(emotion_confidence), 2),
        "tone": tone,
        "sarcasm_detected": "Yes" if sarcasm else "No",
        "complaint_topic": topic,
        "topic_confidence": round(float(topic_confidence), 2),
        "customer_intent": infer_customer_intent(topic, message, manager_request),
        "business_risk": infer_business_risk(topic, risk_level, tone, manager_request),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_confidence": risk_confidence,
        "recovery_score": recovery,
        "escalation_trigger": escalation_trigger,
        "next_best_action": action,
        "risk_reason": risk_reason,
        "recommended_action": action,
        "customer_reply": customer_reply,
        "status": status,
        "assigned_to": assigned,
    }


def infer_customer_intent(topic: str, message: str, manager_request: bool) -> str:
    if manager_request:
        return "Customer wants manager or human escalation."
    if topic == "Delivery Issue":
        return "Customer wants delivery status, tracking, or an expected delivery date."
    if topic == "Refund Issue":
        return "Customer wants refund review or money-related resolution."
    if topic == "Billing Issue":
        return "Customer wants payment or billing issue reviewed."
    if topic == "Product Issue":
        return "Customer wants product support, replacement, or return help."
    if topic == "Technical Issue":
        return "Customer wants technical troubleshooting or account/app support."
    if topic == "Customer Service Issue":
        return "Customer wants better support response or service follow-up."
    return "Customer wants the issue reviewed by support."


def infer_business_risk(topic: str, risk_level: str, tone: str, manager_request: bool) -> str:
    if manager_request:
        return "Escalation risk: customer explicitly requested manager review."
    if risk_level == "High":
        return "High risk of churn, refund demand, negative feedback, or escalation."
    if tone in ["Rude", "Sarcastic", "Angry"]:
        return "Customer dissatisfaction may escalate if follow-up is delayed."
    if topic in ["Refund Issue", "Billing Issue"]:
        return "Money-related complaint may affect trust and retention."
    if topic == "Delivery Issue":
        return "Delivery uncertainty may reduce customer confidence."
    return "Low to moderate business risk."

# ============================================================
# Coupon recovery logic
# ============================================================

def generate_coupon_code(case_id: str, discount_percent: int) -> str:
    short_id = str(case_id).replace("CASE-", "")[-6:]
    return f"REC-{short_id}-{discount_percent}"


def suggest_recovery_coupon(risk_level: str, risk_score: int, topic: str, case_id: str, manager_request: bool) -> Dict[str, str]:
    if manager_request and int(risk_score) >= 75:
        discount = 20
        return {
            "coupon_offer": "20% manager-approved recovery coupon",
            "coupon_code": generate_coupon_code(case_id, discount),
            "coupon_status": "Suggested",
            "coupon_reason": "Manager escalation and high-risk dissatisfaction detected.",
        }
    if risk_level == "High":
        discount = 20 if int(risk_score) >= 85 else 15
        if topic in ["Refund Issue", "Billing Issue"]:
            offer = f"{discount}% goodwill coupon after billing/refund review"
            reason = "High-risk money-related complaint. Offer only after review."
        elif topic == "Delivery Issue":
            offer = f"{discount}% delivery recovery coupon or free shipping offer"
            reason = "High-risk delivery issue. Recovery offer may help retain customer."
        elif topic == "Product Issue":
            offer = f"{discount}% product recovery coupon"
            reason = "High-risk product issue. Coupon may support recovery after resolution."
        else:
            offer = f"{discount}% customer recovery coupon"
            reason = "High-risk customer dissatisfaction."
        return {
            "coupon_offer": offer,
            "coupon_code": generate_coupon_code(case_id, discount),
            "coupon_status": "Suggested",
            "coupon_reason": reason,
        }
    if risk_level == "Medium":
        return {
            "coupon_offer": "10% optional goodwill coupon if unresolved",
            "coupon_code": generate_coupon_code(case_id, 10),
            "coupon_status": "Optional",
            "coupon_reason": "Medium-risk case; coupon only if support follow-up is delayed.",
        }
    return {
        "coupon_offer": "No coupon needed",
        "coupon_code": "",
        "coupon_status": "Not Required",
        "coupon_reason": "Low-risk case; normal support process is enough.",
    }

# ============================================================
# Case update helpers
# ============================================================

def get_active_case_id() -> str:
    if not st.session_state.get("active_case_id"):
        st.session_state.active_case_id = create_case_id()
    return st.session_state.active_case_id


def reset_chat_for_customer(customer_id: str):
    st.session_state.chat_customer_id = customer_id
    st.session_state.active_case_id = ""
    st.session_state.chat_messages = [{"role": "assistant", "content": BOT_GREETING}]
    st.session_state.selected_issue_type = "Auto Detect"
    st.session_state.last_case_id = ""


def create_or_update_case(customer_id: str, message: str, analysis: Dict, coupon: Dict) -> str:
    case_id = get_active_case_id()
    records = load_records()
    now = now_str()
    conversation = format_conversation(st.session_state.chat_messages)
    summary = summarize_conversation(st.session_state.chat_messages, analysis["complaint_topic"], analysis["risk_level"])

    found = False
    for record in records:
        if record.get("case_id") == case_id:
            record.update({
                "customer_id": customer_id,
                "last_updated": now,
                "message": message,
                "conversation": conversation,
                "conversation_summary": summary,
                "clean_text": analysis["clean_text"],
                "analysis_source": analysis["analysis_source"],
                "emotion": analysis["emotion"],
                "emotion_confidence": analysis["emotion_confidence"],
                "tone": analysis["tone"],
                "sarcasm_detected": analysis["sarcasm_detected"],
                "complaint_topic": analysis["complaint_topic"],
                "topic_confidence": analysis["topic_confidence"],
                "customer_intent": analysis["customer_intent"],
                "business_risk": analysis["business_risk"],
                "risk_score": analysis["risk_score"],
                "risk_level": analysis["risk_level"],
                "risk_confidence": analysis["risk_confidence"],
                "recovery_score": analysis["recovery_score"],
                "escalation_trigger": analysis["escalation_trigger"],
                "next_best_action": analysis["next_best_action"],
                "risk_reason": analysis["risk_reason"],
                "recommended_action": analysis["recommended_action"],
                "customer_reply": analysis["customer_reply"],
                "status": analysis.get("status", record.get("status", "New")),
                "assigned_to": analysis.get("assigned_to", record.get("assigned_to", "Unassigned")),
                "emotion_journey": append_journey(record.get("emotion_journey", ""), analysis["emotion"]),
                "topic_journey": append_journey(record.get("topic_journey", ""), analysis["complaint_topic"]),
                "risk_journey": append_journey(record.get("risk_journey", ""), str(analysis["risk_score"])),
                "tone_journey": append_journey(record.get("tone_journey", ""), analysis["tone"]),
            })
            current_coupon_status = str(record.get("coupon_status", ""))
            if current_coupon_status in ["", "Not Required", "Optional", "Suggested"]:
                record["coupon_offer"] = coupon["coupon_offer"]
                record["coupon_code"] = coupon["coupon_code"]
                record["coupon_status"] = coupon["coupon_status"]
                record["coupon_reason"] = coupon["coupon_reason"]
            found = True
            break

    if not found:
        new_record = {col: "" for col in EXPECTED_COLUMNS}
        new_record.update({
            "case_id": case_id,
            "customer_id": customer_id,
            "timestamp": now,
            "last_updated": now,
            "message": message,
            "conversation": conversation,
            "conversation_summary": summary,
            "clean_text": analysis["clean_text"],
            "analysis_source": analysis["analysis_source"],
            "emotion": analysis["emotion"],
            "emotion_confidence": analysis["emotion_confidence"],
            "tone": analysis["tone"],
            "sarcasm_detected": analysis["sarcasm_detected"],
            "complaint_topic": analysis["complaint_topic"],
            "topic_confidence": analysis["topic_confidence"],
            "customer_intent": analysis["customer_intent"],
            "business_risk": analysis["business_risk"],
            "risk_score": analysis["risk_score"],
            "risk_level": analysis["risk_level"],
            "risk_confidence": analysis["risk_confidence"],
            "recovery_score": analysis["recovery_score"],
            "escalation_trigger": analysis["escalation_trigger"],
            "next_best_action": analysis["next_best_action"],
            "risk_reason": analysis["risk_reason"],
            "recommended_action": analysis["recommended_action"],
            "customer_reply": analysis["customer_reply"],
            "status": analysis.get("status", "New"),
            "assigned_to": analysis.get("assigned_to", "Unassigned"),
            "coupon_offer": coupon["coupon_offer"],
            "coupon_code": coupon["coupon_code"],
            "coupon_status": coupon["coupon_status"],
            "coupon_reason": coupon["coupon_reason"],
            "emotion_journey": analysis["emotion"],
            "topic_journey": analysis["complaint_topic"],
            "risk_journey": str(analysis["risk_score"]),
            "tone_journey": analysis["tone"],
        })
        records.append(new_record)

    save_records(records)
    st.session_state.records = records
    return case_id


def update_case(case_id: str, updates: Dict[str, str]):
    records = load_records()
    for record in records:
        if record.get("case_id") == case_id:
            record.update(updates)
            record["last_updated"] = now_str()
            if "risk_score" in record:
                try:
                    record["risk_level"] = get_risk_level(int(record["risk_score"]))
                except Exception:
                    pass
            break
    save_records(records)
    st.session_state.records = records

# ============================================================
# Manager data helpers
# ============================================================

def enrich_manager_data(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) == 0:
        return data
    data = data.copy()
    data["timestamp_dt"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["last_updated_dt"] = pd.to_datetime(data["last_updated"], errors="coerce")
    now = pd.Timestamp.now()
    data["case_age_hours"] = ((now - data["timestamp_dt"]).dt.total_seconds() / 3600).fillna(0).clip(lower=0).round(1)
    data["risk_score"] = pd.to_numeric(data["risk_score"], errors="coerce").fillna(0).astype(int)
    data["recovery_score"] = pd.to_numeric(data["recovery_score"], errors="coerce").fillna(0).astype(int)
    data["priority_score"] = data["risk_score"].copy()
    data.loc[data["status"] == "Escalated", "priority_score"] += 12
    data.loc[data["tone"].isin(["Rude", "Sarcastic"]), "priority_score"] += 5
    data["priority_score"] = data["priority_score"].clip(upper=100)

    def get_priority(score):
        if score >= 88:
            return "Critical"
        if score >= 70:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"

    def get_sla(row):
        if row.get("status") == "Resolved":
            return "Closed"
        age = float(row.get("case_age_hours", 0) or 0)
        risk = row.get("risk_level", "Low")
        if risk == "High" and age > 24:
            return "Overdue"
        if risk == "Medium" and age > 48:
            return "Overdue"
        if risk == "Low" and age > 72:
            return "Overdue"
        return "On Track"

    data["priority"] = data["priority_score"].apply(get_priority)
    data["sla_status"] = data.apply(get_sla, axis=1)
    return data

# ============================================================
# Optional Gemini insight generator for Intelligence Center
# ============================================================

def generate_intelligence_summary(data: pd.DataFrame) -> str:
    if len(data) == 0:
        return "No customer data available yet."

    fallback = []
    top_topic = data["complaint_topic"].value_counts().idxmax() if len(data["complaint_topic"].dropna()) else "Unknown"
    high_count = len(data[data["risk_level"] == "High"])
    escalated_count = len(data[data["status"] == "Escalated"])
    avg_risk = round(data["risk_score"].mean(), 1)
    fallback.append(f"The main complaint driver is {top_topic}.")
    fallback.append(f"There are {high_count} high-risk customers and {escalated_count} escalated customers.")
    fallback.append(f"Average risk score is {avg_risk}.")
    fallback.append("Managers should prioritize escalated, high-risk, and money-related cases first.")
    fallback_text = " ".join(fallback)

    client = get_gemini_client()
    if client is None:
        return fallback_text

    compact = data[[
        "customer_id", "complaint_topic", "emotion", "tone", "risk_score",
        "risk_level", "status", "coupon_status", "next_best_action"
    ]].tail(30).to_dict("records")
    prompt = f"""
You are a customer support operations analyst.
Create a concise manager insight summary from this support data.
Mention complaint drivers, risk pattern, escalation pattern, and recommended focus.
Keep it under 120 words.
Data:
{json.dumps(compact, indent=2)}
"""
    try:
        if types is not None:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
        else:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text.strip() if response.text else fallback_text
    except Exception:
        return fallback_text

# ============================================================
# Authentication
# ============================================================

def get_manager_password() -> str:
    return get_secret_value("MANAGER_PASSWORD", DEFAULT_MANAGER_PASSWORD)


def check_manager_access() -> bool:
    if st.session_state.get("manager_authenticated", False):
        return True
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Manager Access</div>
            <div class="muted">This workspace is protected for internal support managers.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    password = st.text_input("Enter manager password", type="password")
    if st.button("Login"):
        correct_password = get_manager_password()
        if correct_password == "":
            st.error("MANAGER_PASSWORD is not configured in Streamlit Secrets.")
            return False
        if hmac.compare_digest(password, correct_password):
            st.session_state.manager_authenticated = True
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

# ============================================================
# Session state
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "records" not in st.session_state:
    st.session_state.records = load_records()
if "manager_authenticated" not in st.session_state:
    st.session_state.manager_authenticated = False
if "chat_customer_id" not in st.session_state:
    st.session_state.chat_customer_id = "C001"
if "previous_customer_id" not in st.session_state:
    st.session_state.previous_customer_id = "C001"
if "active_case_id" not in st.session_state:
    st.session_state.active_case_id = ""
if "selected_issue_type" not in st.session_state:
    st.session_state.selected_issue_type = "Auto Detect"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": BOT_GREETING}]
if "last_case_id" not in st.session_state:
    st.session_state.last_case_id = ""

apply_theme(st.session_state.theme)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("AI Support Recovery")
st.sidebar.caption("Customer intelligence workspace")

page = st.sidebar.radio(
    "Navigation",
    [
        "AI Support Chat",
        "Manager Command Center",
        "Intelligence Center",
        "Customer Emotion Journey",
        "About System",
    ],
)

st.sidebar.markdown("---")
if st.session_state.get("manager_authenticated", False):
    if st.sidebar.button("Logout Manager"):
        st.session_state.manager_authenticated = False
        st.rerun()

# ============================================================
# Page: AI Support Chat
# ============================================================

if page == "AI Support Chat":
    render_topbar()
    hero(
        "AI Guided Support Assistant",
        "A responsive customer support chatbot that records one active case per customer conversation and alerts managers when risk increases.",
    )

    # Customer Context first, Smart Flow second, conversation third.
    context_col, flow_col = st.columns([0.58, 0.42], gap="large")

    with context_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Customer Context</div>
                <div class="muted">Enter a customer ID or order ID. Changing this ID automatically starts a fresh conversation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        new_customer_id = st.text_input(
            "Customer ID or Order ID",
            value=st.session_state.chat_customer_id,
            placeholder="Example: C001 or ORD123",
        )
        if new_customer_id != st.session_state.previous_customer_id:
            reset_chat_for_customer(new_customer_id)
            st.session_state.previous_customer_id = new_customer_id
            st.rerun()
        st.session_state.chat_customer_id = new_customer_id

        if st.session_state.active_case_id:
            st.success(f"Active case: {st.session_state.active_case_id}")
        else:
            st.info("No active support case yet. Say hi or describe an issue to begin.")

    with flow_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Smart Support Flow</div>
                <div class="muted">Choose a topic hint if you know the issue type. The assistant will still auto-correct based on the customer message.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.selected_issue_type = st.selectbox(
            "Issue hint",
            ISSUE_TOPICS,
            index=ISSUE_TOPICS.index(st.session_state.selected_issue_type)
            if st.session_state.selected_issue_type in ISSUE_TOPICS
            else 0,
        )
        if st.button("Start New Conversation"):
            reset_chat_for_customer(st.session_state.chat_customer_id)
            st.rerun()

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Support Conversation</div>
            <div class="muted">The bot greets customers, asks for details when needed, escalates manager requests, and keeps coupon logic hidden from customers.</div>
            <div style="margin-top:10px">
                <span class="suggestion-chip">where is my product?</span>
                <span class="suggestion-chip">I want a refund</span>
                <span class="suggestion-chip">I need to talk with your manager</span>
                <span class="suggestion-chip">Great, my package is only 10 days late</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    for chat in st.session_state.chat_messages:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    user_message = st.chat_input("Type your message here...")
    if user_message:
        st.session_state.chat_messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.write(user_message)

        # Greetings/thanks do not create a case unless there is already an active case.
        if (is_greeting(user_message) or is_thanks(user_message)) and not st.session_state.active_case_id:
            reply = generate_customer_reply(user_message, "General Complaint", detect_tone(user_message), False, False, "")
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)
        else:
            analysis = analyze_message(user_message, st.session_state.selected_issue_type)
            case_id = get_active_case_id()
            coupon = suggest_recovery_coupon(
                analysis["risk_level"],
                analysis["risk_score"],
                analysis["complaint_topic"],
                case_id,
                is_manager_escalation_request(user_message),
            )
            reply = analysis["customer_reply"]
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            case_id = create_or_update_case(st.session_state.chat_customer_id, user_message, analysis, coupon)
            st.session_state.last_case_id = case_id
            with st.chat_message("assistant"):
                st.write(reply)
                st.caption(f"Support case: {case_id}")

# ============================================================
# Page: Manager Command Center
# ============================================================

elif page == "Manager Command Center":
    render_topbar()
    if not check_manager_access():
        st.stop()

    hero(
        "Manager Command Center",
        "Customer-level support operations: prioritize risk, review internal intelligence, and approve recovery offers without exposing coupon logic to customers.",
    )

    data = enrich_manager_data(prepare_dataframe(load_records()))
    if len(data) == 0:
        st.info("No support cases yet.")
    else:
        kpi_grid([
            ("Customers", str(data["customer_id"].nunique())),
            ("High Risk", str(len(data[data["risk_level"] == "High"]))),
            ("Escalated", str(len(data[data["status"] == "Escalated"]))),
            ("Avg Risk", str(round(data["risk_score"].mean(), 1))),
            ("Recovery Offers", str(len(data[data["coupon_status"].isin(["Suggested", "Optional", "Approved"])]))),
        ])

        tab1, tab2, tab3 = st.tabs(["Priority Queue", "Case Review", "Recovery Offers"])

        with tab1:
            st.subheader("Customer Priority Queue")
            f1, f2, f3 = st.columns(3)
            with f1:
                risk_filter = st.multiselect("Risk", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
            with f2:
                status_filter = st.multiselect("Status", sorted(data["status"].unique()), default=sorted(data["status"].unique()))
            with f3:
                tone_filter = st.multiselect("Tone", sorted(data["tone"].unique()), default=sorted(data["tone"].unique()))

            filtered = data[
                data["risk_level"].isin(risk_filter)
                & data["status"].isin(status_filter)
                & data["tone"].isin(tone_filter)
            ].sort_values(["priority_score", "case_age_hours"], ascending=[False, False])

            cols = [
                "customer_id", "complaint_topic", "emotion", "tone", "risk_score", "risk_level",
                "recovery_score", "status", "assigned_to", "next_best_action", "coupon_status", "last_updated"
            ]
            st.dataframe(filtered[cols], use_container_width=True, hide_index=True)

        with tab2:
            st.subheader("Case Review")
            review_data = data.sort_values(["priority_score", "case_age_hours"], ascending=[False, False])
            selected_idx = st.selectbox(
                "Select customer",
                review_data.index.tolist(),
                format_func=lambda idx: (
                    f"{display_value(data.loc[idx, 'customer_id'])} · "
                    f"{display_value(data.loc[idx, 'risk_level'])} Risk · "
                    f"{display_value(data.loc[idx, 'complaint_topic'])}"
                ),
            )
            record = data.loc[selected_idx].to_dict()
            badges = "".join([
                badge(display_value(record.get("risk_level"), "Low") + " Risk", risk_kind(record.get("risk_level", "Low"))),
                badge(display_value(record.get("priority"), "Low") + " Priority", priority_kind(record.get("priority", "Low"))),
                badge(display_value(record.get("status"), "New"), "blue"),
                badge(display_value(record.get("tone"), "Polite"), "purple"),
                badge(display_value(record.get("coupon_status"), "Not Required"), "orange"),
            ])
            st.markdown(
                f"""
                <div class="panel">
                    <div class="section-title">{safe_text(record.get('customer_id'))}</div>
                    <div class="muted">Last updated: {safe_text(display_value(record.get('last_updated')))}</div>
                    <div style="margin-top:10px">{badges}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([0.52, 0.48], gap="large")
            with c1:
                st.markdown("### Case Summary")
                st.write(f"**Topic:** {display_value(record.get('complaint_topic'))}")
                st.write(f"**Latest message:** {display_value(record.get('message'))}")
                st.write(f"**Conversation summary:** {display_value(record.get('conversation_summary'))}")
                st.write(f"**Status:** {display_value(record.get('status'))}")
                st.write(f"**Assigned to:** {display_value(record.get('assigned_to'))}")
                st.write(f"**Next best action:** {display_value(record.get('next_best_action'))}")
            with c2:
                st.markdown("### Internal Intelligence")
                st.write(f"**Emotion:** {display_value(record.get('emotion'))} ({display_value(record.get('emotion_confidence'))})")
                st.write(f"**Tone:** {display_value(record.get('tone'))}")
                st.write(f"**Sarcasm:** {display_value(record.get('sarcasm_detected'))}")
                st.write(f"**Risk:** {display_value(record.get('risk_score'))}% · {display_value(record.get('risk_level'))}")
                st.write(f"**Recovery Score:** {display_value(record.get('recovery_score'))}%")
                st.write(f"**Trigger:** {display_value(record.get('escalation_trigger'))}")
                st.write(f"**Risk reason:** {display_value(record.get('risk_reason'))}")

            st.markdown("---")
            st.markdown("### Recovery Offer")
            coupon_status = display_value(record.get("coupon_status"), "Not Required")
            coupon_code = display_value(record.get("coupon_code"), "")
            st.write(f"**Offer:** {display_value(record.get('coupon_offer'))}")
            st.write(f"**Reason:** {display_value(record.get('coupon_reason'))}")
            st.write(f"**Status:** {coupon_status}")
            if coupon_code:
                st.code(coupon_code)
            b1, b2, b3 = st.columns(3)
            with b1:
                if coupon_status in ["Suggested", "Optional"] and st.button("Approve Offer", key=f"approve_{record.get('case_id')}"):
                    update_case(record["case_id"], {"coupon_status": "Approved"})
                    st.success("Offer approved.")
                    st.rerun()
            with b2:
                if coupon_status in ["Suggested", "Optional", "Approved"] and st.button("Reject Offer", key=f"reject_{record.get('case_id')}"):
                    update_case(record["case_id"], {"coupon_status": "Rejected"})
                    st.info("Offer rejected.")
                    st.rerun()
            with b3:
                if coupon_status == "Approved" and st.button("Mark Sent", key=f"sent_{record.get('case_id')}"):
                    update_case(record["case_id"], {"coupon_status": "Sent", "status": "Resolved"})
                    st.success("Offer marked sent and case resolved.")
                    st.rerun()

        with tab3:
            st.subheader("Recovery Offers")
            offer_data = data[data["coupon_status"].isin(["Suggested", "Optional", "Approved", "Sent", "Rejected"])]
            if len(offer_data) == 0:
                st.info("No recovery offers yet.")
            else:
                st.dataframe(
                    offer_data[["customer_id", "complaint_topic", "risk_score", "coupon_offer", "coupon_code", "coupon_status", "status"]],
                    use_container_width=True,
                    hide_index=True,
                )

# ============================================================
# Page: Intelligence Center
# ============================================================

elif page == "Intelligence Center":
    render_topbar()
    if not check_manager_access():
        st.stop()

    hero(
        "Intelligence Center",
        "Customer-level analytics focused on risk, escalation, complaint drivers, emotion signals, and recovery outcomes.",
    )

    data = enrich_manager_data(prepare_dataframe(load_records()))
    if len(data) == 0:
        st.info("No support cases available yet.")
    else:
        sent = len(data[data["coupon_status"] == "Sent"])
        suggested_total = len(data[data["coupon_status"].isin(["Suggested", "Optional", "Approved", "Sent", "Rejected"])] )
        sent_rate = round((sent / suggested_total) * 100, 1) if suggested_total else 0
        kpi_grid([
            ("Total Customers", str(data["customer_id"].nunique())),
            ("High Risk", str(len(data[data["risk_level"] == "High"]))),
            ("Escalated", str(len(data[data["status"] == "Escalated"]))),
            ("Avg Recovery", str(round(data["recovery_score"].mean(), 1))),
            ("Coupon Sent Rate", f"{sent_rate}%"),
        ])

        if st.button("Generate Support Insight"):
            with st.spinner("Generating insight..."):
                st.warning(generate_intelligence_summary(data))

        st.markdown("### Customer Risk Watchlist")
        watchlist = data.sort_values(["priority_score", "risk_score"], ascending=[False, False]).head(10)
        st.dataframe(
            watchlist[["customer_id", "risk_score", "risk_level", "recovery_score", "complaint_topic", "emotion", "tone", "status", "next_best_action"]],
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("### Complaint Drivers")
            topic_count = data["complaint_topic"].value_counts().reset_index()
            topic_count.columns = ["Topic", "Customers"]
            st.bar_chart(topic_count.set_index("Topic"))
            avg_risk_topic = data.groupby("complaint_topic")["risk_score"].mean().round(1).reset_index()
            avg_risk_topic.columns = ["Topic", "Avg Risk"]
            st.dataframe(avg_risk_topic.sort_values("Avg Risk", ascending=False), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("### Emotion & Tone Intelligence")
            emotion_count = data["emotion"].value_counts().reset_index()
            emotion_count.columns = ["Emotion", "Customers"]
            st.bar_chart(emotion_count.set_index("Emotion"))
            tone_count = data["tone"].value_counts().reset_index()
            tone_count.columns = ["Tone", "Customers"]
            st.dataframe(tone_count, use_container_width=True, hide_index=True)

        st.markdown("### Escalation Monitor")
        escalated = data[(data["status"] == "Escalated") | (data["escalation_trigger"].astype(str).str.len() > 0)]
        if len(escalated) == 0:
            st.success("No escalated customers yet.")
        else:
            st.dataframe(
                escalated[["customer_id", "message", "risk_score", "risk_level", "assigned_to", "escalation_trigger", "next_best_action"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Recovery Performance")
        recovery_count = data["coupon_status"].value_counts().reset_index()
        recovery_count.columns = ["Coupon Status", "Customers"]
        st.bar_chart(recovery_count.set_index("Coupon Status"))

        st.markdown("---")
        st.subheader("Manager Controls")
        if st.button("Clear All Demo Data"):
            clear_records_file()
            reset_chat_for_customer(st.session_state.chat_customer_id)
            st.success("All demo data cleared.")
            st.rerun()

# ============================================================
# Page: Customer Emotion Journey
# ============================================================

elif page == "Customer Emotion Journey":
    render_topbar()
    if not check_manager_access():
        st.stop()

    hero(
        "Customer Emotion Journey",
        "Visualize how customer emotion, tone, issue type, and risk develop across the support conversation.",
    )

    data = enrich_manager_data(prepare_dataframe(load_records()))
    if len(data) == 0:
        st.info("No customer journey available yet.")
    else:
        selected_customer = st.selectbox("Select customer", sorted(data["customer_id"].unique()))
        row = data[data["customer_id"] == selected_customer].sort_values("last_updated_dt").iloc[-1]
        kpi_grid([
            ("Latest Emotion", display_value(row.get("emotion"))),
            ("Tone", display_value(row.get("tone"))),
            ("Risk", f"{display_value(row.get('risk_score'), '0')}%"),
            ("Recovery", f"{display_value(row.get('recovery_score'), '0')}%"),
            ("Status", display_value(row.get("status"))),
        ])

        st.markdown("### Emotion Journey")
        emotions = [e.strip() for e in str(row.get("emotion_journey", "")).split("→") if e.strip()]
        if emotions:
            path_html = '<div class="emotion-path">'
            for i, e in enumerate(emotions):
                path_html += f'<span class="emotion-node">{safe_text(e)}</span>'
                if i < len(emotions) - 1:
                    path_html += '<span class="arrow">→</span>'
            path_html += '</div>'
            st.markdown(path_html, unsafe_allow_html=True)
        else:
            st.info("No emotion journey yet.")

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("### Topic Path")
            st.info(display_value(row.get("topic_journey"), "No topic path yet."))
            st.markdown("### Tone Path")
            st.warning(display_value(row.get("tone_journey"), "No tone path yet."))
        with col2:
            st.markdown("### Risk Over Time")
            risk_values = parse_risk_journey(row.get("risk_journey", ""))
            if risk_values:
                chart_df = pd.DataFrame({"Step": list(range(1, len(risk_values) + 1)), "Risk Score": risk_values}).set_index("Step")
                st.line_chart(chart_df)
            else:
                st.info("No risk journey yet.")

        st.markdown("### Journey Summary")
        st.write(display_value(row.get("conversation_summary")))
        st.write(f"**Next Best Action:** {display_value(row.get('next_best_action'))}")
        st.write(f"**Escalation Trigger:** {display_value(row.get('escalation_trigger'))}")

# ============================================================
# Page: About
# ============================================================

elif page == "About System":
    render_topbar()
    hero(
        "About This System",
        "An AI-guided customer support recovery system with private manager intelligence, customer-level prioritization, and recovery offer workflow.",
    )
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">What makes it unique?</div>
            <div class="muted">
                The customer only sees a helpful support assistant. Managers privately see emotion, tone, sarcasm, risk, recovery score, next best action, escalation triggers, and recovery offer recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        Features:
        - Responsive light/dark UI for desktop, tablet, and mobile
        - Guided chatbot with issue context
        - Greeting and small-talk handling without creating fake cases
        - One active case per customer conversation
        - Manager escalation detection
        - Sarcasm and rude tone detection
        - Emotion, topic, risk, recovery score, and next best action
        - Intelligence Center with customer-level analytics
        - Visual Customer Emotion Journey
        - Manager-only coupon workflow
        - Optional transformer emotion model support
        """
    )
    st.subheader("Optional transformer setup")
    st.code(
        """# requirements.txt optional additions
transformers
torch

# Streamlit Secrets
ENABLE_TRANSFORMERS = "true"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"""
    )
    st.subheader("Privacy note")
    st.write("This is a demo app. Do not enter passwords, payment details, addresses, or private customer records.")
