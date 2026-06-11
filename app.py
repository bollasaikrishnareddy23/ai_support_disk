# ============================================================
# AI Customer Support Recovery System
# Streamlit customer chatbot + manager command center
#
# Key design:
# - Customers chat normally.
# - One active chat = one support case.
# - Greetings/small talk do not create cases.
# - Manager escalation requests are detected and prioritized.
# - Coupon offers are manager-only, never shown to customers.
# - Manager dashboard is customer-level, not message-level.
# - Optional transformer emotion model can be enabled with secrets.
# ============================================================

import os
import re
import json
import hmac
import html
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="AI Support Recovery System",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Constants
# ============================================================

APP_NAME = "AI Support Recovery"
DATA_FILE = "support_cases.csv"
GEMINI_MODEL = "gemini-2.0-flash"
HIGH_RISK_THRESHOLD = 70
MEDIUM_RISK_THRESHOLD = 40

EXPECTED_COLUMNS = [
    "case_id",
    "customer_id",
    "timestamp",
    "last_updated",
    "message",
    "conversation",
    "clean_text",
    "analysis_source",
    "emotion",
    "tone",
    "sarcasm",
    "escalation_requested",
    "complaint_topic",
    "customer_intent",
    "business_risk",
    "risk_score",
    "risk_level",
    "risk_reason",
    "recommended_action",
    "customer_reply",
    "status",
    "assigned_to",
    "resolution_action",
    "coupon_offer",
    "coupon_code",
    "coupon_status",
    "coupon_reason",
    "emotion_journey",
    "topic_journey",
    "risk_journey",
]


# ============================================================
# Utility
# ============================================================

def is_blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in ["nan", "none", "nat"]


def clean_display(value: Any, fallback: str = "—") -> str:
    if is_blank(value):
        return fallback
    return str(value)


def safe_text(value: Any) -> str:
    return html.escape(clean_display(value, ""))


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_case_id() -> str:
    return "CASE-" + datetime.now().strftime("%m%d%H%M%S%f")[-14:]


# ============================================================
# Theme and CSS
# ============================================================

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Light"


def apply_css(theme: str) -> None:
    dark = theme == "Dark"

    bg = "#07111f" if dark else "#eef5ff"
    bg2 = "#0f172a" if dark else "#e8fff6"
    text = "#e5eefc" if dark else "#0f172a"
    muted = "#9fb0c9" if dark else "#64748b"
    card = "rgba(15, 23, 42, 0.82)" if dark else "rgba(255, 255, 255, 0.88)"
    card_border = "rgba(148, 163, 184, 0.28)" if dark else "rgba(226, 232, 240, 0.95)"
    input_bg = "#111c2f" if dark else "#f8fafc"
    sidebar = "#050b16" if dark else "#06111f"
    hero1 = "#051024" if dark else "#101828"
    hero2 = "#10203a" if dark else "#1e40af"

    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {bg};
                --bg2: {bg2};
                --text: {text};
                --muted: {muted};
                --card: {card};
                --card-border: {card_border};
                --input-bg: {input_bg};
                --hero1: {hero1};
                --hero2: {hero2};
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top right, rgba(45, 212, 191, 0.25), transparent 36%),
                    radial-gradient(circle at 15% 10%, rgba(37, 99, 235, 0.22), transparent 32%),
                    linear-gradient(135deg, var(--bg) 0%, var(--bg2) 100%);
                color: var(--text);
            }}

            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1260px;
            }}

            section[data-testid="stSidebar"] {{
                background: {sidebar};
            }}
            section[data-testid="stSidebar"] * {{
                color: #f8fafc !important;
            }}

            .app-hero {{
                background:
                    linear-gradient(135deg, var(--hero1) 0%, var(--hero2) 100%);
                color: white;
                padding: 28px 32px;
                border-radius: 30px;
                margin-bottom: 22px;
                box-shadow: 0 25px 60px rgba(2, 6, 23, 0.24);
                position: relative;
                overflow: hidden;
            }}
            .app-hero:after {{
                content: "";
                position: absolute;
                right: -70px;
                top: -80px;
                width: 240px;
                height: 240px;
                border-radius: 999px;
                background: rgba(255,255,255,0.14);
            }}
            .app-hero h1 {{
                font-size: 40px;
                line-height: 1.08;
                margin: 0 0 10px 0;
                letter-spacing: -0.035em;
                font-weight: 900;
            }}
            .app-hero p {{
                margin: 0;
                max-width: 860px;
                color: #dbeafe;
                font-size: 16px;
                line-height: 1.55;
            }}

            .glass-card {{
                background: var(--card);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 22px;
                box-shadow: 0 16px 42px rgba(2, 6, 23, 0.10);
                backdrop-filter: blur(16px);
                margin-bottom: 18px;
                color: var(--text);
            }}

            .mini-card {{
                background: var(--card);
                border: 1px solid var(--card-border);
                border-radius: 20px;
                padding: 18px;
                box-shadow: 0 12px 28px rgba(2, 6, 23, 0.08);
                margin-bottom: 16px;
                color: var(--text);
            }}

            .section-title {{
                font-size: 20px;
                font-weight: 900;
                color: var(--text);
                margin-bottom: 7px;
                letter-spacing: -0.02em;
            }}
            .muted {{
                color: var(--muted);
                font-size: 14px;
                line-height: 1.55;
            }}

            .kpi-card {{
                background: var(--card);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 20px;
                min-height: 112px;
                box-shadow: 0 16px 36px rgba(2, 6, 23, 0.08);
            }}
            .kpi-value {{
                font-size: 34px;
                font-weight: 900;
                color: var(--text);
                letter-spacing: -0.04em;
            }}
            .kpi-label {{
                color: var(--muted);
                font-size: 13px;
                margin-top: 5px;
            }}

            .bot-row, .user-row {{
                display: flex;
                gap: 12px;
                align-items: flex-start;
                margin: 14px 0;
            }}
            .user-row {{
                justify-content: flex-end;
            }}
            .avatar-bot, .avatar-user {{
                width: 34px;
                height: 34px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 900;
                flex-shrink: 0;
            }}
            .avatar-bot {{ background: linear-gradient(135deg, #f59e0b, #f97316); }}
            .avatar-user {{ background: linear-gradient(135deg, #ef4444, #f43f5e); }}
            .msg-bubble {{
                max-width: 82%;
                background: var(--card);
                color: var(--text);
                border: 1px solid var(--card-border);
                padding: 13px 15px;
                border-radius: 18px;
                line-height: 1.55;
                box-shadow: 0 10px 26px rgba(2, 6, 23, 0.06);
            }}
            .user-row .msg-bubble {{
                background: rgba(37, 99, 235, 0.12);
                border-color: rgba(37, 99, 235, 0.22);
            }}

            .badge {{
                display: inline-block;
                padding: 6px 11px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 800;
                margin-right: 6px;
                margin-bottom: 6px;
            }}
            .badge-high {{ background:#fee2e2; color:#991b1b; }}
            .badge-medium {{ background:#fef3c7; color:#92400e; }}
            .badge-low {{ background:#dcfce7; color:#166534; }}
            .badge-critical {{ background:#7f1d1d; color:#fff; }}
            .badge-blue {{ background:#dbeafe; color:#1e40af; }}
            .badge-gray {{ background:#f1f5f9; color:#334155; }}
            .badge-purple {{ background:#ede9fe; color:#5b21b6; }}
            .badge-coupon {{ background:#ffedd5; color:#9a3412; }}
            .badge-dark {{ background:#e2e8f0; color:#0f172a; }}

            .path-wrap {{
                display:flex;
                flex-wrap:wrap;
                gap:10px;
                margin: 12px 0;
            }}
            .path-pill {{
                padding: 10px 14px;
                border-radius: 999px;
                font-weight: 850;
                background: var(--card);
                border: 1px solid var(--card-border);
                box-shadow: 0 8px 18px rgba(2, 6, 23, 0.06);
            }}
            .path-arrow {{
                color: var(--muted);
                font-weight: 900;
                align-self: center;
            }}

            div.stButton > button, div.stFormSubmitButton > button {{
                border-radius: 14px;
                border: none;
                background: linear-gradient(135deg, #2563eb, #4f46e5);
                color: white;
                font-weight: 850;
                padding: 0.65rem 1.1rem;
                box-shadow: 0 10px 22px rgba(37, 99, 235, 0.25);
            }}
            div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
                background: linear-gradient(135deg, #1d4ed8, #4338ca);
                color: white;
            }}
            .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {{
                border-radius: 14px !important;
                background: var(--input-bg) !important;
            }}

            #MainMenu {{ visibility: hidden; }}
            footer {{ visibility: hidden; }}

            @media (max-width: 900px) {{
                .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
                .app-hero {{ padding: 22px; border-radius: 22px; }}
                .app-hero h1 {{ font-size: 28px; }}
                .msg-bubble {{ max-width: 92%; }}
                .kpi-value {{ font-size: 26px; }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>{safe_text(title)}</h1>
            <p>{safe_text(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">{safe_text(value)}</div>
            <div class="kpi-label">{safe_text(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "gray") -> str:
    return f'<span class="badge badge-{kind}">{safe_text(text)}</span>'


def render_message(role: str, content: str) -> None:
    if role == "user":
        st.markdown(
            f"""
            <div class="user-row">
                <div class="msg-bubble">{safe_text(content)}</div>
                <div class="avatar-user">👤</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="bot-row">
                <div class="avatar-bot">🎧</div>
                <div class="msg-bubble">{safe_text(content)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Secrets and optional clients/models
# ============================================================

def get_secret_value(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    if value == default or value == "":
        value = os.getenv(name, default)
    return str(value)


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    if genai is None:
        return None
    api_key = get_secret_value("GEMINI_API_KEY", "")
    if api_key == "":
        return None
    return genai.Client(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_transformer_emotion_model():
    enabled = get_secret_value("ENABLE_TRANSFORMERS", "false").lower() == "true"
    if not enabled:
        return None
    try:
        from transformers import pipeline
        model_name = get_secret_value("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
        return pipeline("text-classification", model=model_name, top_k=None)
    except Exception:
        return None


# ============================================================
# Storage
# ============================================================

def load_records() -> List[Dict[str, Any]]:
    if not os.path.exists(DATA_FILE):
        return []
    try:
        df = pd.read_csv(DATA_FILE)
        if df.empty:
            return []
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[EXPECTED_COLUMNS]
        df = df.fillna("")
        return df.to_dict("records")
    except Exception:
        return []


def save_records(records: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[EXPECTED_COLUMNS].fillna("")
    df.to_csv(DATA_FILE, index=False)


def clear_records_file() -> None:
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)


def prepare_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[EXPECTED_COLUMNS].fillna("")
    if len(df) > 0:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0).astype(int)
    return df


# ============================================================
# Intent/tone/topic detection
# ============================================================

def is_greeting(text: str) -> bool:
    t = clean_text(text)
    greetings = {
        "hi", "hai", "hello", "hey", "hii", "hiii", "good morning",
        "good afternoon", "good evening", "yo", "hola"
    }
    return t in greetings or (len(t.split()) <= 3 and any(g in t.split() for g in ["hi", "hai", "hello", "hey"]))


def is_thanks(text: str) -> bool:
    t = clean_text(text)
    return t in ["thanks", "thank you", "ok thanks", "okay thanks", "thank u", "ty"]


def is_rude(text: str) -> bool:
    t = clean_text(text)
    rude_words = [
        "fuck", "fuck off", "shit", "asshole", "idiot", "stupid", "useless",
        "dumb", "nonsense", "bullshit", "bloody", "trash"
    ]
    return any(word in t for word in rude_words)


def is_manager_escalation_request(text: str) -> bool:
    t = clean_text(text)
    phrases = [
        "manager", "supervisor", "human agent", "real person", "talk to someone",
        "speak to someone", "talk with your manager", "talk to your manager",
        "connect me", "escalate", "higher authority", "senior support",
        "live agent", "customer care head"
    ]
    return any(phrase in t for phrase in phrases)


def is_reference_info(text: str) -> bool:
    t = str(text).strip()
    if len(t) > 30:
        return False
    patterns = [
        r"^[A-Za-z]{2,5}\d{2,}$",
        r"^\d{4,}$",
        r"^(ord|order|ref|case)[\s\-#]*[A-Za-z0-9\-]+$",
    ]
    return any(re.match(p, t, flags=re.IGNORECASE) for p in patterns)


def is_eta_request(text: str) -> bool:
    t = clean_text(text)
    phrases = [
        "possible date", "delivery date", "expected date", "when will", "when can",
        "how long", "eta", "arrival date", "arrive", "come", "deliver"
    ]
    return any(p in t for p in phrases)


def detect_sarcasm(text: str) -> Tuple[bool, str]:
    t = clean_text(text)
    positive_words = ["great", "amazing", "wonderful", "fantastic", "perfect", "excellent", "lovely", "nice"]
    negative_context = [
        "late", "delayed", "ignored", "broken", "damaged", "charged", "refund",
        "wrong", "missing", "crash", "error", "cancelled", "nothing", "still no", "again"
    ]
    if any(p in t for p in positive_words) and any(n in t for n in negative_context):
        return True, "Positive words used with negative complaint context."
    return False, "No sarcasm detected."


def detect_topic(text: str, selected_hint: str = "Not sure yet", active_topic: str = "") -> str:
    t = clean_text(text)

    # Follow-up context first
    if is_eta_request(text) and active_topic == "Delivery Issue":
        return "Delivery Issue"
    if is_reference_info(text) and active_topic:
        return active_topic

    # Delivery / tracking
    delivery_patterns = [
        "package", "parcel", "shipment", "shipping", "delivery", "delivered", "late", "delay",
        "delayed", "not received", "missing", "where is", "tracking", "track", "arrive",
        "arrival", "order status", "my product", "where my product", "where is my product"
    ]
    if any(p in t for p in delivery_patterns):
        return "Delivery Issue"

    if any(p in t for p in ["refund", "money back", "return my money", "cashback", "chargeback"]):
        return "Refund Issue"

    if any(p in t for p in ["charged", "charged twice", "payment", "billing", "bill", "invoice", "paid", "card", "transaction"]):
        return "Billing Issue"

    if any(p in t for p in ["broken", "damaged", "defective", "faulty", "quality", "wrong item", "wrong product", "replacement"]):
        return "Product Issue"

    if any(p in t for p in ["login", "password", "app", "website", "error", "crash", "bug", "not working", "technical"]):
        return "Technical Issue"

    if any(p in t for p in ["support", "agent", "customer service", "no reply", "ignored", "nobody helped", "rude"]):
        return "Customer Service Issue"

    if is_manager_escalation_request(text):
        return active_topic if active_topic else "Customer Service Issue"

    if selected_hint and selected_hint != "Not sure yet" and not is_greeting(text):
        return selected_hint

    return active_topic if active_topic else "General Complaint"


def detect_emotion_rule(text: str, topic: str, sarcastic: bool, rude: bool, escalation: bool) -> str:
    t = clean_text(text)

    if rude or escalation:
        return "Angry"
    if sarcastic:
        return "Disappointed"

    angry_words = ["angry", "furious", "mad", "worst", "terrible", "awful", "hate", "unacceptable", "ridiculous"]
    disappointed_words = ["disappointed", "upset", "frustrated", "annoyed", "unhappy", "poor", "bad experience", "still waiting"]
    confused_words = ["confused", "unclear", "do not understand", "don't understand", "why", "how", "what happened"]
    positive_words = ["thanks", "thank you", "appreciate", "great", "good", "excellent", "resolved", "solved", "happy"]

    if any(w in t for w in angry_words):
        return "Angry"
    if any(w in t for w in disappointed_words):
        return "Disappointed"
    if any(w in t for w in confused_words):
        return "Confused"
    if any(w in t for w in positive_words) and topic == "General Complaint":
        return "Satisfied"

    if topic in ["Refund Issue", "Billing Issue", "Customer Service Issue"]:
        return "Disappointed"
    if topic in ["Delivery Issue", "Product Issue", "Technical Issue"]:
        return "Confused"

    return "Neutral"


def detect_emotion_transformer(text: str) -> Tuple[str, float, str]:
    model = get_transformer_emotion_model()
    if model is None:
        return "", 0.0, "not_enabled"
    try:
        results = model(text)
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            results = results[0]
        top = max(results, key=lambda x: x.get("score", 0))
        label = str(top.get("label", "neutral")).lower()
        score = float(top.get("score", 0.0))
        emotion_map = {
            "anger": "Angry",
            "disgust": "Angry",
            "fear": "Confused",
            "joy": "Satisfied",
            "neutral": "Neutral",
            "sadness": "Disappointed",
            "surprise": "Confused",
        }
        return emotion_map.get(label, "Neutral"), score, label
    except Exception:
        return "", 0.0, "failed"


def detect_tone(emotion: str, sarcastic: bool, rude: bool, escalation: bool) -> str:
    if rude:
        return "Rude"
    if sarcastic:
        return "Sarcastic"
    if escalation:
        return "Escalation Request"
    if emotion == "Angry":
        return "Angry"
    if emotion == "Satisfied":
        return "Positive"
    if emotion == "Confused":
        return "Confused"
    return "Polite"


def get_risk_level(score: int) -> str:
    if score >= HIGH_RISK_THRESHOLD:
        return "High"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def calculate_risk(emotion: str, topic: str, text: str, sarcastic: bool, rude: bool, escalation: bool) -> Tuple[int, str]:
    t = clean_text(text)
    score_map = {
        "Satisfied": 5,
        "Neutral": 15,
        "Confused": 30,
        "Disappointed": 45,
        "Angry": 70,
    }
    score = score_map.get(emotion, 15)
    reasons = [f"Base score from emotion: {emotion}."]

    if topic in ["Refund Issue", "Billing Issue"]:
        score += 10
        reasons.append(f"{topic} is money-related.")
    if topic == "Customer Service Issue":
        score += 10
        reasons.append("Customer service issue can escalate quickly.")
    if topic == "Delivery Issue" and any(w in t for w in ["missing", "not received", "where is", "late", "delayed"]):
        score += 10
        reasons.append("Delivery issue suggests delay or missing item.")
    if any(w in t for w in ["refund", "money back", "cancel", "cancellation", "never buy"]):
        score += 15
        reasons.append("Customer mentioned refund/cancellation risk.")
    if any(w in t for w in ["again", "still", "no reply", "ignored", "nobody helped", "waiting"]):
        score += 10
        reasons.append("Message suggests repeated or unresolved issue.")
    if sarcastic:
        score += 10
        reasons.append("Sarcastic tone indicates hidden dissatisfaction.")
    if rude:
        score += 15
        reasons.append("Rude/abusive tone indicates high frustration.")
    if escalation:
        score += 15
        reasons.append("Customer requested manager/human escalation.")

    score = max(0, min(score, 100))
    return score, " ".join(reasons)


def generate_customer_reply(text: str, topic: str, active_topic: str, rude: bool, escalation: bool, reference: bool, eta: bool) -> str:
    if is_greeting(text):
        return "Hi! I’m here to help. You can tell me about a delivery, refund, billing, product, technical, or support issue."

    if is_thanks(text):
        return "You’re welcome. I’m here if you need any more help."

    if escalation:
        return (
            "I understand that you want manager support. I’ve marked your case for manager review "
            "so a manager or senior support team member can review the issue."
        )

    if rude and topic == "General Complaint" and not active_topic:
        return (
            "I understand you’re upset. I’m here to help, but please describe the issue so I can record it properly for our team."
        )

    if reference:
        return "Thank you. I’ve added that reference to your case so the team can use it while reviewing your issue."

    if eta and (active_topic == "Delivery Issue" or topic == "Delivery Issue"):
        return (
            "I understand you want the expected delivery date. I’ve updated your delivery case so the team can check the shipment status and next possible update."
        )

    if topic == "Delivery Issue":
        return (
            "I’m sorry about the delivery problem. I’ve recorded this as a delivery issue so the team can check the shipment status, delay reason, and next update."
        )
    if topic == "Refund Issue":
        return (
            "I understand your refund concern. I’ve recorded your request so the team can review the order and follow up with the next steps."
        )
    if topic == "Billing Issue":
        return (
            "Thank you for reporting the billing concern. I’ve recorded it so the team can review the payment or invoice details carefully."
        )
    if topic == "Product Issue":
        return (
            "I’m sorry about the product issue. I’ve recorded it so the team can review the item condition and possible replacement or support options."
        )
    if topic == "Technical Issue":
        return (
            "Thank you for reporting the technical issue. I’ve recorded the problem so the team can review it and work on the next steps."
        )
    if topic == "Customer Service Issue":
        return (
            "I’m sorry the support experience has not been helpful. I’ve recorded this so the team can review the service issue and follow up properly."
        )

    return "Thanks. Could you describe what happened in a little more detail so I can record the issue properly?"


def is_actionable_message(text: str, topic: str, active_case_id: str, rude: bool, escalation: bool, reference: bool, eta: bool) -> bool:
    if is_greeting(text) or is_thanks(text):
        return False
    if escalation:
        return True
    if topic != "General Complaint":
        return True
    if active_case_id and (rude or reference or eta):
        return True
    if rude and not active_case_id:
        return False
    return False


# ============================================================
# Optional Gemini manager analysis
# ============================================================

def extract_json(text: str) -> Dict[str, Any]:
    text = str(text).strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def gemini_private_analysis(message: str, conversation: str) -> Dict[str, str]:
    client = get_gemini_client()
    if client is None:
        return {}

    prompt = f"""
You are a private customer support intelligence analyst.
Analyze the latest customer message and recent conversation.
Return only valid JSON, no markdown.

Required JSON:
{{
  "customer_intent": "short intent",
  "business_risk": "short business risk",
  "recommended_action": "short manager action"
}}

Conversation:
{conversation}

Latest message:
{message}
"""
    try:
        if types is not None:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"),
            )
        else:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        if not response.text:
            return {}
        return extract_json(response.text)
    except Exception:
        return {}


# ============================================================
# Case lifecycle
# ============================================================

def format_conversation(messages: List[Dict[str, str]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            lines.append(f"Customer: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n\n".join(lines)


def get_active_topic() -> str:
    if st.session_state.active_case_id == "":
        return ""
    records = load_records()
    for r in records:
        if str(r.get("case_id", "")) == st.session_state.active_case_id:
            return str(r.get("complaint_topic", ""))
    return ""


def get_active_case_id() -> str:
    if st.session_state.active_case_id == "":
        st.session_state.active_case_id = create_case_id()
    return st.session_state.active_case_id


def suggest_recovery_coupon(risk_level: str, risk_score: int, topic: str, case_id: str) -> Dict[str, str]:
    risk_score = int(risk_score)
    short_id = str(case_id).replace("CASE-", "")[-6:]

    if risk_level == "High":
        discount = 20 if risk_score >= 85 else 15
        if topic in ["Refund Issue", "Billing Issue"]:
            offer = f"{discount}% goodwill coupon after billing/refund review"
            reason = "High-risk money-related complaint. Coupon should be manager-approved only after review."
        elif topic == "Delivery Issue":
            offer = f"{discount}% apology coupon or free-shipping recovery offer"
            reason = "High-risk delivery issue. Coupon may help recover customer trust."
        elif topic == "Product Issue":
            offer = f"{discount}% product recovery coupon"
            reason = "High-risk product issue. Coupon may help reduce churn after support action."
        else:
            offer = f"{discount}% customer recovery coupon"
            reason = "High-risk dissatisfaction. Coupon may help reduce escalation risk."
        return {
            "coupon_offer": offer,
            "coupon_code": f"REC-{short_id}-{discount}",
            "coupon_status": "Suggested",
            "coupon_reason": reason,
        }

    if risk_level == "Medium":
        return {
            "coupon_offer": "10% goodwill coupon if unresolved",
            "coupon_code": f"REC-{short_id}-10",
            "coupon_status": "Optional",
            "coupon_reason": "Medium-risk case. Coupon is optional if follow-up is delayed.",
        }

    return {
        "coupon_offer": "No coupon needed",
        "coupon_code": "",
        "coupon_status": "Not Required",
        "coupon_reason": "Low-risk case. Normal support process is enough.",
    }


def analyze_message(message: str, selected_hint: str, active_topic: str) -> Dict[str, Any]:
    cleaned = clean_text(message)
    rude = is_rude(message)
    escalation = is_manager_escalation_request(message)
    reference = is_reference_info(message)
    eta = is_eta_request(message)
    sarcastic, sarcasm_reason = detect_sarcasm(message)

    topic = detect_topic(message, selected_hint, active_topic)

    transformer_emotion, transformer_score, transformer_label = detect_emotion_transformer(message)
    rule_emotion = detect_emotion_rule(message, topic, sarcastic, rude, escalation)

    # Use transformer only when it is confident and not contradicted by strong rules.
    if transformer_emotion and transformer_score >= 0.70 and not (rude or escalation or sarcastic):
        emotion = transformer_emotion
        source = f"Transformer ({transformer_label})"
    else:
        emotion = rule_emotion
        source = "Rule Engine"

    tone = detect_tone(emotion, sarcastic, rude, escalation)
    risk_score, risk_reason = calculate_risk(emotion, topic, message, sarcastic, rude, escalation)
    risk_level = get_risk_level(risk_score)

    customer_reply = generate_customer_reply(message, topic, active_topic, rude, escalation, reference, eta)

    conversation = format_conversation(st.session_state.chat_messages)
    ai_private = gemini_private_analysis(message, conversation)

    if escalation:
        customer_intent = "Customer wants manager or human support."
        business_risk = "Escalation request indicates high dissatisfaction and potential churn risk."
        recommended_action = "Manager should review this case and follow up as soon as possible."
    elif topic == "Delivery Issue":
        customer_intent = "Customer wants delivery status, tracking, or resolution."
        business_risk = "Customer satisfaction may decrease if delivery status is not clarified."
        recommended_action = "Route to delivery team and provide clear shipment update."
    elif topic == "Refund Issue":
        customer_intent = "Customer wants refund or money-related resolution."
        business_risk = "Refund concern may lead to escalation or cancellation if delayed."
        recommended_action = "Review refund eligibility and update customer clearly."
    elif topic == "Billing Issue":
        customer_intent = "Customer wants billing/payment issue reviewed."
        business_risk = "Billing issues can reduce trust and create urgent dissatisfaction."
        recommended_action = "Route to billing team and verify payment/invoice details."
    elif topic == "Product Issue":
        customer_intent = "Customer wants product issue reviewed or resolved."
        business_risk = "Product issues may cause refund, replacement, or negative feedback."
        recommended_action = "Route to product/support team and review replacement options."
    elif topic == "Technical Issue":
        customer_intent = "Customer wants technical support."
        business_risk = "Technical failure may prevent usage and reduce satisfaction."
        recommended_action = "Route to technical team and request troubleshooting details if needed."
    else:
        customer_intent = "Customer wants the issue reviewed and resolved."
        business_risk = "Customer satisfaction may decrease if the issue is not handled promptly."
        recommended_action = "Review the case, assign the correct team, and follow up if risk is medium or high."

    customer_intent = clean_display(ai_private.get("customer_intent", customer_intent), customer_intent)
    business_risk = clean_display(ai_private.get("business_risk", business_risk), business_risk)
    recommended_action = clean_display(ai_private.get("recommended_action", recommended_action), recommended_action)

    return {
        "clean_text": cleaned,
        "analysis_source": source,
        "emotion": emotion,
        "tone": tone,
        "sarcasm": "Yes" if sarcastic else "No",
        "sarcasm_reason": sarcasm_reason,
        "escalation_requested": "Yes" if escalation else "No",
        "complaint_topic": topic,
        "customer_intent": customer_intent,
        "business_risk": business_risk,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "recommended_action": recommended_action,
        "customer_reply": customer_reply,
        "actionable": is_actionable_message(message, topic, st.session_state.active_case_id, rude, escalation, reference, eta),
        "manager_status": "Escalated" if escalation else "New",
        "manager_owner": "Manager" if escalation else "Unassigned",
    }


def append_journey(existing: str, value: Any) -> str:
    value = clean_display(value, "")
    if value == "":
        return existing
    parts = [p.strip() for p in str(existing).split("→") if p.strip()]
    parts.append(value)
    return " → ".join(parts[-12:])


def create_or_update_active_case(customer_id: str, message: str, analysis: Dict[str, Any], coupon: Dict[str, str]) -> str:
    case_id = get_active_case_id()
    records = load_records()
    current_time = now_str()
    conversation = format_conversation(st.session_state.chat_messages)

    found = False
    for record in records:
        if str(record.get("case_id", "")) == str(case_id):
            record["customer_id"] = customer_id
            record["last_updated"] = current_time
            record["message"] = message
            record["conversation"] = conversation
            record["clean_text"] = analysis["clean_text"]
            record["analysis_source"] = analysis["analysis_source"]
            record["emotion"] = analysis["emotion"]
            record["tone"] = analysis["tone"]
            record["sarcasm"] = analysis["sarcasm"]
            record["escalation_requested"] = analysis["escalation_requested"]
            record["complaint_topic"] = analysis["complaint_topic"]
            record["customer_intent"] = analysis["customer_intent"]
            record["business_risk"] = analysis["business_risk"]
            record["risk_score"] = max(int(record.get("risk_score", 0) or 0), int(analysis["risk_score"]))
            record["risk_level"] = get_risk_level(int(record["risk_score"]))
            record["risk_reason"] = analysis["risk_reason"]
            record["recommended_action"] = analysis["recommended_action"]
            record["customer_reply"] = analysis["customer_reply"]
            if analysis["manager_status"] == "Escalated":
                record["status"] = "Escalated"
                record["assigned_to"] = "Manager"
            record["emotion_journey"] = append_journey(record.get("emotion_journey", ""), analysis["emotion"])
            record["topic_journey"] = append_journey(record.get("topic_journey", ""), analysis["complaint_topic"])
            record["risk_journey"] = append_journey(record.get("risk_journey", ""), analysis["risk_score"])

            current_coupon_status = str(record.get("coupon_status", ""))
            if current_coupon_status in ["", "Not Required", "Optional", "Suggested"]:
                record.update(coupon)
            found = True
            break

    if not found:
        records.append({
            "case_id": case_id,
            "customer_id": customer_id,
            "timestamp": current_time,
            "last_updated": current_time,
            "message": message,
            "conversation": conversation,
            "clean_text": analysis["clean_text"],
            "analysis_source": analysis["analysis_source"],
            "emotion": analysis["emotion"],
            "tone": analysis["tone"],
            "sarcasm": analysis["sarcasm"],
            "escalation_requested": analysis["escalation_requested"],
            "complaint_topic": analysis["complaint_topic"],
            "customer_intent": analysis["customer_intent"],
            "business_risk": analysis["business_risk"],
            "risk_score": analysis["risk_score"],
            "risk_level": analysis["risk_level"],
            "risk_reason": analysis["risk_reason"],
            "recommended_action": analysis["recommended_action"],
            "customer_reply": analysis["customer_reply"],
            "status": analysis["manager_status"],
            "assigned_to": analysis["manager_owner"],
            "resolution_action": "",
            "coupon_offer": coupon["coupon_offer"],
            "coupon_code": coupon["coupon_code"],
            "coupon_status": coupon["coupon_status"],
            "coupon_reason": coupon["coupon_reason"],
            "emotion_journey": analysis["emotion"],
            "topic_journey": analysis["complaint_topic"],
            "risk_journey": str(analysis["risk_score"]),
        })

    save_records(records)
    st.session_state.records = records
    return case_id


def update_case_record(case_id: str, updates: Dict[str, Any]) -> None:
    records = load_records()
    for record in records:
        if str(record.get("case_id", "")) == str(case_id):
            for k, v in updates.items():
                record[k] = v
            record["last_updated"] = now_str()
            break
    save_records(records)
    st.session_state.records = records


# ============================================================
# Manager data
# ============================================================

def enrich_manager_data(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) == 0:
        return data

    data = data.copy().fillna("")
    data["risk_score"] = pd.to_numeric(data["risk_score"], errors="coerce").fillna(0).astype(int)
    data["timestamp_dt"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["last_updated_dt"] = pd.to_datetime(data["last_updated"], errors="coerce")

    now = pd.Timestamp.now()
    data["case_age_hours"] = ((now - data["timestamp_dt"]).dt.total_seconds() / 3600).fillna(0).clip(lower=0).round(1)
    data["priority_score"] = data["risk_score"].copy()
    data.loc[data["status"] == "Escalated", "priority_score"] += 10
    data.loc[data["escalation_requested"] == "Yes", "priority_score"] += 10
    data["priority_score"] = data["priority_score"].clip(upper=100)

    def priority(row):
        if row["priority_score"] >= 85:
            return "Critical"
        if row["priority_score"] >= 70:
            return "High"
        if row["priority_score"] >= 40:
            return "Medium"
        return "Low"

    def sla(row):
        if row["status"] == "Resolved":
            return "Closed"
        if row["risk_level"] == "High" and row["case_age_hours"] > 24:
            return "Overdue"
        if row["risk_level"] == "Medium" and row["case_age_hours"] > 48:
            return "Overdue"
        if row["risk_level"] == "Low" and row["case_age_hours"] > 72:
            return "Overdue"
        return "On Track"

    data["priority"] = data.apply(priority, axis=1)
    data["sla_status"] = data.apply(sla, axis=1)

    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].apply(lambda x: "" if is_blank(x) else str(x))
    return data


def customer_level_view(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) == 0:
        return data
    data = data.copy()
    data["sort_updated"] = pd.to_datetime(data["last_updated"], errors="coerce")
    data = data.sort_values(["risk_score", "sort_updated"], ascending=[False, False])
    view = data.groupby("customer_id", as_index=False).first()
    view = view.sort_values(["priority_score", "sort_updated"], ascending=[False, False])
    return view


# ============================================================
# Session state
# ============================================================

def intro_message() -> str:
    return "Hi, I’m your support assistant. What type of issue are you facing today?"


def reset_chat(new_customer: bool = False) -> None:
    st.session_state.chat_messages = [{"role": "assistant", "content": intro_message()}]
    st.session_state.active_case_id = ""
    st.session_state.last_case_id = ""
    if new_customer:
        st.session_state.selected_issue_type = "Not sure yet"


if "records" not in st.session_state:
    st.session_state.records = load_records()
if "manager_authenticated" not in st.session_state:
    st.session_state.manager_authenticated = False
if "chat_customer_id" not in st.session_state:
    st.session_state.chat_customer_id = "C001"
if "customer_input_value" not in st.session_state:
    st.session_state.customer_input_value = st.session_state.chat_customer_id
if "active_case_id" not in st.session_state:
    st.session_state.active_case_id = ""
if "last_case_id" not in st.session_state:
    st.session_state.last_case_id = ""
if "selected_issue_type" not in st.session_state:
    st.session_state.selected_issue_type = "Not sure yet"
if "chat_messages" not in st.session_state:
    reset_chat()


# ============================================================
# Sidebar and theme selector
# ============================================================

apply_css(st.session_state.ui_theme)

st.sidebar.title(APP_NAME)
st.sidebar.caption("Customer Recovery Workspace")
page = st.sidebar.radio(
    "Workspace",
    ["AI Support Chat", "Manager Command Center", "Journey Monitor", "About System"],
)
st.sidebar.markdown("---")
if st.session_state.get("manager_authenticated", False):
    if st.sidebar.button("Logout Manager"):
        st.session_state.manager_authenticated = False
        st.rerun()

# Theme corner
_, theme_col = st.columns([0.78, 0.22])
with theme_col:
    st.selectbox("Theme", ["Light", "Dark"], key="ui_theme", label_visibility="collapsed")


# ============================================================
# Manager auth
# ============================================================

def get_manager_password() -> str:
    return get_secret_value("MANAGER_PASSWORD", "")


def check_manager_access() -> bool:
    if st.session_state.get("manager_authenticated", False):
        return True

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Manager Access</div>
            <div class="muted">This workspace is protected for internal support teams.</div>
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
# Page: AI Support Chat
# ============================================================

if page == "AI Support Chat":
    hero(
        "AI Customer Support Recovery System",
        "Customers chat with a support assistant. Important issues become manager-reviewed recovery cases.",
    )

    top_left, top_right = st.columns([0.62, 0.38], gap="large")

    with top_left:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Customer Context</div>
                <div class="muted">Changing the customer/order ID automatically starts a fresh conversation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        new_customer_id = st.text_input(
            "Customer ID or Order ID",
            value=st.session_state.customer_input_value,
            placeholder="Example: C001 or ORD123",
        )
        if new_customer_id != st.session_state.chat_customer_id:
            st.session_state.chat_customer_id = new_customer_id
            st.session_state.customer_input_value = new_customer_id
            reset_chat(new_customer=True)
            st.rerun()

        st.session_state.selected_issue_type = st.selectbox(
            "Issue Type",
            [
                "Not sure yet",
                "Delivery Issue",
                "Refund Issue",
                "Billing Issue",
                "Product Issue",
                "Technical Issue",
                "Customer Service Issue",
                "General Complaint",
            ],
            index=[
                "Not sure yet",
                "Delivery Issue",
                "Refund Issue",
                "Billing Issue",
                "Product Issue",
                "Technical Issue",
                "Customer Service Issue",
                "General Complaint",
            ].index(st.session_state.selected_issue_type),
        )
        if st.session_state.active_case_id:
            st.info(f"Active case: {st.session_state.active_case_id}")
        if st.button("Start New Chat"):
            reset_chat(new_customer=False)
            st.rerun()

    with top_right:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Smart Support Flow</div>
                <p class="muted">1. Greet the assistant.</p>
                <p class="muted">2. Describe the issue naturally.</p>
                <p class="muted">3. Ask for manager support if needed.</p>
                <p class="muted">4. Managers privately see emotion, tone, risk, SLA, and recovery suggestions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Support Conversation</div>
            <div class="muted">Greetings will not create cases. Actionable messages update the same support case until a new chat or new customer ID is started.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.chat_messages:
        render_message(message["role"], message["content"])

    with st.form("chat_form", clear_on_submit=True):
        user_message = st.text_input("Message", placeholder="Type your message here...")
        sent = st.form_submit_button("Send Message")

    if sent and user_message.strip():
        st.session_state.chat_messages.append({"role": "user", "content": user_message})

        active_topic = get_active_topic()
        analysis = analyze_message(user_message, st.session_state.selected_issue_type, active_topic)

        if analysis["actionable"]:
            case_id = get_active_case_id()
            coupon = suggest_recovery_coupon(
                analysis["risk_level"],
                analysis["risk_score"],
                analysis["complaint_topic"],
                case_id,
            )
            st.session_state.chat_messages.append({"role": "assistant", "content": analysis["customer_reply"]})
            saved_case_id = create_or_update_active_case(
                st.session_state.chat_customer_id,
                user_message,
                analysis,
                coupon,
            )
            st.session_state.last_case_id = saved_case_id
        else:
            st.session_state.chat_messages.append({"role": "assistant", "content": analysis["customer_reply"]})

        st.rerun()


# ============================================================
# Page: Manager Command Center
# ============================================================

elif page == "Manager Command Center":
    if not check_manager_access():
        st.stop()

    hero(
        "Manager Command Center",
        "Customer-level risk monitoring, escalation detection, and manager-only recovery offers.",
    )

    data = enrich_manager_data(prepare_dataframe(load_records()))
    customer_view = customer_level_view(data)

    if len(data) == 0:
        st.info("No support cases yet.")
    else:
        open_cases = len(data[data["status"] != "Resolved"])
        high_customers = len(customer_view[customer_view["risk_level"] == "High"])
        escalations = len(data[data["status"] == "Escalated"])
        offers = len(data[data["coupon_status"].isin(["Suggested", "Optional", "Approved"])])
        avg_risk = round(float(customer_view["risk_score"].mean()), 1) if len(customer_view) else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            kpi_card("Open Cases", open_cases)
        with k2:
            kpi_card("High-Risk Customers", high_customers)
        with k3:
            kpi_card("Escalations", escalations)
        with k4:
            kpi_card("Recovery Offers", offers)
        with k5:
            kpi_card("Avg Customer Risk", avg_risk)

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["Priority Queue", "Case Review", "Coupon Center", "Analytics"])

        with tab1:
            st.subheader("Customer Priority Queue")
            queue = customer_view.copy()
            display_cols = [
                "customer_id", "message", "complaint_topic", "emotion", "tone", "risk_score",
                "risk_level", "priority", "sla_status", "status", "assigned_to", "coupon_status", "last_updated"
            ]
            st.dataframe(queue[display_cols].rename(columns={"message": "latest_issue"}), use_container_width=True, hide_index=True)
            csv = queue.to_csv(index=False).encode("utf-8")
            st.download_button("Download Customer Queue", data=csv, file_name="customer_priority_queue.csv", mime="text/csv")

        with tab2:
            st.subheader("Customer Review Workspace")
            options = customer_view.index.tolist()
            selected_idx = st.selectbox(
                "Select customer",
                options,
                format_func=lambda idx: (
                    f"{clean_display(customer_view.loc[idx, 'customer_id'])} | "
                    f"{clean_display(customer_view.loc[idx, 'risk_level'])} Risk | "
                    f"{clean_display(customer_view.loc[idx, 'risk_score'])}% | "
                    f"{clean_display(customer_view.loc[idx, 'complaint_topic'])}"
                ),
            )
            record = customer_view.loc[selected_idx].to_dict()

            left, right = st.columns(2, gap="large")
            with left:
                st.markdown(
                    """
                    <div class="glass-card"><div class="section-title">Case Summary</div></div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write(f"**Customer ID:** {clean_display(record.get('customer_id'))}")
                st.write(f"**Latest Issue:** {clean_display(record.get('message'))}")
                st.write(f"**Topic:** {clean_display(record.get('complaint_topic'))}")
                st.write(f"**Risk Score:** {clean_display(record.get('risk_score'))}%")
                st.write(f"**Risk Level:** {clean_display(record.get('risk_level'))}")
                st.write(f"**Priority:** {clean_display(record.get('priority'))}")
                st.write(f"**SLA:** {clean_display(record.get('sla_status'))}")
                st.write(f"**Status:** {clean_display(record.get('status'))}")
                st.write(f"**Assigned To:** {clean_display(record.get('assigned_to'))}")

            with right:
                st.markdown(
                    """
                    <div class="glass-card"><div class="section-title">Internal Intelligence</div></div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write(f"**Emotion:** {clean_display(record.get('emotion'))}")
                st.write(f"**Tone:** {clean_display(record.get('tone'))}")
                st.write(f"**Sarcasm:** {clean_display(record.get('sarcasm'))}")
                st.write(f"**Escalation Requested:** {clean_display(record.get('escalation_requested'))}")
                st.write(f"**Intent:** {clean_display(record.get('customer_intent'))}")
                st.write(f"**Business Risk:** {clean_display(record.get('business_risk'))}")
                st.write(f"**Risk Reason:** {clean_display(record.get('risk_reason'))}")
                st.warning(clean_display(record.get("recommended_action")))

            st.markdown("---")
            st.subheader("Manager-Only Recovery Offer")
            coupon_status = clean_display(record.get("coupon_status"), "Not Required")
            coupon_code = clean_display(record.get("coupon_code"), "")
            coupon_offer = clean_display(record.get("coupon_offer"), "No coupon needed")
            coupon_reason = clean_display(record.get("coupon_reason"), "")
            case_id = clean_display(record.get("case_id"), "")

            st.write(f"**Offer:** {coupon_offer}")
            st.write(f"**Status:** {coupon_status}")
            if coupon_code:
                st.code(coupon_code)
            if coupon_reason:
                st.caption(coupon_reason)

            if coupon_status in ["Suggested", "Optional"]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("Approve Coupon", key=f"approve_{case_id}"):
                        update_case_record(case_id, {"coupon_status": "Approved", "resolution_action": f"Coupon approved: {coupon_code}"})
                        st.rerun()
                with c2:
                    if st.button("Reject Coupon", key=f"reject_{case_id}"):
                        update_case_record(case_id, {"coupon_status": "Rejected", "resolution_action": "Coupon rejected by manager."})
                        st.rerun()
                with c3:
                    if st.button("Mark Sent", key=f"sent_{case_id}"):
                        update_case_record(case_id, {"coupon_status": "Sent", "status": "Resolved", "resolution_action": f"Coupon sent: {coupon_code}"})
                        st.rerun()
            elif coupon_status == "Approved":
                if st.button("Mark Coupon Sent", key=f"sent_approved_{case_id}"):
                    update_case_record(case_id, {"coupon_status": "Sent", "status": "Resolved", "resolution_action": f"Coupon sent: {coupon_code}"})
                    st.rerun()

        with tab3:
            st.subheader("Coupon Center")
            coupon_data = data[data["coupon_status"].isin(["Suggested", "Optional", "Approved", "Sent", "Rejected"])]
            if len(coupon_data) == 0:
                st.info("No coupon-related cases yet.")
            else:
                st.dataframe(
                    coupon_data[["customer_id", "complaint_topic", "risk_score", "risk_level", "coupon_offer", "coupon_code", "coupon_status", "status"]],
                    use_container_width=True,
                    hide_index=True,
                )

        with tab4:
            st.subheader("Analytics")
            a1, a2 = st.columns(2)
            with a1:
                risk_table = customer_view["risk_level"].value_counts().reset_index()
                risk_table.columns = ["Risk Level", "Customers"]
                st.bar_chart(risk_table.set_index("Risk Level"))
            with a2:
                topic_table = data["complaint_topic"].value_counts().reset_index()
                topic_table.columns = ["Topic", "Cases"]
                st.bar_chart(topic_table.set_index("Topic"))

            b1, b2 = st.columns(2)
            with b1:
                tone_table = data["tone"].value_counts().reset_index()
                tone_table.columns = ["Tone", "Cases"]
                st.bar_chart(tone_table.set_index("Tone"))
            with b2:
                status_table = data["status"].value_counts().reset_index()
                status_table.columns = ["Status", "Cases"]
                st.bar_chart(status_table.set_index("Status"))

            st.markdown("---")
            if st.button("Clear All Demo Data"):
                clear_records_file()
                st.session_state.records = []
                reset_chat(new_customer=False)
                st.success("All demo data cleared.")
                st.rerun()


# ============================================================
# Page: Journey Monitor
# ============================================================

elif page == "Journey Monitor":
    if not check_manager_access():
        st.stop()

    hero(
        "Customer Emotion Journey",
        "Visualize how each customer's emotion, topic, and risk changes across the support journey.",
    )

    data = enrich_manager_data(prepare_dataframe(load_records()))
    if len(data) == 0:
        st.info("No customer journey available yet.")
    else:
        selected_customer = st.selectbox("Select customer", sorted(data["customer_id"].dropna().unique().tolist()))
        customer_data = data[data["customer_id"] == selected_customer].copy().sort_values("last_updated_dt")
        latest = customer_data.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Latest Emotion", clean_display(latest.get("emotion")))
        with c2:
            kpi_card("Latest Tone", clean_display(latest.get("tone")))
        with c3:
            kpi_card("Latest Risk", f"{clean_display(latest.get('risk_score'))}%")
        with c4:
            kpi_card("Recovery Status", clean_display(latest.get("coupon_status")))

        st.markdown("---")
        st.subheader("Customer Emotion Journey")
        emotion_path = str(latest.get("emotion_journey", "")) or " → ".join(customer_data["emotion"].astype(str).tolist())
        emotions = [p.strip() for p in emotion_path.split("→") if p.strip()]
        if emotions:
            parts = []
            for i, e in enumerate(emotions):
                parts.append(f'<span class="path-pill">{safe_text(e)}</span>')
                if i < len(emotions) - 1:
                    parts.append('<span class="path-arrow">→</span>')
            st.markdown(f'<div class="path-wrap">{"".join(parts)}</div>', unsafe_allow_html=True)

        st.subheader("Topic Journey")
        topic_path = str(latest.get("topic_journey", "")) or " → ".join(customer_data["complaint_topic"].astype(str).tolist())
        topics = [p.strip() for p in topic_path.split("→") if p.strip()]
        if topics:
            parts = []
            for i, t in enumerate(topics):
                parts.append(f'<span class="path-pill">{safe_text(t)}</span>')
                if i < len(topics) - 1:
                    parts.append('<span class="path-arrow">→</span>')
            st.markdown(f'<div class="path-wrap">{"".join(parts)}</div>', unsafe_allow_html=True)

        st.subheader("Customer Timeline")
        cols = ["last_updated", "message", "emotion", "tone", "complaint_topic", "risk_score", "risk_level", "status", "coupon_status"]
        st.dataframe(customer_data[cols], use_container_width=True, hide_index=True)

        st.subheader("Risk Over Time")
        risk_points = [int(x.strip()) for x in str(latest.get("risk_journey", "")).split("→") if x.strip().isdigit()]
        if risk_points:
            chart_df = pd.DataFrame({"step": list(range(1, len(risk_points) + 1)), "risk_score": risk_points}).set_index("step")
            st.line_chart(chart_df)
        else:
            chart_data = customer_data[["last_updated_dt", "risk_score"]].dropna().set_index("last_updated_dt")
            if len(chart_data):
                st.line_chart(chart_data)


# ============================================================
# Page: About
# ============================================================

elif page == "About System":
    hero(
        "About AI Support Recovery System",
        "A customer support chatbot with private manager intelligence, escalation detection, sarcasm/rude tone detection, and recovery offer workflow.",
    )
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">What makes this system unique?</div>
            <div class="muted">
                It separates the customer experience from manager intelligence. Customers only see helpful support replies.
                Managers privately see emotion, tone, risk, escalation requests, SLA status, and coupon recovery suggestions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        Features:
        - Responsive customer chatbot
        - Automatic new conversation when customer ID changes
        - Greeting/small-talk handling
        - Manager escalation detection
        - Sarcasm and rude tone detection
        - Optional transformer emotion model
        - Customer-level manager queue
        - Manager-only coupon recovery workflow
        - Customer emotion journey visualization
        - Light/dark theme selector
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
