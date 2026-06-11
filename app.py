# ============================================================
# VoiceOps AI Support Desk
# GitHub-ready Streamlit app
#
# Customer side:
# - Guided real-time support chatbot
# - One chat session = one support case
# - Dynamic customer replies based on message/topic/context
#
# Manager side:
# - Private command center
# - Risk score, emotion, topic, business risk
# - SLA tracking, owner assignment, notes, resolution
# - Recovery coupon recommendation and approval workflow
#
# Performance design:
# - Fast rule-based analysis is always available
# - Gemini API is optional
# - Local transformer models are optional and disabled by default
# ============================================================

import os
import re
import json
import hmac
import html
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from google import genai
except Exception:  # App still runs without Gemini package, but requirements include it.
    genai = None


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="VoiceOps AI Support Desk",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Constants
# ============================================================

DATA_FILE = "complaint_records.csv"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
HIGH_RISK_THRESHOLD = 70
MEDIUM_RISK_THRESHOLD = 40

ISSUE_TYPES = [
    "Not sure yet",
    "Delivery Issue",
    "Refund Issue",
    "Billing Issue",
    "Product Issue",
    "Technical Issue",
    "Customer Service Issue",
    "General Complaint",
]

STATUS_OPTIONS = ["New", "In Progress", "Escalated", "Resolved"]
OWNER_OPTIONS = [
    "Unassigned",
    "Support Team",
    "Billing Team",
    "Delivery Team",
    "Technical Team",
    "Product Team",
    "Manager",
]

EXPECTED_COLUMNS = [
    "case_id",
    "customer_id",
    "timestamp",
    "message",
    "conversation",
    "clean_text",
    "analysis_source",
    "emotion",
    "sarcasm_detected",
    "sarcasm_reason",
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
    "internal_notes",
    "resolution_action",
    "coupon_offer",
    "coupon_code",
    "coupon_status",
    "coupon_reason",
    "last_updated",
]


# ============================================================
# Custom Styling
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --bg: #f6f8fb;
            --ink: #0f172a;
            --muted: #64748b;
            --line: #e2e8f0;
            --blue: #2563eb;
            --green: #16a34a;
            --amber: #d97706;
            --red: #dc2626;
            --purple: #7c3aed;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 32%),
                radial-gradient(circle at top right, rgba(124, 58, 237, 0.10), transparent 30%),
                linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1300px;
        }

        section[data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.96)),
                radial-gradient(circle at 85% 15%, rgba(59,130,246,0.45), transparent 32%);
            color: white;
            padding: 32px 34px;
            border-radius: 30px;
            margin-bottom: 22px;
            box-shadow: 0 24px 55px rgba(15, 23, 42, 0.24);
            border: 1px solid rgba(255,255,255,0.10);
        }

        .hero h1 {
            font-size: 42px;
            line-height: 1.05;
            margin: 0 0 10px 0;
            font-weight: 900;
            letter-spacing: -0.04em;
        }

        .hero p {
            color: #cbd5e1;
            font-size: 17px;
            max-width: 900px;
            margin: 0;
            line-height: 1.55;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.90);
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-radius: 24px;
            padding: 22px;
            box-shadow: 0 16px 38px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(10px);
            margin-bottom: 18px;
        }

        .case-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 24px;
            padding: 22px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 16px;
        }

        .case-title {
            font-size: 20px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 5px;
        }

        .case-meta {
            color: #64748b;
            font-size: 13px;
            margin-bottom: 12px;
        }

        .case-message {
            color: #334155;
            font-size: 15px;
            line-height: 1.58;
            margin-top: 14px;
            padding: 14px;
            border-radius: 16px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }

        .kpi-card {
            background: rgba(255,255,255,0.95);
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 20px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            min-height: 112px;
        }

        .kpi-value {
            font-size: 34px;
            font-weight: 900;
            color: #0f172a;
            letter-spacing: -0.04em;
        }

        .kpi-label {
            color: #64748b;
            font-size: 14px;
            margin-top: 4px;
        }

        .section-title {
            font-size: 22px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }

        .muted {
            color: #64748b;
            font-size: 14px;
            line-height: 1.6;
        }

        .badge {
            display: inline-block;
            padding: 6px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid rgba(0,0,0,0.04);
        }

        .badge-high { background: #fee2e2; color: #991b1b; }
        .badge-medium { background: #fef3c7; color: #92400e; }
        .badge-low { background: #dcfce7; color: #166534; }
        .badge-critical { background: #7f1d1d; color: #ffffff; }
        .badge-blue { background: #dbeafe; color: #1e40af; }
        .badge-gray { background: #f1f5f9; color: #334155; }
        .badge-purple { background: #ede9fe; color: #5b21b6; }
        .badge-overdue { background: #fee2e2; color: #991b1b; }
        .badge-track { background: #dcfce7; color: #166534; }
        .badge-closed { background: #e0e7ff; color: #3730a3; }
        .badge-coupon { background: #fff7ed; color: #9a3412; }
        .badge-sarcasm { background: #fae8ff; color: #86198f; }

        div.stButton > button,
        div.stFormSubmitButton > button {
            border-radius: 14px;
            border: none;
            background: #2563eb;
            color: white;
            font-weight: 800;
            padding: 0.65rem 1.1rem;
            box-shadow: 0 10px 18px rgba(37, 99, 235, 0.20);
        }

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {
            background: #1d4ed8;
            color: white;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 14px !important;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 18px;
            padding: 0.25rem;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UI Helpers
# ============================================================

def safe_text(value) -> str:
    return html.escape(str(value))


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{safe_text(title)}</h1>
            <p>{safe_text(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">{safe_text(value)}</div>
            <div class="kpi-label">{safe_text(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{safe_text(text)}</span>'


def risk_badge(level: str) -> str:
    kind = "high" if level == "High" else "medium" if level == "Medium" else "low"
    return badge(f"{level} Risk", kind)


def priority_badge(priority: str) -> str:
    kind = "critical" if priority == "Critical" else "high" if priority == "High" else "medium" if priority == "Medium" else "low"
    return badge(f"{priority} Priority", kind)


def sla_badge(value: str) -> str:
    kind = "overdue" if value == "Overdue" else "closed" if value == "Closed" else "track"
    return badge(value, kind)


# ============================================================
# Secrets and Optional Clients
# ============================================================

def get_secret_value(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.getenv(name, default)

    if value is None:
        return default
    return str(value)


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    api_key = get_secret_value("GEMINI_API_KEY", "")
    if api_key == "" or genai is None:
        return None
    return genai.Client(api_key=api_key)


def get_gemini_model_name() -> str:
    return get_secret_value("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def transformers_enabled() -> bool:
    return get_secret_value("ENABLE_TRANSFORMERS", "false").lower() == "true"


@st.cache_resource(show_spinner="Loading optional emotion model...")
def load_emotion_pipeline():
    # Optional dependency. Do not put transformers/torch in requirements unless you enable this.
    from transformers import pipeline
    model_name = get_secret_value("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
    return pipeline("text-classification", model=model_name, top_k=None)


# ============================================================
# Storage
# ============================================================

def load_records() -> List[Dict]:
    if not os.path.exists(DATA_FILE):
        return []

    try:
        df = pd.read_csv(DATA_FILE)
        if df.empty:
            return []
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[EXPECTED_COLUMNS].to_dict("records")
    except Exception:
        return []


def save_records(records: List[Dict]) -> None:
    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[EXPECTED_COLUMNS].to_csv(DATA_FILE, index=False)


def clear_records_file() -> None:
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)


def prepare_dataframe(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if len(df) > 0:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0).astype(int)
    return df[EXPECTED_COLUMNS]


def create_case_id() -> str:
    return "CASE-" + datetime.now().strftime("%Y%m%d%H%M%S%f")[-14:]


def update_case_record(case_id: str, updates: Dict) -> None:
    records = load_records()
    for record in records:
        if str(record.get("case_id", "")) == str(case_id):
            for key, value in updates.items():
                record[key] = value
            record["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            break
    save_records(records)
    st.session_state.records = records


# ============================================================
# Text, Topic, Emotion, Sarcasm, Risk
# ============================================================

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_topic_rule(text: str, selected_issue_type: str = "Not sure yet", chat_history: Optional[List[Dict]] = None) -> str:
    """
    Detect the topic from the current message and recent chat context.
    The issue-type dropdown is only a weak hint; it should not override clear
    intent from the customer message or conversation.
    """
    t = str(text).lower().strip()
    words = t.split()

    # Strong delivery/order-status patterns must be checked before generic
    # product words. Customers often say "where is my product" when they mean
    # delivery/tracking, not product quality.
    delivery_phrases = [
        "where is my product", "where is my item", "where is my order",
        "where is my package", "where is my parcel", "missing package",
        "missing parcel", "package missing", "product missing",
        "not received", "did not receive", "haven't received", "have not received",
        "when will it arrive", "when arrive", "arrival date", "possible date",
        "expected date", "delivery date", "eta", "tracking", "shipment",
        "shipping", "delayed", "delay", "late delivery", "out for delivery"
    ]
    if any(p in t for p in delivery_phrases):
        return "Delivery Issue"

    if any(p in t for p in ["refund", "money back", "return my money", "chargeback"]):
        return "Refund Issue"

    if any(p in t for p in ["charged", "payment", "billing", "bill", "invoice", "paid", "card", "transaction", "charged twice"]):
        return "Billing Issue"

    if any(p in t for p in ["broken", "damaged", "defective", "faulty", "quality", "wrong item", "replacement", "not working product"]):
        return "Product Issue"

    if any(p in t for p in ["support", "customer service", "agent", "no reply", "ignored", "nobody helped", "rude", "no one helped"]):
        return "Customer Service Issue"

    if any(p in t for p in ["app", "website", "login", "password", "error", "crash", "bug", "reset link", "technical", "otp"]):
        return "Technical Issue"

    # Short follow-up questions should inherit the previous conversation topic
    # instead of being forced by the dropdown.
    if len(words) <= 5:
        context_topic = infer_topic_from_history(chat_history)
        if context_topic != "General Complaint":
            return context_topic

    if selected_issue_type and selected_issue_type != "Not sure yet" and len(words) > 5:
        return selected_issue_type

    return "General Complaint"


def infer_topic_from_history(chat_history: Optional[List[Dict]]) -> str:
    """Infer the current issue topic from recent customer messages."""
    if not chat_history:
        return "General Complaint"
    recent_user_text = " ".join(
        str(item.get("content", ""))
        for item in chat_history[-8:]
        if item.get("role") == "user"
    ).lower()
    # Avoid recursion by using direct patterns only.
    if any(p in recent_user_text for p in ["where is", "package", "parcel", "order", "shipment", "shipping", "delivery", "tracking", "arrive", "delayed", "missing"]):
        return "Delivery Issue"
    if any(p in recent_user_text for p in ["refund", "money back", "return my money", "chargeback"]):
        return "Refund Issue"
    if any(p in recent_user_text for p in ["charged", "payment", "billing", "invoice", "card", "paid"]):
        return "Billing Issue"
    if any(p in recent_user_text for p in ["broken", "damaged", "defective", "faulty", "wrong item", "replacement"]):
        return "Product Issue"
    if any(p in recent_user_text for p in ["login", "password", "app", "website", "error", "bug", "crash"]):
        return "Technical Issue"
    if any(p in recent_user_text for p in ["support", "agent", "no reply", "ignored", "nobody helped", "rude"]):
        return "Customer Service Issue"
    return "General Complaint"


def detect_emotion_rule(text: str) -> str:
    """
    Fast fallback emotion detector.
    This is intentionally stronger than a basic sentiment rule so manager metrics
    do not stay Neutral for clear complaints such as refund, missing package,
    billing problem, or profanity.
    """
    t = str(text).lower()

    angry_words = [
        "angry", "furious", "mad", "worst", "terrible", "awful", "hate",
        "unacceptable", "ridiculous", "fuck", "shit", "damn", "useless",
        "never buy", "scam", "fraud", "cheated", "asshole", "bastard", "fuck off"
    ]
    disappointed_words = [
        "disappointed", "upset", "frustrated", "annoyed", "unhappy", "poor",
        "bad experience", "not satisfied", "missing", "delayed", "late",
        "refund", "money back", "charged", "billing", "broken", "damaged",
        "wrong item", "not working", "no reply", "ignored", "nobody helped"
    ]
    confused_words = [
        "confused", "unclear", "do not understand", "don't understand",
        "why", "how", "what happened", "not sure", "where is", "cant find", "can't find", "possible date", "eta", "when arrive", "delivery date"
    ]
    satisfied_words = [
        "thank", "thanks", "appreciate", "great", "good", "excellent",
        "resolved", "solved", "happy", "helpful"
    ]

    if any(w in t for w in angry_words):
        return "Angry"
    if any(w in t for w in disappointed_words):
        return "Disappointed"
    if any(w in t for w in confused_words):
        return "Confused"
    if any(w in t for w in satisfied_words):
        return "Satisfied"
    return "Neutral"


def detect_emotion_optional_transformer(text: str, fallback_emotion: str) -> Tuple[str, str]:
    if not transformers_enabled():
        return fallback_emotion, "rule"
    try:
        clf = load_emotion_pipeline()
        results = clf(text)
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        top = max(results, key=lambda x: x.get("score", 0))
        raw = str(top.get("label", "neutral")).lower()
        mapper = {
            "anger": "Angry",
            "disgust": "Angry",
            "fear": "Confused",
            "joy": "Satisfied",
            "neutral": "Neutral",
            "sadness": "Disappointed",
            "surprise": "Confused",
        }
        return mapper.get(raw, fallback_emotion), "transformer"
    except Exception:
        return fallback_emotion, "rule"


def detect_sarcasm_rule(text: str) -> Tuple[bool, str]:
    t = str(text).lower()
    positive_words = ["great", "amazing", "wonderful", "perfect", "excellent", "fantastic", "nice"]
    negative_context = ["late", "delayed", "broken", "damaged", "wrong", "charged", "refund", "nobody", "ignored", "not working", "worst"]
    sarcasm_markers = ["yeah right", "thanks for nothing", "so helpful", "what a joke", "only", "again"]

    if any(p in t for p in positive_words) and any(n in t for n in negative_context):
        return True, "Positive wording appears with a negative complaint context."
    if any(m in t for m in sarcasm_markers) and any(n in t for n in negative_context):
        return True, "Message contains sarcasm markers with dissatisfaction."
    return False, "No sarcasm pattern detected."


def get_risk_level(score: int) -> str:
    if score >= HIGH_RISK_THRESHOLD:
        return "High"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def calculate_risk(emotion: str, topic: str, text: str, sarcastic: bool) -> Tuple[int, str]:
    t = str(text).lower()
    base_scores = {"Satisfied": 5, "Neutral": 15, "Confused": 30, "Disappointed": 45, "Angry": 75}
    score = base_scores.get(emotion, 15)
    reasons = [f"Base score from emotion: {emotion}."]

    if any(w in t for w in ["refund", "money back", "chargeback"]):
        score += 10
        reasons.append("Customer mentioned refund or money back.")
    if any(w in t for w in ["cancel", "cancellation", "never buy", "unsubscribe"]):
        score += 15
        reasons.append("Customer may be considering cancellation or churn.")
    if any(w in t for w in ["again", "still", "no reply", "nobody helped", "ignored", "waiting", "third time", "many times"]):
        score += 10
        reasons.append("Message suggests repeated or unresolved support problems.")
    if topic in ["Refund Issue", "Billing Issue", "Customer Service Issue"]:
        score += 5
        reasons.append(f"{topic} is a sensitive complaint category.")
    if sarcastic:
        score += 10
        reasons.append("Sarcasm or indirect dissatisfaction detected.")

    score = max(0, min(score, 100))
    return score, " ".join(reasons)


# ============================================================
# Dynamic Reply Generator
# ============================================================

def is_abusive_only(message: str) -> bool:
    """True when the message is only abuse/profanity and no actual issue details."""
    t = clean_text(message)
    abusive_only = {
        "fuck", "fuck you", "fuck off", "shit", "damn", "stupid",
        "idiot", "asshole", "bastard", "moron", "nonsense"
    }
    return t in abusive_only


def get_smalltalk_reply(message: str) -> Optional[str]:
    """
    Handle greetings and thanks before case creation.
    These messages should not create a support case.
    """
    t = clean_text(message)

    greetings = {
        "hi", "hai", "hello", "hey", "hii", "helo", "good morning",
        "good afternoon", "good evening", "yo"
    }
    thanks = {"thanks", "thank you", "thankyou", "ty", "ok thanks", "okay thanks"}
    bye = {"bye", "goodbye", "see you", "thank you bye", "thanks bye"}

    if t in greetings:
        return (
            "Hi! I’m here to help. Tell me what happened, or choose an issue type like "
            "delivery, refund, billing, product, technical, or support."
        )

    if t in thanks:
        return "You’re welcome. Send any extra details here if you want the team to review them."

    if t in bye:
        return "Thank you for contacting support. Have a good day."

    return None


def looks_like_reference(text: str) -> bool:
    value = str(text).strip()
    if len(value) < 3 or len(value) > 30:
        return False
    return bool(re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", value)) or value.isdigit()


def build_customer_reply(
    message: str,
    topic: str,
    risk_level: str,
    emotion: str,
    selected_issue_type: str,
    chat_history: Optional[List[Dict]] = None,
) -> str:
    """Create a customer-facing reply that is specific to the current message and context."""
    raw = str(message).strip()
    t = raw.lower()
    short = len(raw.split()) <= 5
    context_topic = infer_topic_from_history(chat_history)
    effective_topic = topic if topic != "General Complaint" else context_topic

    if looks_like_reference(raw):
        return (
            f"Thank you. I’ve added reference **{raw}** to your support case. "
            "Our team can use it to review the issue more accurately."
        )

    # Context-aware follow-up replies.
    eta_words = ["possible date", "date", "when", "arrive", "arrival", "eta", "expected"]
    if any(w in t for w in eta_words) and effective_topic == "Delivery Issue":
        return (
            "I understand you want the expected delivery date. I can’t see live tracking from here, "
            "but I’ve added your request for an ETA to the case so the team can check the shipment status and update you."
        )

    if any(w in t for w in ["status", "update", "any update", "what happened"]) and effective_topic != "General Complaint":
        return (
            "I’ve added your request for an update to the support case. "
            "The team will review the latest details and follow up with the next step."
        )

    # If the customer is upset but has already given an issue, de-escalate and confirm the case.
    angry_prefix = "I’m sorry for the frustrating experience. " if risk_level == "High" or emotion == "Angry" else ""

    if effective_topic == "Delivery Issue":
        if any(p in t for p in ["where is", "missing", "not received", "haven't received", "have not received"]):
            return angry_prefix + (
                "I’ve recorded this as a missing or delayed delivery issue. "
                "The team can check the shipment status, tracking details, and next update."
            )
        return angry_prefix + (
            "I’ve recorded this as a delivery issue so the team can check the shipment status, delay reason, and next update."
        )

    if effective_topic == "Refund Issue":
        return angry_prefix + (
            "I’ve recorded your refund concern so the team can review the order and follow up with the next steps."
        )

    if effective_topic == "Billing Issue":
        return angry_prefix + (
            "I’ve recorded this billing concern so the team can review the charge or payment details carefully."
        )

    if effective_topic == "Product Issue":
        return angry_prefix + (
            "I’ve recorded this product issue so the team can review the item condition and possible replacement or support options."
        )

    if effective_topic == "Technical Issue":
        return angry_prefix + (
            "I’ve recorded this technical issue so the team can review the error and check what is causing the problem."
        )

    if effective_topic == "Customer Service Issue":
        return angry_prefix + (
            "I’ve recorded this as a support experience issue so the team can review the previous response and follow up properly."
        )

    # Do not create vague robotic replies for very short unclear messages.
    if short:
        return (
            "I can help with that. Could you tell me what happened and whether it is about delivery, refund, billing, product, technical, or support?"
        )

    if "thank" in t or "thanks" in t:
        return "Thank you for sharing that. I’ve recorded your feedback for the team."

    return (
        "Thank you for explaining. I’ve recorded your message as a support case so our team can review it."
    )



# ============================================================
# Gemini Manager Analysis Optional
# ============================================================

def extract_json(text: str) -> Dict:
    cleaned = str(text).strip().replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def analyze_with_gemini(message: str, topic_hint: str, chat_history: Optional[List[Dict]]) -> Optional[Dict]:
    client = get_gemini_client()
    if client is None:
        return None

    history_text = ""
    if chat_history:
        for item in chat_history[-8:]:
            history_text += f"{item.get('role', '')}: {item.get('content', '')}\n"

    prompt = f"""
You are a private customer-support intelligence analyst.
Analyze the latest customer message and recent chat context.
Return only valid JSON with this exact structure:
{{
  "emotion": "Angry | Confused | Disappointed | Satisfied | Neutral",
  "complaint_topic": "Refund Issue | Delivery Issue | Billing Issue | Product Issue | Customer Service Issue | Technical Issue | General Complaint",
  "customer_intent": "short internal explanation",
  "business_risk": "short internal business risk",
  "risk_score": 0,
  "risk_reason": "short internal reason",
  "recommended_action": "short manager action"
}}

Topic hint from UI/rules: {topic_hint}
Recent chat:
{history_text}
Latest customer message:
{message!r}
"""
    try:
        response = client.models.generate_content(
            model=get_gemini_model_name(),
            contents=prompt,
            config={"temperature": 0.15, "response_mime_type": "application/json"},
        )
        if not response.text:
            return None
        result = extract_json(response.text)
        return result
    except Exception:
        return None


def normalize_analysis(result: Dict, fallback: Dict) -> Dict:
    allowed_emotions = ["Angry", "Confused", "Disappointed", "Satisfied", "Neutral"]
    allowed_topics = ["Refund Issue", "Delivery Issue", "Billing Issue", "Product Issue", "Customer Service Issue", "Technical Issue", "General Complaint"]

    emotion = str(result.get("emotion", fallback["emotion"]))
    topic = str(result.get("complaint_topic", fallback["complaint_topic"]))
    if emotion not in allowed_emotions:
        emotion = fallback["emotion"]
    if topic not in allowed_topics:
        topic = fallback["complaint_topic"]

    # Do not let Gemini downgrade clear complaint signals to Neutral.
    if fallback.get("emotion") in ["Angry", "Disappointed", "Confused"] and emotion == "Neutral":
        emotion = fallback["emotion"]

    try:
        risk_score = int(result.get("risk_score", fallback["risk_score"]))
    except Exception:
        risk_score = fallback["risk_score"]

    # Do not let Gemini reduce a locally detected risk signal.
    risk_score = max(risk_score, int(fallback.get("risk_score", 0)))
    risk_score = max(0, min(risk_score, 100))

    return {
        **fallback,
        "emotion": emotion,
        "complaint_topic": topic,
        "risk_score": risk_score,
        "risk_level": get_risk_level(risk_score),
        "customer_intent": str(result.get("customer_intent", fallback["customer_intent"])),
        "business_risk": str(result.get("business_risk", fallback["business_risk"])),
        "risk_reason": str(result.get("risk_reason", fallback["risk_reason"])),
        "recommended_action": str(result.get("recommended_action", fallback["recommended_action"])),
        "analysis_source": "Gemini + local reply",
    }


def analyze_feedback(message: str, selected_issue_type: str, chat_history: Optional[List[Dict]]) -> Dict:
    cleaned = clean_text(message)

    smalltalk_reply = get_smalltalk_reply(message)
    if smalltalk_reply is not None:
        return {
            "clean_text": cleaned,
            "analysis_source": "Small talk handler",
            "emotion": "Neutral",
            "sarcasm_detected": "No",
            "sarcasm_reason": "Small-talk message; no complaint detected.",
            "complaint_topic": "General Complaint",
            "customer_intent": "Customer is greeting or using small talk.",
            "business_risk": "No business risk detected because no issue was described.",
            "risk_score": 0,
            "risk_level": "Low",
            "risk_reason": "Greeting or small-talk message only.",
            "recommended_action": "No manager action needed unless the customer describes an issue.",
            "customer_reply": smalltalk_reply,
            "actionable_case": False,
        }

    if is_abusive_only(message):
        context_topic = infer_topic_from_history(chat_history)
        if context_topic != "General Complaint":
            return {
                "clean_text": cleaned,
                "analysis_source": "Abuse/context handler",
                "emotion": "Angry",
                "sarcasm_detected": "No",
                "sarcasm_reason": "No sarcasm pattern detected.",
                "complaint_topic": context_topic,
                "customer_intent": "Customer is angry and wants the existing issue handled urgently.",
                "business_risk": "High escalation risk because the customer is using abusive language after reporting an issue.",
                "risk_score": 88,
                "risk_level": "High",
                "risk_reason": "Abusive language detected in an existing support context.",
                "recommended_action": "Manager or support lead should review and follow up quickly.",
                "customer_reply": "I’m sorry you’re upset. I’ve updated your existing case as urgent so the team can review it more carefully.",
                "actionable_case": True,
            }
        return {
            "clean_text": cleaned,
            "analysis_source": "Abuse handler",
            "emotion": "Angry",
            "sarcasm_detected": "No",
            "sarcasm_reason": "No sarcasm pattern detected.",
            "complaint_topic": "General Complaint",
            "customer_intent": "Customer is upset but has not described the issue yet.",
            "business_risk": "No actionable issue yet, but customer tone is negative.",
            "risk_score": 0,
            "risk_level": "Low",
            "risk_reason": "Abusive-only message without issue details.",
            "recommended_action": "No manager action needed unless the customer describes the issue.",
            "customer_reply": "I’m sorry you’re upset. I can help record the problem for our team, but I need a little detail about what went wrong.",
            "actionable_case": False,
        }

    topic = detect_topic_rule(cleaned, selected_issue_type, chat_history)
    emotion_rule = detect_emotion_rule(cleaned)
    emotion, emotion_source = detect_emotion_optional_transformer(message, emotion_rule)
    sarcastic, sarcasm_reason = detect_sarcasm_rule(message)
    risk_score, risk_reason = calculate_risk(emotion, topic, cleaned, sarcastic)
    risk_level = get_risk_level(risk_score)

    fallback = {
        "clean_text": cleaned,
        "analysis_source": f"Local rules ({emotion_source})",
        "emotion": emotion,
        "sarcasm_detected": "Yes" if sarcastic else "No",
        "sarcasm_reason": sarcasm_reason,
        "complaint_topic": topic,
        "customer_intent": "Customer wants the issue reviewed and resolved.",
        "business_risk": "Customer satisfaction may decrease if the issue is not handled promptly.",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "recommended_action": "Review the case, assign the correct team, and follow up if the risk is medium or high.",
        "actionable_case": True,
    }

    gemini_result = analyze_with_gemini(message, topic, chat_history)
    analysis = normalize_analysis(gemini_result, fallback) if gemini_result else fallback

    # Always generate the customer reply locally for consistency and speed.
    analysis["customer_reply"] = build_customer_reply(
        message=message,
        topic=analysis["complaint_topic"],
        risk_level=analysis["risk_level"],
        emotion=analysis["emotion"],
        selected_issue_type=selected_issue_type,
        chat_history=chat_history,
    )
    return analysis


def build_coupon_message(record: Dict) -> str:
    """Message a manager can copy after approving a recovery coupon."""
    code = str(record.get("coupon_code", "")).strip()
    offer = str(record.get("coupon_offer", "customer recovery offer")).strip()
    topic = str(record.get("complaint_topic", "your recent issue")).lower()

    if not code:
        return "No coupon code is available for this case."

    return (
        f"We’re sorry for the inconvenience with {topic}. "
        f"As a goodwill gesture, we’d like to offer you {offer}. "
        f"You can use this coupon code on your next order: {code}."
    )


def reanalyze_existing_records() -> int:
    """Re-run local/Gemini analysis for existing demo rows after rule/model changes."""
    records = load_records()
    updated_count = 0

    for record in records:
        message = str(record.get("message", "")).strip()
        if not message:
            continue

        conversation = str(record.get("conversation", ""))
        fake_history = [{"role": "user", "content": conversation or message}]
        analysis = analyze_feedback(message, "Not sure yet", fake_history)
        coupon = suggest_recovery_coupon(
            analysis["risk_level"],
            analysis["risk_score"],
            analysis["complaint_topic"],
            str(record.get("case_id", create_case_id()))
        )

        for key in [
            "clean_text", "analysis_source", "emotion", "sarcasm_detected",
            "sarcasm_reason", "complaint_topic", "customer_intent", "business_risk",
            "risk_score", "risk_level", "risk_reason", "recommended_action", "customer_reply"
        ]:
            record[key] = analysis[key]

        if str(record.get("coupon_status", "")) not in ["Approved", "Sent", "Rejected"]:
            record["coupon_offer"] = coupon["coupon_offer"]
            record["coupon_code"] = coupon["coupon_code"]
            record["coupon_status"] = coupon["coupon_status"]
            record["coupon_reason"] = coupon["coupon_reason"]

        record["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_count += 1

    save_records(records)
    st.session_state.records = records
    return updated_count


# ============================================================
# Conversation and Case Helpers
# ============================================================

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


def get_active_case_id() -> str:
    if st.session_state.active_case_id == "":
        st.session_state.active_case_id = create_case_id()
    return st.session_state.active_case_id


def generate_coupon_code(case_id: str, discount_percent: int) -> str:
    short_id = str(case_id).replace("CASE-", "")[-6:]
    return f"VOC-{short_id}-{discount_percent}"


def suggest_recovery_coupon(risk_level: str, risk_score: int, complaint_topic: str, case_id: str) -> Dict:
    risk_score = int(risk_score)
    if risk_level == "High":
        discount = 20 if risk_score >= 85 else 15
        if complaint_topic in ["Refund Issue", "Billing Issue"]:
            offer = f"{discount}% goodwill coupon after billing/refund review"
            reason = "High-risk money-related complaint. Manager should review before offering compensation."
        elif complaint_topic == "Delivery Issue":
            offer = f"{discount}% apology coupon or free shipping recovery offer"
            reason = "High-risk delivery issue. Recovery offer may help rebuild trust."
        else:
            offer = f"{discount}% customer recovery coupon"
            reason = "High-risk dissatisfaction. Recovery offer may reduce escalation risk."
        return {"coupon_offer": offer, "coupon_code": generate_coupon_code(case_id, discount), "coupon_status": "Suggested", "coupon_reason": reason}

    if risk_level == "Medium":
        return {
            "coupon_offer": "10% goodwill coupon if issue remains unresolved",
            "coupon_code": generate_coupon_code(case_id, 10),
            "coupon_status": "Optional",
            "coupon_reason": "Medium-risk case. Coupon is optional if resolution is delayed.",
        }

    return {"coupon_offer": "No coupon needed", "coupon_code": "", "coupon_status": "Not Required", "coupon_reason": "Low-risk case. Normal support is enough."}


def create_or_update_active_case(customer_id: str, user_message: str, analysis: Dict, coupon: Dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    case_id = get_active_case_id()
    conversation = format_conversation(st.session_state.chat_messages)
    records = load_records()
    case_found = False

    for record in records:
        if str(record.get("case_id", "")) == str(case_id):
            record.update({
                "customer_id": customer_id,
                "message": user_message,
                "conversation": conversation,
                "clean_text": analysis["clean_text"],
                "analysis_source": analysis["analysis_source"],
                "emotion": analysis["emotion"],
                "sarcasm_detected": analysis["sarcasm_detected"],
                "sarcasm_reason": analysis["sarcasm_reason"],
                "complaint_topic": analysis["complaint_topic"],
                "customer_intent": analysis["customer_intent"],
                "business_risk": analysis["business_risk"],
                "risk_score": analysis["risk_score"],
                "risk_level": analysis["risk_level"],
                "risk_reason": analysis["risk_reason"],
                "recommended_action": analysis["recommended_action"],
                "customer_reply": analysis["customer_reply"],
                "last_updated": now,
            })
            if str(record.get("coupon_status", "")) in ["", "Not Required", "Optional", "Suggested"]:
                record.update(coupon)
            case_found = True
            break

    if not case_found:
        new_record = {
            "case_id": case_id,
            "customer_id": customer_id,
            "timestamp": now,
            "message": user_message,
            "conversation": conversation,
            "clean_text": analysis["clean_text"],
            "analysis_source": analysis["analysis_source"],
            "emotion": analysis["emotion"],
            "sarcasm_detected": analysis["sarcasm_detected"],
            "sarcasm_reason": analysis["sarcasm_reason"],
            "complaint_topic": analysis["complaint_topic"],
            "customer_intent": analysis["customer_intent"],
            "business_risk": analysis["business_risk"],
            "risk_score": analysis["risk_score"],
            "risk_level": analysis["risk_level"],
            "risk_reason": analysis["risk_reason"],
            "recommended_action": analysis["recommended_action"],
            "customer_reply": analysis["customer_reply"],
            "status": "New",
            "assigned_to": "Unassigned",
            "internal_notes": "",
            "resolution_action": "",
            **coupon,
            "last_updated": now,
        }
        records.append(new_record)

    save_records(records)
    st.session_state.records = records
    return case_id


# ============================================================
# Manager Data Helpers
# ============================================================

def enrich_manager_data(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) == 0:
        return data
    data = data.copy()
    data["timestamp_dt"] = pd.to_datetime(data["timestamp"], errors="coerce")
    now = pd.Timestamp.now()
    data["case_age_hours"] = ((now - data["timestamp_dt"]).dt.total_seconds() / 3600).fillna(0).clip(lower=0).round(1)
    data["priority_score"] = pd.to_numeric(data["risk_score"], errors="coerce").fillna(0).astype(int)
    data.loc[data["status"] == "Escalated", "priority_score"] += 10
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
    return data


def generate_manager_brief(record: Dict) -> str:
    return (
        f"Priority: {record.get('priority', record.get('risk_level', 'Unknown'))}\n\n"
        f"Main issue: {record.get('complaint_topic', 'General issue')} from customer {record.get('customer_id', '')}.\n\n"
        f"Why it matters: {record.get('business_risk', 'This may affect satisfaction.')}\n\n"
        f"Recommended action: {record.get('recommended_action', 'Review and follow up.')}\n\n"
        f"Recovery offer: {record.get('coupon_offer', 'No coupon needed')} | Status: {record.get('coupon_status', 'Not Required')}"
    )


def generate_journey_insight(customer_id: str, customer_data: pd.DataFrame) -> str:
    if len(customer_data) == 0:
        return "No customer journey available."
    first_score = int(customer_data.iloc[0]["risk_score"])
    latest_score = int(customer_data.iloc[-1]["risk_score"])
    trend = "increasing" if latest_score > first_score else "decreasing" if latest_score < first_score else "stable"
    latest = customer_data.iloc[-1]
    return (
        f"Customer {customer_id} has a {trend} risk pattern. Latest topic is {latest['complaint_topic']}, "
        f"latest emotion is {latest['emotion']}, and latest risk level is {latest['risk_level']}. "
        "Manager should review the latest transcript and follow up if the case is medium or high risk."
    )


# ============================================================
# Password
# ============================================================

def check_manager_access() -> bool:
    if st.session_state.get("manager_authenticated", False):
        return True

    st.markdown('<div class="glass-card"><div class="section-title">Manager Access</div><div class="muted">Protected internal workspace.</div></div>', unsafe_allow_html=True)
    password = st.text_input("Enter manager password", type="password")
    if st.button("Login"):
        correct = get_secret_value("MANAGER_PASSWORD", "")
        if correct == "":
            st.error("MANAGER_PASSWORD is missing in Streamlit Secrets.")
            return False
        if hmac.compare_digest(password, correct):
            st.session_state.manager_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ============================================================
# Updated UI, Customer-Level Workflow, and Journey Pages
# ============================================================

# Add one extra column for within-customer emotion history. Redefining this
# global works because storage functions read EXPECTED_COLUMNS at runtime.
EXPECTED_COLUMNS = [
    "case_id",
    "customer_id",
    "timestamp",
    "message",
    "conversation",
    "emotion_journey",
    "clean_text",
    "analysis_source",
    "emotion",
    "sarcasm_detected",
    "sarcasm_reason",
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
    "internal_notes",
    "resolution_action",
    "coupon_offer",
    "coupon_code",
    "coupon_status",
    "coupon_reason",
    "last_updated",
]

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(37,99,235,.13), transparent 28%),
                radial-gradient(circle at 90% 20%, rgba(20,184,166,.11), transparent 27%),
                linear-gradient(180deg, #f7fbff 0%, #eef4fb 55%, #f8fafc 100%) !important;
        }

        .hero {
            background:
                linear-gradient(135deg, #07111f 0%, #102238 55%, #153b58 100%) !important;
            border-radius: 34px !important;
            position: relative;
            overflow: hidden;
        }

        .hero:after {
            content: "";
            position: absolute;
            right: -70px;
            top: -80px;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(56,189,248,.35), transparent 65%);
            border-radius: 999px;
        }

        .ops-grid-card {
            background: rgba(255,255,255,.82);
            border: 1px solid rgba(203,213,225,.9);
            border-radius: 28px;
            padding: 24px;
            box-shadow: 0 22px 50px rgba(15,23,42,.08);
            backdrop-filter: blur(18px);
            min-height: 190px;
        }

        .ops-card-title {
            font-size: 21px;
            font-weight: 950;
            color: #07111f;
            letter-spacing: -.03em;
            margin-bottom: 8px;
        }

        .flow-step {
            display: flex;
            gap: 12px;
            align-items: flex-start;
            margin: 12px 0;
        }

        .flow-num {
            min-width: 28px;
            height: 28px;
            border-radius: 10px;
            background: #0f172a;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 13px;
        }

        .flow-text {
            color: #475569;
            line-height: 1.55;
            font-size: 14px;
        }

        .queue-card {
            background: white;
            border: 1px solid #dbeafe;
            border-left: 6px solid #2563eb;
            border-radius: 24px;
            padding: 18px 20px;
            box-shadow: 0 14px 36px rgba(37,99,235,.08);
        }

        .intel-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbeafe;
            border-radius: 26px;
            padding: 22px;
            box-shadow: 0 18px 40px rgba(15,23,42,.07);
            margin-bottom: 16px;
        }

        .mini-label {
            color: #64748b;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-weight: 900;
            margin-bottom: 4px;
        }

        .mini-value {
            color: #0f172a;
            font-size: 17px;
            font-weight: 850;
            margin-bottom: 14px;
        }

        .emotion-strip {
            display: flex;
            align-items: stretch;
            gap: 10px;
            overflow-x: auto;
            padding: 14px 4px 20px 4px;
        }

        .emotion-node {
            min-width: 165px;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 14px;
            box-shadow: 0 14px 30px rgba(15,23,42,.07);
        }

        .emotion-name {
            font-size: 18px;
            font-weight: 950;
            color: #0f172a;
            letter-spacing: -.02em;
        }

        .emotion-meta {
            color: #64748b;
            font-size: 12px;
            line-height: 1.5;
            margin-top: 6px;
        }

        .arrow-node {
            display: flex;
            align-items: center;
            color: #94a3b8;
            font-size: 24px;
            font-weight: 900;
        }

        .chat-wrap {
            background: rgba(255,255,255,.74);
            border: 1px solid rgba(226,232,240,.95);
            border-radius: 30px;
            padding: 22px;
            box-shadow: 0 22px 55px rgba(15,23,42,.08);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Updated Helpers
# ============================================================

def reset_chat_for_new_customer() -> None:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hi, I’m your support assistant. What type of issue are you facing today?"}
    ]
    st.session_state.active_case_id = ""
    st.session_state.selected_issue_type = "Not sure yet"


def make_emotion_event(emotion: str, risk_score: int, topic: str, message: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    short_message = str(message).replace("|", "/").replace("\n", " ")[:70]
    return f"{stamp}|{emotion}|{risk_score}|{topic}|{short_message}"


def append_emotion_event(existing: str, emotion: str, risk_score: int, topic: str, message: str) -> str:
    events = [line for line in str(existing or "").split("\n") if line.strip()]
    events.append(make_emotion_event(emotion, risk_score, topic, message))
    return "\n".join(events[-25:])


def parse_emotion_journey(text: str) -> List[Dict]:
    events = []
    for line in str(text or "").split("\n"):
        parts = line.split("|", 4)
        if len(parts) == 5:
            stamp, emotion, risk, topic, message = parts
            events.append({"time": stamp, "emotion": emotion, "risk": risk, "topic": topic, "message": message})
    return events


def render_emotion_journey(events: List[Dict]) -> None:
    if not events:
        st.info("No emotion journey has been recorded yet.")
        return

    html_parts = ['<div class="emotion-strip">']
    for i, event in enumerate(events):
        html_parts.append(
            f"""
            <div class="emotion-node">
                <div class="emotion-name">{safe_text(event.get('emotion',''))}</div>
                <div class="emotion-meta">
                    Risk: {safe_text(event.get('risk',''))}%<br>
                    Topic: {safe_text(event.get('topic',''))}<br>
                    {safe_text(event.get('time',''))}
                </div>
            </div>
            """
        )
        if i < len(events) - 1:
            html_parts.append('<div class="arrow-node">→</div>')
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def create_or_update_active_case(customer_id: str, user_message: str, analysis: Dict, coupon: Dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    case_id = get_active_case_id()
    conversation = format_conversation(st.session_state.chat_messages)
    records = load_records()
    case_found = False

    for record in records:
        if str(record.get("case_id", "")) == str(case_id):
            emotion_journey = append_emotion_event(
                record.get("emotion_journey", ""),
                analysis["emotion"],
                analysis["risk_score"],
                analysis["complaint_topic"],
                user_message,
            )
            record.update({
                "customer_id": customer_id,
                "message": user_message,
                "conversation": conversation,
                "emotion_journey": emotion_journey,
                "clean_text": analysis["clean_text"],
                "analysis_source": analysis["analysis_source"],
                "emotion": analysis["emotion"],
                "sarcasm_detected": analysis["sarcasm_detected"],
                "sarcasm_reason": analysis["sarcasm_reason"],
                "complaint_topic": analysis["complaint_topic"],
                "customer_intent": analysis["customer_intent"],
                "business_risk": analysis["business_risk"],
                "risk_score": analysis["risk_score"],
                "risk_level": analysis["risk_level"],
                "risk_reason": analysis["risk_reason"],
                "recommended_action": analysis["recommended_action"],
                "customer_reply": analysis["customer_reply"],
                "last_updated": now,
            })
            if str(record.get("coupon_status", "")) in ["", "Not Required", "Optional", "Suggested"]:
                record.update(coupon)
            case_found = True
            break

    if not case_found:
        new_record = {
            "case_id": case_id,
            "customer_id": customer_id,
            "timestamp": now,
            "message": user_message,
            "conversation": conversation,
            "emotion_journey": append_emotion_event("", analysis["emotion"], analysis["risk_score"], analysis["complaint_topic"], user_message),
            "clean_text": analysis["clean_text"],
            "analysis_source": analysis["analysis_source"],
            "emotion": analysis["emotion"],
            "sarcasm_detected": analysis["sarcasm_detected"],
            "sarcasm_reason": analysis["sarcasm_reason"],
            "complaint_topic": analysis["complaint_topic"],
            "customer_intent": analysis["customer_intent"],
            "business_risk": analysis["business_risk"],
            "risk_score": analysis["risk_score"],
            "risk_level": analysis["risk_level"],
            "risk_reason": analysis["risk_reason"],
            "recommended_action": analysis["recommended_action"],
            "customer_reply": analysis["customer_reply"],
            "status": "New",
            "assigned_to": "Unassigned",
            "internal_notes": "",
            "resolution_action": "",
            **coupon,
            "last_updated": now,
        }
        records.append(new_record)

    save_records(records)
    st.session_state.records = records
    return case_id


def customer_queue_summary(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) == 0:
        return data

    data = data.copy()
    if "timestamp_dt" not in data.columns:
        data["timestamp_dt"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["updated_dt"] = pd.to_datetime(data["last_updated"], errors="coerce")
    data["updated_dt"] = data["updated_dt"].fillna(data["timestamp_dt"])

    sorted_data = data.sort_values(
        by=["customer_id", "priority_score", "updated_dt"],
        ascending=[True, False, False],
    )
    summary = sorted_data.groupby("customer_id", as_index=False).head(1).copy()

    case_counts = data.groupby("customer_id")["case_id"].nunique()
    open_counts = data[data["status"] != "Resolved"].groupby("customer_id")["case_id"].nunique()
    max_risk = data.groupby("customer_id")["risk_score"].max()

    summary["case_count"] = summary["customer_id"].map(case_counts).fillna(0).astype(int)
    summary["open_case_count"] = summary["customer_id"].map(open_counts).fillna(0).astype(int)
    summary["max_customer_risk"] = summary["customer_id"].map(max_risk).fillna(summary["risk_score"]).astype(int)

    return summary.sort_values(by=["priority_score", "updated_dt"], ascending=[False, False])


def get_customer_representative_record(data: pd.DataFrame, customer_id: str) -> Dict:
    customer_rows = data[data["customer_id"] == customer_id].copy()
    if len(customer_rows) == 0:
        return {}
    customer_rows = customer_rows.sort_values(by=["priority_score", "case_age_hours"], ascending=[False, False])
    return customer_rows.iloc[0].to_dict()


# ============================================================
# Session State
# ============================================================

if "records" not in st.session_state:
    st.session_state.records = load_records()
if "manager_authenticated" not in st.session_state:
    st.session_state.manager_authenticated = False
if "chat_customer_id" not in st.session_state:
    st.session_state.chat_customer_id = "C001"
if "selected_issue_type" not in st.session_state:
    st.session_state.selected_issue_type = "Not sure yet"
if "active_case_id" not in st.session_state:
    st.session_state.active_case_id = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hi, I’m your support assistant. What type of issue are you facing today?"}
    ]


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("VoiceOps")
page = st.sidebar.radio("Workspace", ["AI Support Chat", "Manager Command Center", "Journey Monitor", "About System"])
st.sidebar.markdown("---")
if st.session_state.get("manager_authenticated", False):
    if st.sidebar.button("Logout Manager"):
        st.session_state.manager_authenticated = False
        st.rerun()


# ============================================================
# Page 1: AI Support Chat
# ============================================================

if page == "AI Support Chat":
    hero("AI Support Chat", "A guided support assistant that creates one case per customer conversation and gives message-specific replies.")

    flow_col, context_col = st.columns([1.0, 1.0], gap="large")

    with flow_col:
        st.markdown(
            """
            <div class="ops-grid-card">
                <div class="ops-card-title">Smart Support Flow</div>
                <div class="flow-step"><div class="flow-num">1</div><div class="flow-text">Greet the customer and understand the issue.</div></div>
                <div class="flow-step"><div class="flow-num">2</div><div class="flow-text">Use the issue type only as a hint, not a forced category.</div></div>
                <div class="flow-step"><div class="flow-num">3</div><div class="flow-text">One customer conversation updates one active support case.</div></div>
                <div class="flow-step"><div class="flow-num">4</div><div class="flow-text">Managers privately see risk, emotion, routing, and recovery actions.</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with context_col:
        st.markdown(
            """
            <div class="ops-grid-card">
                <div class="ops-card-title">Customer Context</div>
                <div class="muted">Change the customer ID to automatically start a fresh conversation for that customer.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        new_customer_id = st.text_input(
            "Customer ID or Order ID",
            value=st.session_state.chat_customer_id,
            placeholder="Example: C001 or ORD123",
        )
        if new_customer_id.strip() != st.session_state.chat_customer_id:
            st.session_state.chat_customer_id = new_customer_id.strip() or "C001"
            reset_chat_for_new_customer()
            st.rerun()

        st.session_state.selected_issue_type = st.selectbox(
            "Issue Type",
            ISSUE_TYPES,
            index=ISSUE_TYPES.index(st.session_state.selected_issue_type),
        )
        if st.session_state.active_case_id:
            st.info(f"Active case: {st.session_state.active_case_id}")
        if st.button("Start New Chat"):
            reset_chat_for_new_customer()
            st.rerun()

    st.markdown(
        """
        <div class="chat-wrap">
            <div class="section-title">Support Conversation</div>
            <div class="muted">Every actionable message updates the same support case until a new chat or new customer ID is started.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for chat in st.session_state.chat_messages:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    with st.form("support_message_form", clear_on_submit=True):
        user_message = st.text_input("Message", placeholder="Type your message here...")
        send_clicked = st.form_submit_button("Send Message")

    if send_clicked:
        if user_message.strip() == "":
            st.warning("Please type a message before sending.")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_message})
            with st.spinner("Reviewing message..."):
                analysis = analyze_feedback(user_message, st.session_state.selected_issue_type, st.session_state.chat_messages)
                assistant_reply = analysis["customer_reply"]
                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_reply})

                if analysis.get("actionable_case", True):
                    case_id = get_active_case_id()
                    coupon = suggest_recovery_coupon(
                        analysis["risk_level"],
                        analysis["risk_score"],
                        analysis["complaint_topic"],
                        case_id,
                    )
                    create_or_update_active_case(st.session_state.chat_customer_id, user_message, analysis, coupon)
            st.rerun()


# ============================================================
# Page 2: Manager Command Center
# ============================================================

elif page == "Manager Command Center":
    if not check_manager_access():
        st.stop()

    hero("Manager Command Center", "A customer-level operations console for risk, SLA, team ownership, and recovery decisions.")
    data = enrich_manager_data(prepare_dataframe(load_records()))

    if len(data) == 0:
        st.info("No support cases yet.")
    else:
        customer_summary = customer_queue_summary(data)
        open_customers = len(customer_summary[customer_summary["open_case_count"] > 0])
        high_priority = len(customer_summary[customer_summary["priority"].isin(["Critical", "High"])])
        overdue_customers = len(customer_summary[customer_summary["sla_status"] == "Overdue"])
        coupon_customers = len(customer_summary[customer_summary["coupon_status"].isin(["Suggested", "Optional", "Approved"])])
        avg_risk = round(customer_summary["risk_score"].mean(), 1)

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: kpi_card("Customers", len(customer_summary))
        with k2: kpi_card("Open Customers", open_customers)
        with k3: kpi_card("High Priority", high_priority)
        with k4: kpi_card("Overdue", overdue_customers)
        with k5: kpi_card("Avg Risk", avg_risk)

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["Priority Queue", "Case Review", "Coupon Center", "Analytics"])

        with tab1:
            st.subheader("Customer Priority Queue")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                selected_status = st.multiselect("Status", sorted(customer_summary["status"].dropna().unique().tolist()), default=sorted(customer_summary["status"].dropna().unique().tolist()))
            with f2:
                selected_priority = st.multiselect("Priority", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
            with f3:
                selected_sla = st.multiselect("SLA", ["Overdue", "On Track", "Closed"], default=["Overdue", "On Track", "Closed"])
            with f4:
                selected_owner = st.multiselect("Owner", sorted(customer_summary["assigned_to"].dropna().unique().tolist()), default=sorted(customer_summary["assigned_to"].dropna().unique().tolist()))

            filtered = customer_summary[
                customer_summary["status"].isin(selected_status)
                & customer_summary["priority"].isin(selected_priority)
                & customer_summary["sla_status"].isin(selected_sla)
                & customer_summary["assigned_to"].isin(selected_owner)
            ].sort_values(by=["priority_score", "case_age_hours"], ascending=[False, False])

            queue_columns = [
                "customer_id", "case_count", "open_case_count", "last_updated", "message",
                "complaint_topic", "emotion", "risk_score", "risk_level", "priority",
                "sla_status", "coupon_status", "assigned_to", "status"
            ]
            st.dataframe(filtered[queue_columns], use_container_width=True, hide_index=True)
            st.download_button("Download Customer Queue CSV", filtered.to_csv(index=False).encode("utf-8"), "customer_priority_queue.csv", "text/csv")

        with tab2:
            st.subheader("Customer Review Workspace")
            review_summary = customer_queue_summary(data)
            selected_customer = st.selectbox(
                "Select customer",
                review_summary["customer_id"].tolist(),
                format_func=lambda cid: (
                    f"{cid} | "
                    f"{review_summary[review_summary['customer_id'] == cid].iloc[0]['risk_level']} Risk | "
                    f"{review_summary[review_summary['customer_id'] == cid].iloc[0]['risk_score']}% | "
                    f"{review_summary[review_summary['customer_id'] == cid].iloc[0]['complaint_topic']}"
                ),
            )
            record = get_customer_representative_record(data, selected_customer)

            st.markdown(
                f"""
                <div class="queue-card">
                    <div class="case-title">{safe_text(record.get('customer_id',''))} · Customer Risk Profile</div>
                    <div class="case-meta">Latest update: {safe_text(record.get('last_updated', ''))} · Representative case age: {safe_text(record.get('case_age_hours', ''))} hours</div>
                    {priority_badge(record.get('priority','Low'))}
                    {risk_badge(record.get('risk_level','Low'))}
                    {sla_badge(record.get('sla_status','On Track'))}
                    {badge(record.get('status','New'), 'blue')}
                    {badge(record.get('complaint_topic','General Complaint'), 'purple')}
                    {badge(record.get('assigned_to','Unassigned'), 'gray')}
                    {badge(record.get('coupon_status','Not Required'), 'coupon')}
                    <div class="case-message">Latest customer issue: {safe_text(record.get('message',''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            left, right = st.columns([1.05, 0.95], gap="large")
            with left:
                st.markdown('<div class="intel-card"><div class="ops-card-title">Case Summary</div>', unsafe_allow_html=True)
                st.write(f"**Customer ID:** {record.get('customer_id', '')}")
                st.write(f"**Latest Issue:** {record.get('message', '')}")
                st.write(f"**Topic:** {record.get('complaint_topic', '')}")
                st.write(f"**Risk Score:** {record.get('risk_score', '')}%")
                st.write(f"**Risk Level:** {record.get('risk_level', '')}")
                st.write(f"**Priority:** {record.get('priority', '')}")
                st.write(f"**SLA:** {record.get('sla_status', '')}")
                st.markdown('</div>', unsafe_allow_html=True)

            with right:
                st.markdown('<div class="intel-card"><div class="ops-card-title">Internal Intelligence</div>', unsafe_allow_html=True)
                st.write(f"**Emotion:** {record.get('emotion', '')}")
                st.write(f"**Sarcasm:** {record.get('sarcasm_detected', 'No')}")
                st.write(f"**Intent:** {record.get('customer_intent', '')}")
                st.write(f"**Business Risk:** {record.get('business_risk', '')}")
                st.write(f"**Risk Reason:** {record.get('risk_reason', '')}")
                st.warning(record.get("recommended_action", "Review and follow up."))
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### Recovery Offer")
            coupon_status = str(record.get("coupon_status", "Not Required"))
            coupon_code = str(record.get("coupon_code", ""))
            st.write(f"**Offer:** {record.get('coupon_offer', 'No coupon needed')}")
            st.write(f"**Reason:** {record.get('coupon_reason', '')}")
            if coupon_code:
                st.code(coupon_code)
                st.text_area("Manager copy message", value=build_coupon_message(record), height=100, disabled=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                if coupon_status in ["Suggested", "Optional"] and st.button("Approve Coupon", key=f"approve_{record['case_id']}"):
                    update_case_record(record["case_id"], {"coupon_status": "Approved", "resolution_action": f"Coupon approved: {coupon_code}"})
                    st.rerun()
            with c2:
                if coupon_status in ["Suggested", "Optional"] and st.button("Reject Coupon", key=f"reject_{record['case_id']}"):
                    update_case_record(record["case_id"], {"coupon_status": "Rejected", "resolution_action": "Coupon rejected by manager."})
                    st.rerun()
            with c3:
                if coupon_status in ["Suggested", "Optional", "Approved"] and coupon_code and st.button("Mark Coupon Sent", key=f"sent_{record['case_id']}"):
                    update_case_record(record["case_id"], {"coupon_status": "Sent", "resolution_action": f"Coupon sent: {coupon_code}", "status": "Resolved"})
                    st.rerun()

            st.markdown("---")
            st.markdown("### Case Handling")
            with st.form("case_update_form"):
                u1, u2 = st.columns(2)
                current_status = record["status"] if record["status"] in STATUS_OPTIONS else "New"
                current_owner = record["assigned_to"] if record["assigned_to"] in OWNER_OPTIONS else "Unassigned"
                with u1:
                    new_status = st.selectbox("Status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(current_status))
                    new_owner = st.selectbox("Assign To", OWNER_OPTIONS, index=OWNER_OPTIONS.index(current_owner))
                with u2:
                    notes = st.text_area("Internal Notes", value=str(record.get("internal_notes", "")), height=110)
                    action = st.text_area("Resolution Action", value=str(record.get("resolution_action", "")), height=110)
                if st.form_submit_button("Save Case Update"):
                    update_case_record(record["case_id"], {"status": new_status, "assigned_to": new_owner, "internal_notes": notes, "resolution_action": action})
                    st.rerun()

        with tab3:
            st.subheader("Coupon Center")
            coupon_data = customer_queue_summary(data[data["coupon_status"].isin(["Suggested", "Optional", "Approved", "Sent", "Rejected"])])
            if len(coupon_data) == 0:
                st.info("No coupon-related customers yet.")
            else:
                st.dataframe(
                    coupon_data[["customer_id", "complaint_topic", "risk_score", "risk_level", "coupon_offer", "coupon_code", "coupon_status", "status"]],
                    use_container_width=True,
                    hide_index=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                with c1: kpi_card("Suggested", len(data[data["coupon_status"] == "Suggested"]))
                with c2: kpi_card("Approved", len(data[data["coupon_status"] == "Approved"]))
                with c3: kpi_card("Sent", len(data[data["coupon_status"] == "Sent"]))
                with c4: kpi_card("Rejected", len(data[data["coupon_status"] == "Rejected"]))

        with tab4:
            st.subheader("Analytics")
            a1, a2 = st.columns(2)
            with a1:
                risk_table = customer_summary["risk_level"].value_counts().reset_index()
                risk_table.columns = ["Risk Level", "Customers"]
                st.bar_chart(risk_table.set_index("Risk Level"))
            with a2:
                status_table = customer_summary["status"].value_counts().reset_index()
                status_table.columns = ["Status", "Customers"]
                st.bar_chart(status_table.set_index("Status"))
            a3, a4 = st.columns(2)
            with a3:
                topic_table = customer_summary["complaint_topic"].value_counts().reset_index()
                topic_table.columns = ["Topic", "Customers"]
                st.bar_chart(topic_table.set_index("Topic"))
            with a4:
                owner_table = customer_summary["assigned_to"].value_counts().reset_index()
                owner_table.columns = ["Owner", "Customers"]
                st.bar_chart(owner_table.set_index("Owner"))
            st.markdown("---")
            if st.button("Re-analyze Existing Cases"):
                count = reanalyze_existing_records()
                st.success(f"Re-analyzed {count} cases with the latest emotion/risk rules.")
                st.rerun()
            if st.button("Clear All Demo Data"):
                clear_records_file()
                st.session_state.records = []
                st.session_state.active_case_id = ""
                st.session_state.chat_messages = [{"role": "assistant", "content": "Hi, I’m your support assistant. What type of issue are you facing today?"}]
                st.rerun()


# ============================================================
# Page 3: Journey Monitor
# ============================================================

elif page == "Journey Monitor":
    if not check_manager_access():
        st.stop()
    hero("Customer Emotion Journey", "Visualize how each customer’s emotion, topic, and risk evolve across support interactions.")
    data = enrich_manager_data(prepare_dataframe(load_records()))
    if len(data) == 0:
        st.info("No customer journey available yet.")
    else:
        data["timestamp_dt"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data = data.sort_values("timestamp_dt")
        selected_customer = st.selectbox("Select customer", sorted(data["customer_id"].dropna().unique().tolist()))
        customer_data = data[data["customer_id"] == selected_customer].copy().sort_values("timestamp_dt")
        latest = get_customer_representative_record(enrich_manager_data(customer_data), selected_customer)

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Current Emotion", latest.get("emotion", ""))
        with c2: kpi_card("Current Risk", f"{latest.get('risk_score', '')}%")
        with c3: kpi_card("Risk Level", latest.get("risk_level", ""))
        with c4: kpi_card("Recovery", latest.get("coupon_status", ""))

        st.markdown("---")
        st.subheader("Customer Emotion Journey")
        all_events: List[Dict] = []
        for _, row in customer_data.iterrows():
            events = parse_emotion_journey(row.get("emotion_journey", ""))
            if events:
                all_events.extend(events)
            else:
                all_events.append({
                    "time": str(row.get("timestamp", "")),
                    "emotion": str(row.get("emotion", "")),
                    "risk": str(row.get("risk_score", "")),
                    "topic": str(row.get("complaint_topic", "")),
                    "message": str(row.get("message", ""))[:70],
                })
        render_emotion_journey(all_events)

        st.subheader("Journey Table")
        st.dataframe(
            customer_data[["timestamp", "message", "emotion", "complaint_topic", "risk_score", "risk_level", "coupon_status", "status"]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Risk Over Time")
        chart_data = customer_data[["timestamp_dt", "risk_score"]].dropna().set_index("timestamp_dt")
        if len(chart_data) > 0:
            st.line_chart(chart_data)
        else:
            st.info("Not enough data to show a risk chart yet.")

        if st.button("Generate Journey Insight"):
            st.warning(generate_journey_insight(selected_customer, customer_data))


# ============================================================
# Page 4: About
# ============================================================

elif page == "About System":
    hero("About VoiceOps AI Support Desk", "A guided AI support assistant with a customer-level operations dashboard for risk, workflow, and recovery offers.")
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Project Summary</div>
            <div class="muted">
                Customers chat with a guided support assistant. One conversation becomes one support case.
                Managers review customers, not individual chat messages, and privately see risk, emotion,
                ownership, SLA, and recovery actions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        Features:
        - Guided support chatbot with message-specific replies
        - New customer ID automatically starts a fresh conversation
        - Customer-level priority queue
        - Case review by customer ID and risk
        - Internal intelligence without exposing customer transcript clutter
        - Optional Gemini manager analysis
        - Optional transformer emotion model
        - Sarcasm-aware risk adjustment
        - Visual customer emotion journey
        - Coupon approval workflow kept private for managers
        """
    )
    st.subheader("Performance Design")
    st.write(
        """
        The app runs fast by default using local rule-based analysis and optional Gemini API analysis.
        Transformer models are disabled by default because they can slow down Streamlit Cloud.
        You can enable them later using `ENABLE_TRANSFORMERS = "true"` in Streamlit Secrets and adding
        `transformers` and `torch` to requirements.txt.
        """
    )
