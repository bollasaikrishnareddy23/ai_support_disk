# VoiceOps AI Support Desk

A Streamlit-based AI customer support and complaint recovery system.

## What it does

- Guided real-time support chatbot
- One chat session becomes one support case
- Dynamic customer replies based on the issue
- Private manager command center
- Emotion, topic, sarcasm, risk, priority, and SLA tracking
- Coupon recovery suggestion and approval workflow
- Customer journey monitor
- CSV storage for demo use

## Project structure

```text
voiceops_ai_support_desk/
├── app.py
├── requirements.txt
├── README.md
├── sample_chats.txt
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud setup

Add these secrets in Streamlit Cloud:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
MANAGER_PASSWORD = "your_manager_password_here"
GEMINI_MODEL = "gemini-2.0-flash"
ENABLE_TRANSFORMERS = "false"
```

The app still works if `GEMINI_API_KEY` is missing because it has local fallback logic.

## Optional transformer model

Transformer models are disabled by default for better Streamlit Cloud performance.

To enable optional transformer emotion detection:

1. Add these to `requirements.txt`:

```txt
transformers
torch
```

2. Set this secret:

```toml
ENABLE_TRANSFORMERS = "true"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
```

## Demo manager password

Set `MANAGER_PASSWORD` in secrets. The manager pages are protected.

## Notes

This app is for demo and learning purposes. Do not enter real private customer data, passwords, payment details, addresses, or confidential company records.
