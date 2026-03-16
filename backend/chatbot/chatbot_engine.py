import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ✅ Allowed healthcare keywords (expand as needed)
HEALTH_KEYWORDS = [
    "pneumonia", "x-ray", "xray", "lungs", "lung", "cough", "fever", "breathing",
    "shortness of breath", "infection", "chest", "respiratory", "symptoms",
    "treatment", "medicine", "antibiotic", "doctor", "hospital", "diagnosis",
    "covid", "sars", "mri", "ct", "scan", "report", "gradcam", "risk", "prevention",
    "precautions", "healthy", "health", "pain", "fatigue", "oxygen", "spo2"
]

# ✅ Emergency red flags
EMERGENCY_KEYWORDS = [
    "severe chest pain", "not breathing", "unconscious", "blue lips",
    "very low oxygen", "spo2 below", "fainting", "blood cough", "collapse"
]

SYSTEM_PROMPT = """
You are Smart HealthAI Assistant — a healthcare-only assistant.
You must ONLY answer questions related to:
- Pneumonia, chest X-ray analysis, lung/respiratory health
- Symptoms, precautions, basic medical guidance
- Explaining AI diagnosis results (confidence, grad-cam)
- General wellness and patient care guidance

Strict rules:
1) If the user asks anything NOT related to healthcare, politely refuse.
2) Do not answer questions about politics, entertainment, coding, sports, or general topics.
3) Do NOT claim you are a doctor. Provide informational guidance only.
4) If the user describes emergency symptoms, tell them to seek urgent medical care.
5) Keep answers short, clear, and supportive.
"""

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def is_health_query(msg: str) -> bool:
    msg = normalize(msg)
    return any(k in msg for k in HEALTH_KEYWORDS)

def is_emergency(msg: str) -> bool:
    msg = normalize(msg)
    return any(k in msg for k in EMERGENCY_KEYWORDS)

def chatbot_response(user_message: str) -> str:
    if not user_message or not user_message.strip():
        return "Please enter a valid healthcare-related question."

    msg = normalize(user_message)

    # ✅ Emergency handling
    if is_emergency(msg):
        return (
            "⚠️ This may be an emergency. Please seek urgent medical care immediately "
            "or contact local emergency services."
        )

    # ✅ Strict healthcare-only filter
    if not is_health_query(msg):
        return (
            "I can help only with healthcare-related questions (pneumonia, chest X-rays, "
            "symptoms, precautions, diagnosis explanation). Please ask a medical question."
        )

    # ✅ Call Groq model
    try:
        chat = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=250
        )
        return chat.choices[0].message.content.strip()

    except Exception as e:
        print("Groq Chatbot Error:", repr(e))
        return "Assistant temporarily unavailable. Please try again later."
