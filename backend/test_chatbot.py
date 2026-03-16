import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat = client.chat.completions.create(
    model="llama-3.1-8b-instant",   # ✅ supported
    messages=[{"role": "user", "content": "Say hello in one line"}],
    temperature=0.4,
    max_tokens=60
)

print(chat.choices[0].message.content)
