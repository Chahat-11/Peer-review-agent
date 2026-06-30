import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    max_tokens=100,
    messages=[{"role": "user", "content": "Summarize in one sentence: Peer review is the cornerstone of academic knowledge validation."}]
)

print("✅ Groq API works!")
print("Response:", response.choices[0].message.content)