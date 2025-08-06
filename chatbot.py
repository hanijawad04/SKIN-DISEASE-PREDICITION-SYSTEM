import httpx
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Loaded from .env
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def get_bot_response(user_input):
    try:
        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a friendly skin disease assistant. Answer user questions in simple, helpful language."},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }

        response = httpx.post(GROQ_API_URL, headers=headers, json=data, timeout=10.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()

    except Exception as e:
        return f"Sorry, something went wrong: {str(e)}"
