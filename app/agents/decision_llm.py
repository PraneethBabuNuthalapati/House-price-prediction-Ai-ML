import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def decide_mode(query: str) -> str:
    prompt = f"""
    You are an AI routing system.

Decide the intent of the user.

Return ONLY one word:
- "price" → if user only wants prediction
- "full" → if user wants explanation

User query:
{query}
    """
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    decision = response.choices[0].message.content.strip().lower()
    
    if "full" in decision:
        return "full"
    return "price"