import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_explanation(top_features):
    if not top_features:
        return "The model did not identify strong feature impacts for this prediction."

    feature_text = "\n".join(
        [f"{f['feature']} → {round(f['impact'], 2)}" for f in top_features]
    )
    
    prompt = f"""
You are a real estate AI assistant.

Convert technical SHAP output into simple explanation.

DO NOT mention feature codes like 'cat__' or 'num__'.
Use natural language like:
- "kitchen quality"
- "neighborhood"

SHAP factors:
{feature_text}

Explain:
- why price is high/low
- keep it simple
- max 2–3 sentences
"""

    try:
        resonse = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resonse.choices[0].message.content
    except Exception:
        strongest = max(top_features, key=lambda f: abs(f.get("impact", 0)))
        impact = strongest.get("impact", 0)
        direction = "higher" if impact >= 0 else "lower"
        return (
            f"The strongest driver of this estimate is {strongest.get('feature', 'a key feature')}, "
            f"which pushes the predicted price {direction}. "
            "Other top features also contribute to the final value."
        )
