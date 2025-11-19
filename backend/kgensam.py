import os
import requests
import random

# ----------- Fuzzy Attribute Extraction for Chatbot -----------

def get_fuzzy_attributes(user_id):
    """
    Returns a list of ambiguous or relevant attributes for the user.
    In production, could query KG or feedback; for now, returns domain-relevant slots.
    """
    # These are the probable preferences/slots for Coimbatore food domain.
    return ["spice_level", "veg_nonveg", "cuisine", "area"]

def calculate_attribute_uncertainty(user_id, attribute):
    """
    Returns an uncertainty score [0,1] for an attribute (can be based on user feedback, KG stats, etc).
    Replace random with real computed logic for production.
    """
    # You can add MongoDB/Qdrant analysis here for real uncertainty.
    return round(random.uniform(0, 1), 3)

def explain_recommendation(user_id, food_id):
    """
    Provides a short, plain-language explanation for a recommendation.
    In production, chain KG/feedback/LLM info; here, it calls Groq LLM for a generic reason.
    """
    prompt = f"Explain concisely to user {user_id} why food {food_id} was recommended. Be direct, plain and user-friendly."
    try:
        reply = groq_chat_api(prompt, [])
        return reply
    except Exception as e:
        print("Groq/explanation API Failure:", e)
        return "Sorry, cannot explain this recommendation at the moment."

# ----------- Groq LLM API Wrapper (for chat/explanation) -----------

def groq_chat_api(prompt, dialog_history=[]):
    api_url = os.getenv("GROQ_URL")
    api_key = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")  # MOVE key to .env, never hardcode!

    # Prepare messages (convert "bot" to "assistant" for API)
    messages = []
    for item in dialog_history:
        if item.get("role") == "user" and item.get("content"):
            messages.append({"role": "user", "content": item["content"]})
        elif item.get("role") == "bot" and item.get("content"):
            messages.append({"role": "assistant", "content": item["content"]})
    messages.append({"role": "user", "content": prompt})

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": messages,
        "temperature": 0.85
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Groq API Error Response:", getattr(response, "text", None))
        return "AI explanation not available right now."
    return reply
